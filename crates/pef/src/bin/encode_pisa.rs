//! Binary to encode PISA posting lists with Elias-Fano encoding.
//!
//! Usage:
//!   cargo run --release --bin encode_pisa -- <input.docs> [--output <output.ef>]
//!
//! The PISA format is:
//!   - Header: 2 x u32 (typically [1, num_documents])
//!   - For each posting list:
//!     - u32: length of the posting list
//!     - length x u32: document IDs (d-gaps, i.e., delta encoded)

use std::env;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::time::Instant;

use pef::optimal_bits_per_value;

/// Read a u32 from a reader (little-endian)
fn read_u32<R: Read>(reader: &mut R) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

/// Read a posting list from PISA format
fn read_posting_list<R: Read>(reader: &mut R) -> std::io::Result<Option<Vec<u32>>> {
    let mut len_buf = [0u8; 4];
    match reader.read_exact(&mut len_buf) {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let len = u32::from_le_bytes(len_buf) as usize;

    let mut postings = vec![0u32; len];
    for posting in &mut postings {
        *posting = read_u32(reader)?;
    }
    Ok(Some(postings))
}

/// Statistics about the encoding
#[derive(Default)]
struct Stats {
    num_lists: u64,
    total_postings: u64,
    total_input_bytes: u64,
    total_ef_bytes: u64,
    total_vbyte_bytes: u64,
    lists_by_size: [u64; 7], // 1, 2-10, 11-100, 101-1000, 1001-10000, 10001-100000, >100000
}

fn grouped_elias_fano_bits(postings: &[u32]) -> u32 {
    let mut start = 0;
    let mut total = 0;
    const LIMIT: u32 = 8 * 8;
    while start < postings.len() {
        let mut end = start + 1;
        let mut next_bits = 0;
        while end < postings.len() {
            let max = postings[end] - postings[start] + 1;
            let len = (end - start) as u32;
            next_bits = 4 + optimal_bits_per_value(max - len, len).0;
            if next_bits > LIMIT {
                next_bits = LIMIT;
                break;
            }
            end += 1;
        }
        total += next_bits;
        start = end;
    }
    total
}

/// Compute VByte encoded size for a posting list (d-gaps)
fn vbyte_size(postings: &[u32]) -> u64 {
    if postings.is_empty() {
        return 0;
    }
    let mut total = 0u64;
    let mut prev = 0u32;
    for &doc in postings {
        let gap = doc - prev;
        prev = doc;
        // VByte: 7 bits per byte, continue bit in MSB
        total += match gap {
            0..=127 => 1,
            128..=16383 => 2,
            16384..=2097151 => 3,
            2097152..=268435455 => 4,
            _ => 5,
        };
    }
    total
}

impl Stats {
    fn add_list(&mut self, postings: &[u32]) {
        let len = postings.len();
        self.num_lists += 1;
        self.total_postings += len as u64;
        self.total_input_bytes += (len * 4) as u64;

        // Compute Elias-Fano encoding size
        let ef_bits = grouped_elias_fano_bits(postings);
        self.total_ef_bytes += ef_bits.div_ceil(8) as u64;

        // Compute VByte encoding size
        self.total_vbyte_bytes += vbyte_size(postings);

        let bucket = match len {
            1 => 0,
            2..=10 => 1,
            11..=100 => 2,
            101..=1000 => 3,
            1001..=10000 => 4,
            10001..=100000 => 5,
            _ => 6,
        };
        self.lists_by_size[bucket] += 1;
    }

    fn print(&self) {
        println!("\n=== Encoding Statistics ===");
        println!("Total posting lists: {:>12}", self.num_lists);
        println!("Total postings:      {:>12}", self.total_postings);
        println!();
        println!("List size distribution:");
        let buckets = [
            "1", "2-10", "11-100", "101-1K", "1K-10K", "10K-100K", ">100K",
        ];
        for (i, &name) in buckets.iter().enumerate() {
            let count = self.lists_by_size[i];
            let pct = 100.0 * count as f64 / self.num_lists as f64;
            println!("  {:>10}: {:>10} ({:>5.1}%)", name, count, pct);
        }
        println!();
        println!("Space usage:");
        println!(
            "  Input (raw u32):   {:>10} bytes ({:>6.2} MB)",
            self.total_input_bytes,
            self.total_input_bytes as f64 / 1_000_000.0
        );
        println!(
            "  Elias-Fano:        {:>10} bytes ({:>6.2} MB)",
            self.total_ef_bytes,
            self.total_ef_bytes as f64 / 1_000_000.0
        );
        println!(
            "  VByte:             {:>10} bytes ({:>6.2} MB)",
            self.total_vbyte_bytes,
            self.total_vbyte_bytes as f64 / 1_000_000.0
        );
        println!();
        let ef_ratio = self.total_input_bytes as f64 / self.total_ef_bytes as f64;
        let vbyte_ratio = self.total_input_bytes as f64 / self.total_vbyte_bytes as f64;
        let ef_bits = (self.total_ef_bytes * 8) as f64 / self.total_postings as f64;
        let vbyte_bits = (self.total_vbyte_bytes * 8) as f64 / self.total_postings as f64;
        println!("Elias-Fano:  {:>5.2}x compression, {:>5.2} bits/posting", ef_ratio, ef_bits);
        println!("VByte:       {:>5.2}x compression, {:>5.2} bits/posting", vbyte_ratio, vbyte_bits);
    }
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <input.docs>", args[0]);
        eprintln!();
        eprintln!("Encodes PISA posting lists with Elias-Fano encoding.");
        std::process::exit(1);
    }

    let input_path = &args[1];

    println!("Reading PISA posting lists from: {}", input_path);

    let file = File::open(input_path)?;
    let mut reader = BufReader::with_capacity(1024 * 1024, file);

    // Read header
    let header1 = read_u32(&mut reader)?;
    let header2 = read_u32(&mut reader)?;
    println!("Header: [{}, {}]", header1, header2);
    println!("Number of documents: {}", header2);

    let mut stats = Stats::default();

    let start = Instant::now();
    let mut last_report = Instant::now();

    // Read and encode all posting lists
    while let Some(postings) = read_posting_list(&mut reader)? {
        if postings.is_empty() {
            continue;
        }

        // Encode with Elias-Fano
        // let ef = EliasFano::new(postings.iter().copied(), max, len);
        stats.add_list(&postings);

        // Progress report every second
        if last_report.elapsed().as_secs() >= 1 {
            print!(
                "\rProcessed {:>10} lists, {:>12} postings...",
                stats.num_lists, stats.total_postings
            );
            std::io::stdout().flush()?;
            last_report = Instant::now();
        }
    }

    let elapsed = start.elapsed();
    println!(
        "\rProcessed {:>10} lists, {:>12} postings in {:.2}s",
        stats.num_lists,
        stats.total_postings,
        elapsed.as_secs_f64()
    );

    stats.print();

    Ok(())
}
