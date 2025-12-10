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
use std::io::{BufReader, BufWriter, Read, Write};
use std::time::Instant;

use pef::EliasFano;

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
    total_high_bits: u64,
    total_low_bits: u64,
    lists_by_size: [u64; 7], // 1, 2-10, 11-100, 101-1000, 1001-10000, 10001-100000, >100000
}

impl Stats {
    fn add_list(&mut self, len: usize, ef: &EliasFano) {
        self.num_lists += 1;
        self.total_postings += len as u64;
        self.total_input_bytes += (len * 4) as u64;
        self.total_high_bits += (ef.high_bits().len() * 8) as u64;
        self.total_low_bits += (ef.low_bits().len() * 8) as u64;

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
        let buckets = ["1", "2-10", "11-100", "101-1K", "1K-10K", "10K-100K", ">100K"];
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
        let ef_total = self.total_high_bits + self.total_low_bits;
        println!(
            "  EF high bits:      {:>10} bytes ({:>6.2} MB)",
            self.total_high_bits,
            self.total_high_bits as f64 / 1_000_000.0
        );
        println!(
            "  EF low bits:       {:>10} bytes ({:>6.2} MB)",
            self.total_low_bits,
            self.total_low_bits as f64 / 1_000_000.0
        );
        println!(
            "  EF total:          {:>10} bytes ({:>6.2} MB)",
            ef_total,
            ef_total as f64 / 1_000_000.0
        );
        println!();
        let compression_ratio = self.total_input_bytes as f64 / ef_total as f64;
        let bits_per_posting = (ef_total * 8) as f64 / self.total_postings as f64;
        println!("Compression ratio:   {:>10.2}x", compression_ratio);
        println!("Bits per posting:    {:>10.2}", bits_per_posting);
    }
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <input.docs> [--output <output.ef>]", args[0]);
        eprintln!();
        eprintln!("Encodes PISA posting lists with Elias-Fano encoding.");
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --output <file>  Write encoded data to file (optional)");
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = args
        .iter()
        .position(|x| x == "--output")
        .map(|i| args.get(i + 1))
        .flatten();

    println!("Reading PISA posting lists from: {}", input_path);

    let file = File::open(input_path)?;
    let mut reader = BufReader::with_capacity(1024 * 1024, file);

    // Read header
    let header1 = read_u32(&mut reader)?;
    let header2 = read_u32(&mut reader)?;
    println!("Header: [{}, {}]", header1, header2);
    println!("Number of documents: {}", header2);

    let mut stats = Stats::default();
    let mut encoded_data: Vec<EliasFano> = Vec::new();

    let start = Instant::now();
    let mut last_report = Instant::now();

    // Read and encode all posting lists
    while let Some(postings) = read_posting_list(&mut reader)? {
        if postings.is_empty() {
            continue;
        }

        // PISA format stores absolute doc IDs (already sorted)
        // The max value is the last ID (since they're sorted)
        // Add 1 because EliasFano expects exclusive max
        let max = postings.last().copied().unwrap_or(0) + 1;
        let len = postings.len() as u32;

        // Encode with Elias-Fano
        let ef = EliasFano::new(postings.iter().copied(), max, len);
        stats.add_list(postings.len(), &ef);

        if output_path.is_some() {
            encoded_data.push(ef);
        }

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

    // Optionally write encoded data
    if let Some(output) = output_path {
        println!("\nWriting encoded data to: {}", output);
        let file = File::create(output)?;
        let mut writer = BufWriter::new(file);

        // Simple format: for each posting list:
        // - u32: bits_per_value
        // - u32: high_bits length (in u64s)
        // - u32: low_bits length (in u64s)
        // - high_bits data
        // - low_bits data

        for ef in &encoded_data {
            writer.write_all(&ef.bits_per_value().to_le_bytes())?;
            writer.write_all(&(ef.high_bits().len() as u32).to_le_bytes())?;
            writer.write_all(&(ef.low_bits().len() as u32).to_le_bytes())?;
            for &word in ef.high_bits() {
                writer.write_all(&word.to_le_bytes())?;
            }
            for &word in ef.low_bits() {
                writer.write_all(&word.to_le_bytes())?;
            }
        }

        writer.flush()?;
        println!("Done!");
    }

    Ok(())
}
