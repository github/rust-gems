//! Benchmarking tool for hriblt set reconciliation.
//!
//! This tool runs trials to measure the success rate of decoding set differences.

use std::{collections::HashSet, ops::Range, str::FromStr};

use clap::{Parser, ValueEnum};
use rand::prelude::*;

use hriblt::{DecodedValue, DecodingSession, DefaultHashFunctions, EncodingSession};

/// A diff size specification that can be a single value or a range.
#[derive(Debug, Clone)]
struct DiffSizeSpec {
    range: Range<u32>,
}

impl FromStr for DiffSizeSpec {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Try parsing as a range first (e.g., "1..10" or "1..=10")
        if let Some((start, end)) = s.split_once("..=") {
            let start: u32 = start.parse().map_err(|_| format!("invalid range start: {}", start))?;
            let end: u32 = end.parse().map_err(|_| format!("invalid range end: {}", end))?;
            if start > end {
                return Err(format!("range start {} must be <= end {}", start, end));
            }
            return Ok(DiffSizeSpec { range: start..end + 1 });
        }
        if let Some((start, end)) = s.split_once("..") {
            let start: u32 = start.parse().map_err(|_| format!("invalid range start: {}", start))?;
            let end: u32 = end.parse().map_err(|_| format!("invalid range end: {}", end))?;
            if start >= end {
                return Err(format!("range start {} must be < end {}", start, end));
            }
            return Ok(DiffSizeSpec { range: start..end });
        }
        // Otherwise parse as a single value
        let val: u32 = s.parse().map_err(|_| format!("invalid diff size: {}", s))?;
        Ok(DiffSizeSpec { range: val..val + 1 })
    }
}

/// How to iterate through the diff size range.
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
enum DiffSizeMode {
    /// Pick a random value from the range for each trial
    #[default]
    Random,
    /// Iterate through the range incrementally, looping if needed
    Incremental,
}

/// Iterator over diff sizes based on the mode.
enum DiffSizeIter {
    Random {
        range: Range<u32>,
    },
    Incremental {
        range: Range<u32>,
        current: u32,
    },
}

impl DiffSizeIter {
    fn new(spec: &DiffSizeSpec, mode: DiffSizeMode) -> Self {
        match mode {
            DiffSizeMode::Random => DiffSizeIter::Random {
                range: spec.range.clone(),
            },
            DiffSizeMode::Incremental => DiffSizeIter::Incremental {
                range: spec.range.clone(),
                current: spec.range.start,
            },
        }
    }

    fn next_diff_size<R: Rng + ?Sized>(&mut self, rng: &mut R) -> u32 {
        match self {
            DiffSizeIter::Random { range } => rng.random_range(range.clone()),
            DiffSizeIter::Incremental { range, current } => {
                let val = *current;
                *current += 1;
                if *current >= range.end {
                    *current = range.start;
                }
                val
            }
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "hriblt-bench")]
#[command(about = "Run reconciliation trials to measure decoding success rate")]
struct Args {
    /// Number of trials to run
    #[arg(short, long, default_value_t = 100)]
    trials: u32,

    /// Size of each set (number of elements)
    #[arg(short, long, default_value_t = 1000)]
    set_size: u32,

    /// Number of differences between the sets (single value or range like "1..10" or "1..=10")
    #[arg(short, long, default_value = "10")]
    diff_size: DiffSizeSpec,

    /// How to select diff sizes from a range
    #[arg(long, value_enum, default_value_t = DiffSizeMode::Random)]
    diff_mode: DiffSizeMode,

    /// Multiplier for max symbols to try (max_symbols = diff_size * multiplier)
    #[arg(short, long, default_value_t = 10)]
    multiplier: u32,

    /// Random seed (optional, for reproducibility)
    #[arg(long)]
    seed: Option<u64>,

    /// Print each trial as a TSV row to stdout
    #[arg(long)]
    tsv: bool,
}

/// Result of a single trial
struct TrialResult {
    success: bool,
    coded_symbols: Option<usize>,
}

fn run_trial<R: Rng + ?Sized>(
    rng: &mut R,
    set_size: u32,
    diff_size: u32,
    max_symbols: usize,
) -> TrialResult {
    // Ensure we have at least 32 symbols to work with
    let max_symbols = max_symbols.max(32);

    // Generate base set of random u64 values
    let base_set: HashSet<u64> = (0..set_size).map(|_| rng.random()).collect();

    // Create set A as the base set
    let set_a: Vec<u64> = base_set.iter().copied().collect();

    // Create set B by removing some elements and adding new ones
    let mut set_b: HashSet<u64> = base_set.clone();

    // Remove diff_size/2 elements from set B
    let removals = diff_size / 2;
    let additions = diff_size - removals;

    let mut to_remove: Vec<u64> = set_b.iter().copied().collect();
    to_remove.shuffle(rng);
    for val in to_remove.into_iter().take(removals as usize) {
        set_b.remove(&val);
    }

    // Add diff_size - removals new elements to set B
    for _ in 0..additions {
        loop {
            let new_val: u64 = rng.random();
            if !base_set.contains(&new_val) && set_b.insert(new_val) {
                break;
            }
        }
    }

    let set_b: Vec<u64> = set_b.into_iter().collect();

    // Create encoding sessions for both sets with max capacity
    let state = DefaultHashFunctions;

    let mut encoder_a = EncodingSession::new(state, 0..max_symbols);
    encoder_a.extend(set_a.iter().copied());

    let mut encoder_b = EncodingSession::new(state, 0..max_symbols);
    encoder_b.extend(set_b.iter().copied());

    // Merge the two encodings (negated to get the difference)
    let mut merged = encoder_a.merge(encoder_b, true);

    // Start with 1x the diff size, grow by 10% until success or max
    let mut current_symbols = (diff_size as usize).max(1);
    let mut decoding_session = DecodingSession::new(state);

    while current_symbols <= max_symbols {
        // Split off symbols up to current_symbols
        let chunk_start = decoding_session.consumed_coded_symbols();
        let chunk_end = current_symbols.min(max_symbols);

        if chunk_end > chunk_start {
            let chunk = merged.split_off(chunk_end - chunk_start);
            decoding_session.append(chunk);
        }

        if decoding_session.is_done() {
            let coded_symbols = decoding_session.consumed_coded_symbols();
            // Verify the decoded difference matches expected
            let decoded: HashSet<_> = decoding_session
                .into_decoded_iter()
                .map(|v| match v {
                    DecodedValue::Addition(x) | DecodedValue::Deletion(x) => x,
                })
                .collect();

            return TrialResult {
                success: decoded.len() == diff_size as usize,
                coded_symbols: Some(coded_symbols),
            };
        }

        // Grow by 10%, but at least 1
        let growth = (current_symbols / 10).max(1);
        current_symbols += growth;
    }

    TrialResult {
        success: false,
        coded_symbols: None,
    }
}

fn main() {
    let args = Args::parse();

    let mut rng: Box<dyn RngCore> = match args.seed {
        Some(seed) => Box::new(StdRng::seed_from_u64(seed)),
        None => Box::new(rand::rng()),
    };

    let is_range = args.diff_size.range.end - args.diff_size.range.start > 1;
    let range_desc = if is_range {
        format!("{}..{}", args.diff_size.range.start, args.diff_size.range.end)
    } else {
        format!("{}", args.diff_size.range.start)
    };

    eprintln!("Running {} trials...", args.trials);
    eprintln!("  Set size: {}", args.set_size);
    eprintln!("  Diff size: {} ({:?})", range_desc, args.diff_mode);
    eprintln!("  Max symbols multiplier: {}x", args.multiplier);
    eprintln!();

    if args.tsv {
        println!("trial\tset_size\tdiff_size\tsuccess\tcoded_symbols\toverhead");
    }

    let mut diff_iter = DiffSizeIter::new(&args.diff_size, args.diff_mode);

    let mut successes = 0;
    let mut failures = 0;

    for i in 0..args.trials {
        let diff_size = diff_iter.next_diff_size(&mut *rng);
        let max_symbols = (diff_size * args.multiplier) as usize;
        let result = run_trial(&mut *rng, args.set_size, diff_size, max_symbols);

        if args.tsv {
            let coded_symbols_str = result
                .coded_symbols
                .map(|n| n.to_string())
                .unwrap_or_default();
            let overhead_str = result
                .coded_symbols
                .map(|n| {
                    if diff_size > 0 {
                        format!("{:.2}", n as f64 / diff_size as f64)
                    } else {
                        String::new()
                    }
                })
                .unwrap_or_default();
            println!(
                "{}\t{}\t{}\t{}\t{}\t{}",
                i + 1,
                args.set_size,
                diff_size,
                result.success,
                coded_symbols_str,
                overhead_str
            );
        }

        if result.success {
            successes += 1;
        } else {
            failures += 1;
            if !args.tsv && failures <= 5 {
                eprintln!("Trial {} failed (diff_size={})", i + 1, diff_size);
            }
        }
    }

    eprintln!();
    eprintln!("Results:");
    eprintln!("  Successes: {}/{} ({:.1}%)", successes, args.trials, 
             100.0 * successes as f64 / args.trials as f64);
    eprintln!("  Failures:  {}/{} ({:.1}%)", failures, args.trials,
             100.0 * failures as f64 / args.trials as f64);
}
