use std::io::Write;
use std::time::Instant;

use itertools::Itertools;
use rand::{RngCore, SeedableRng};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

use crate::config::GeoConfig;
use crate::diff_count::GeoDiffCount;
use crate::distinct_count::GeoDistinctCount;
use crate::{Count, Diff, Distinct};

use super::hll::{Hll, HllConfig};

pub type SimulationConfig = (String, Box<SimulationCountFactory>);
pub type SimulationCountFactory = dyn Fn() -> Box<dyn SimulationCount> + Send + Sync;

pub trait SimulationCount {
    fn push_hash(&mut self, hash: u64);
    fn size(&self) -> f32;
    fn bytes_in_memory(&self) -> usize;
}
impl<C: GeoConfig<Diff> + Clone> SimulationCount for GeoDiffCount<'_, C> {
    fn push_hash(&mut self, hash: u64) {
        <Self as Count<_>>::push_hash(self, hash)
    }
    fn size(&self) -> f32 {
        <Self as Count<_>>::size(self)
    }
    fn bytes_in_memory(&self) -> usize {
        <Self as Count<_>>::bytes_in_memory(self)
    }
}
impl<C: GeoConfig<Distinct>> SimulationCount for GeoDistinctCount<'_, C> {
    fn push_hash(&mut self, hash: u64) {
        <Self as Count<_>>::push_hash(self, hash)
    }
    fn size(&self) -> f32 {
        <Self as Count<_>>::size(self)
    }
    fn bytes_in_memory(&self) -> usize {
        <Self as Count<_>>::bytes_in_memory(self)
    }
}
impl<C: HllConfig> SimulationCount for Hll<C> {
    fn push_hash(&mut self, hash: u64) {
        <Self as Count<_>>::push_hash(self, hash)
    }
    fn size(&self) -> f32 {
        <Self as Count<_>>::size(self)
    }
    fn bytes_in_memory(&self) -> usize {
        <Self as Count<_>>::bytes_in_memory(self)
    }
}

/// SimulationResult holds all relevant statistical information we care about.
#[derive(Clone)]
pub struct SimulationResult {
    pub relative_error: Stats,
    pub bytes_in_memory: Stats,
}

impl SimulationResult {
    pub fn relative_error(&self) -> f64 {
        self.relative_error.mean
    }

    pub fn relative_var(&self) -> f64 {
        self.relative_error.var
    }

    pub fn relative_stddev(&self) -> f64 {
        self.relative_error.std
    }

    pub fn upscaled_relative_stddev(&self) -> f64 {
        (self.relative_stddev().powf(2.0) + 1.0 / self.relative_error.n as f64).sqrt()
    }

    pub fn mean_space(&self) -> f64 {
        self.bytes_in_memory.mean
    }
}

pub type SimulationResults = Vec<Vec<SimulationResult>>;

pub fn run_simulations(
    configs: &[SimulationConfig],
    samples: usize,
    set_sizes: &[usize],
) -> SimulationResults {
    println!("+--------------------+");
    println!("+ Running Simulation +");
    println!("+--------------------+");
    println!();

    println!("Parameters:");
    println!();
    println!("   number of configs = {}", configs.len());
    println!("   number of samples = {}", samples);
    println!("   number of sets    = {}", set_sizes.len());
    println!();

    println!("Running simulations:");
    println!();
    let results = configs
        .iter()
        .map(|(name, f)| {
            print!("   {} ... ", name);
            std::io::stdout().flush().expect("stdout can be flushed");
            let t = Instant::now();
            let result = simulate(f, samples, set_sizes);
            println!(" done ({:.2} s)", t.elapsed().as_secs_f32());
            result
        })
        .collect_vec();
    println!();

    results
}

pub fn write_simulation_results<F: std::io::Write>(
    configs: &Vec<SimulationConfig>,
    set_sizes: &[usize],
    results: SimulationResults,
    mut f: F,
) -> std::io::Result<()> {
    println!("+--------------------+");
    println!("+ Writing Results    +");
    println!("+--------------------+");
    println!();

    write!(f, "set_size")?;
    for (name, _) in configs {
        for cat in ["relative_error", "bytes_in_memory"] {
            let prefix = format!("{name}:{cat}");
            write!(
                f,
                ";{prefix}.n;{prefix}.mean;{prefix}.var;{prefix}.std;{prefix}.median;{prefix}.min;{prefix}.max",
            )?;
        }
    }
    writeln!(f)?;
    for i in 0..set_sizes.len() {
        write!(f, "{}", set_sizes[i])?;
        for stats in &results {
            for stats in [&stats[i].relative_error, &stats[i].bytes_in_memory] {
                write!(
                    f,
                    ";{};{};{};{};{};{};{}",
                    stats.n, stats.mean, stats.var, stats.std, stats.median, stats.min, stats.max
                )?;
            }
        }
        writeln!(f)?;
    }
    println!("   done");
    println!();

    Ok(())
}

pub fn geo_set_sizes(factor: f64, max: usize) -> Vec<usize> {
    (1..)
        .map(|i| (10f64 * factor.powi(i)) as usize)
        .dedup()
        .take_while(|n| *n <= max)
        .collect_vec()
}

pub fn simulate<F: Fn() -> Box<dyn SimulationCount> + Send + Sync>(
    f: F,
    samples: usize,
    set_sizes: &[usize],
) -> Vec<SimulationResult> {
    // check that set_sizes are sorted!
    assert!(set_sizes.iter().tuple_windows().all(|(a, b)| a < b));
    (0..samples)
        .into_par_iter()
        .map(|_| {
            let mut t = f();
            let mut last_set_size = 0;
            let mut rnd = rand::rngs::StdRng::from_os_rng();
            set_sizes
                .iter()
                .map(move |set_size| {
                    (last_set_size..*set_size).for_each(|_| t.push_hash(rnd.next_u64()));
                    last_set_size = *set_size;
                    (
                        (t.size() as f64 / *set_size as f64) - 1.0,
                        t.bytes_in_memory() as f64,
                    )
                })
                .collect::<Vec<_>>()
        }) // [[(f64, f64)]] : for each sample, for each set size, two observations (relative error, bytes in memory)
        .collect::<Vec<_>>()
        .into_iter()
        .fold(
            vec![(Vec::with_capacity(samples), Vec::with_capacity(samples)); set_sizes.len()],
            |mut agg, observations_per_set_size| {
                observations_per_set_size.into_iter().enumerate().for_each(
                    |(set_size_idx, (relative_error, bytes_in_memory))| {
                        agg[set_size_idx].0.push(relative_error);
                        agg[set_size_idx].1.push(bytes_in_memory);
                    },
                );
                agg
            },
        ) // [([f64], [f64])] : for each set size, two lists of observations (relative error, bytes in memory)
        .into_iter()
        .map(|(relative_error, bytes_in_memory)| SimulationResult {
            relative_error: Stats::from_values(relative_error),
            bytes_in_memory: Stats::from_values(bytes_in_memory),
        }) // [(Stats, Stats)] : for each set size, two stats values
        .collect::<Vec<_>>()
}

#[derive(Clone, Debug)]
pub struct Stats {
    pub n: usize,
    pub mean: f64,
    pub var: f64,
    pub std: f64,
    pub median: f64,
    pub min: f64,
    pub max: f64,
}

impl Stats {
    pub fn from_values(mut values: Vec<f64>) -> Self {
        assert!(!values.is_empty(), "requires observations");
        values.sort_by(|x, y| x.total_cmp(y));
        let n = values.len();
        let sum: f64 = values.iter().sum();
        let mean = sum / n as f64;
        let var: f64 = values
            .iter()
            .map(|x| (x - mean).powf(2.0))
            .reduce(|x, y| x + y)
            .expect("has at least one value")
            / n as f64;
        let std = var.sqrt();
        let median = values[n / 2];
        let min = values[0];
        let max = values[n - 1];
        Self {
            n,
            mean,
            var,
            std,
            median,
            min,
            max,
        }
    }
}
