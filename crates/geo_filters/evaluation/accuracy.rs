use std::fs::File;
use std::path::PathBuf;

use clap::Parser;
use geo_filters::build_hasher::DefaultBuildHasher;
use geo_filters::config::VariableConfig;
use itertools::Itertools;
use once_cell::sync::Lazy;
use regex::{Captures, Regex};
use std::sync::Arc;

use geo_filters::diff_count::{GeoDiffCount, GeoDiffCount13, GeoDiffCount7};
use geo_filters::distinct_count::{GeoDistinctCount, GeoDistinctCount13, GeoDistinctCount7};
use geo_filters::evaluation::hll::{Hll, Hll14, Hll8, VariableHllConfig};
use geo_filters::evaluation::simulation::{
    geo_set_sizes, run_simulations, write_simulation_results, SimulationConfig,
    SimulationCountFactory,
};

const NUMBER_OF_SAMPLES: usize = 500;
const SET_SIZE_RATIO: f64 = 1.1;
const SET_SIZE_MAX: usize = 1_000_000;

fn main() {
    let accuracy = Accuracy::parse();
    accuracy.run();
}

#[derive(Parser)]
#[clap(after_help = "\x1b[1;4;37mConfigurations:\x1b[0;37m
  geo_diff/BUCKET_TYPE/b=N/bytes=N/msb=N
  geo_diff_7
  geo_diff_13
  geo_distinct/BUCKET_TYPE/b=N/bytes=N/msb=N
  geo_distinct_7
  geo_distinct_13
  hll/p=N
  hll_8
  hll_14

\x1b[1;4;37mBucket types:\x1b[0;37m
  u8, u16, u32, u64
")]
#[clap(after_long_help = "\x1b[1;4;37mConfigurations:\x1b[0;37m
  geo_diff/BUCKET_TYPE/b=N/bytes=N/msb=N      Diff count with the given parameters
  geo_diff_7                                  Predefined configuration for b=7
  geo_diff_13                                 Predefined configuration for b=13

  geo_distinct/BUCKET_TYPE/b=N/bytes=N/msb=N  Distinct count with the given parameters
  geo_distinct_7                              Predefined configuration for b=7
  geo_distinct_13                             Predefined configuration for b=13

  hll/p=N                                     HyperLogLog count with the given precision
  hll_8                                       Predefined configuration for p=8
  hll_14                                      Predefined configuration for p=14

  Note that the predefined configurations are much faster than configurations with
  explicitly specified parameters!

\x1b[1;4;37mBucket types:\x1b[0;37m
  u8, u16, u32, u64

\x1b[1;4;37mExample usage:\x1b[0;37m
  Compare different precisions for diff count:

      accuracy geo_diff/u16/b=7/bytes={16,32,64,128}/msb=10

  Compare predefined configurations of geo distinct count and HyperLogLog:

      accuracy geo_distinct_{7,13} hll_{8,14}

\x1b[1;4;37mPlotting results:\x1b[0;37m
  Results can be plotted using an R script. Ensure R and the `tidyverse` package are
  installed and run the following:

      evaluation/plot-accuracy.r accuracy.csv

  The plots will be saved to `accuracy.pdf`.
")]
struct Accuracy {
    #[clap(long, short = 'o', value_name = "FILE", default_value = "accuracy.csv")]
    output: PathBuf,

    #[clap(long, short = 'n', value_name = "COUNT", default_value_t = NUMBER_OF_SAMPLES)]
    samples: usize,

    #[clap(long, short = 'r', value_name = "RATIO", default_value_t = SET_SIZE_RATIO)]
    set_size_ratio: f64,

    #[clap(long, short = 'm', value_name = "SIZE", default_value_t = SET_SIZE_MAX)]
    set_size_max: usize,

    #[clap(long, short = 'S', value_name = "SIZE", conflicts_with_all = ["set_size_ratio", "set_size_max"])]
    set_size: Vec<usize>,

    #[clap(value_name = "CONFIG", required = true)]
    config: Vec<String>,
}

impl Accuracy {
    fn run(self) {
        if self.set_size_ratio <= 1.0 {
            panic!("set size ration must be > 1.0");
        }
        let configs = self
            .config
            .iter()
            .map(|c| {
                simulation_config_from_str(c)
                    .unwrap_or_else(|_| panic!("not a valid configuration: {}", c))
            })
            .collect_vec();
        let set_sizes = if self.set_size.is_empty() {
            geo_set_sizes(self.set_size_ratio, self.set_size_max)
        } else {
            self.set_size.into_iter().sorted().collect_vec()
        };
        let n = self.samples;
        let results = run_simulations(&configs, n, &set_sizes);

        let mut output = self.output;
        output.set_extension("csv");
        let f = File::create(&output)
            .unwrap_or_else(|_| panic!("cannot create file: {}", output.display()));
        write_simulation_results(&configs, &set_sizes, results, f)
            .unwrap_or_else(|_| panic!("cannot write file: {}", output.display()));
        println!("   csv file = {}", output.display());
        println!();
    }
}

struct SimulationConfigParser(
    Regex,
    Arc<dyn Fn(Captures) -> Box<SimulationCountFactory> + Send + Sync>,
);

impl SimulationConfigParser {
    fn new<F: Fn(Captures) -> Box<SimulationCountFactory> + Sync + Send + 'static>(
        re: &str,
        f: F,
    ) -> Self {
        Self(Regex::new(re).expect(""), Arc::new(f))
    }

    fn parse(&self, name: &str) -> Option<SimulationConfig> {
        self.0
            .captures(name)
            .map(self.1.as_ref())
            .map(|p| (name.to_string(), p))
    }
}

static SIMULATION_CONFIG_FROM_STR: Lazy<Vec<SimulationConfigParser>> = Lazy::new(|| {
    vec![
        SimulationConfigParser::new(r#"geo_diff/(\w+)/b=(\d+)/bytes=(\d+)/msb=(\d+)"#, |c| {
            let t = capture_bucket_type(&c, 1);
            let [b, bytes, msb] = capture_usizes(&c, [2, 3, 4]);
            match t {
                BucketType::U8 => {
                    let c = VariableConfig::<_, u8, DefaultBuildHasher>::new(b, bytes, msb);
                    Box::new(move || Box::new(GeoDiffCount::new(c.clone())))
                }
                BucketType::U16 => {
                    let c = VariableConfig::<_, u16, DefaultBuildHasher>::new(b, bytes, msb);
                    Box::new(move || Box::new(GeoDiffCount::new(c.clone())))
                }
                BucketType::U32 => {
                    let c = VariableConfig::<_, u32, DefaultBuildHasher>::new(b, bytes, msb);
                    Box::new(move || Box::new(GeoDiffCount::new(c.clone())))
                }
                BucketType::U64 => {
                    let c = VariableConfig::<_, u64, DefaultBuildHasher>::new(b, bytes, msb);
                    Box::new(move || Box::new(GeoDiffCount::new(c.clone())))
                }
            }
        }),
        SimulationConfigParser::new(r#"geo_diff_7"#, |_| {
            Box::new(|| Box::new(GeoDiffCount7::default()))
        }),
        SimulationConfigParser::new(r#"geo_diff_13"#, |_| {
            Box::new(|| Box::new(GeoDiffCount13::default()))
        }),
        SimulationConfigParser::new(r#"geo_distinct/(\w+)/b=(\d+)/bytes=(\d+)/msb=(\d+)"#, |c| {
            let t = capture_bucket_type(&c, 1);
            let [b, bytes, msb] = capture_usizes(&c, [2, 3, 4]);

            match t {
                BucketType::U8 => {
                    let c = VariableConfig::<_, u8, DefaultBuildHasher>::new(b, bytes, msb);
                    Box::new(move || Box::new(GeoDistinctCount::new(c.clone())))
                }
                BucketType::U16 => {
                    let c = VariableConfig::<_, u16, DefaultBuildHasher>::new(b, bytes, msb);
                    Box::new(move || Box::new(GeoDistinctCount::new(c.clone())))
                }
                BucketType::U32 => {
                    let c = VariableConfig::<_, u32, DefaultBuildHasher>::new(b, bytes, msb);
                    Box::new(move || Box::new(GeoDistinctCount::new(c.clone())))
                }
                BucketType::U64 => {
                    let c = VariableConfig::<_, u64, DefaultBuildHasher>::new(b, bytes, msb);
                    Box::new(move || Box::new(GeoDistinctCount::new(c.clone())))
                }
            }
        }),
        SimulationConfigParser::new(r#"geo_distinct_7"#, |_| {
            Box::new(|| Box::new(GeoDistinctCount7::default()))
        }),
        SimulationConfigParser::new(r#"geo_distinct_13"#, |_| {
            Box::new(|| Box::new(GeoDistinctCount13::default()))
        }),
        SimulationConfigParser::new(r#"hll/p=(\d+)"#, |c| {
            let [p] = capture_usizes(&c, [1]);
            Box::new(move || Box::new(Hll::new(VariableHllConfig::new(p))))
        }),
        SimulationConfigParser::new(r#"hll_8"#, |_| Box::new(|| Box::new(Hll8::default()))),
        SimulationConfigParser::new(r#"hll_14"#, |_| Box::new(|| Box::new(Hll14::default()))),
    ]
});

fn simulation_config_from_str(name: &str) -> Result<SimulationConfig, String> {
    SIMULATION_CONFIG_FROM_STR
        .iter()
        .find_map(|p| p.parse(name))
        .ok_or("invalid config".to_string())
}

fn capture_usizes<const N: usize>(c: &Captures, is: [usize; N]) -> [usize; N] {
    let mut values = [0; N];
    for i in 0..is.len() {
        values[i] = c
            .get(is[i])
            .expect("capture to exist")
            .as_str()
            .parse::<usize>()
            .expect("number string");
    }
    values
}

enum BucketType {
    U8,
    U16,
    U32,
    U64,
}

fn capture_bucket_type(c: &Captures, i: usize) -> BucketType {
    match c.get(i).expect("capture to exist").as_str() {
        "u8" => BucketType::U8,
        "u16" => BucketType::U16,
        "u32" => BucketType::U32,
        "u64" => BucketType::U64,
        t => panic!("bucket type must be one of u8, u16, u32, u64, got: {}", t),
    }
}
