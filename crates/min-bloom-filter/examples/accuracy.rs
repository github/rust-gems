use min_bloom_filter::MinBloomFilter;

const ENTRY_COUNT: usize = 3_000_000;
const QUERY_COUNT: usize = 10_000_000;
const RATES: [(&str, f64); 2] = [("1%", 0.01), ("0.1%", 0.001)];
const GEOMETRIC_RATIOS: [f64; 2] = [0.5, 0.2];

fn main() {
    println!(
        "| Target FPP | Ratio | Nonzero FPP | Exact | Overestimated | Mean excess | Mean relative excess |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|");

    for (rate_name, rate) in RATES {
        for ratio in GEOMETRIC_RATIOS {
            let mut filter = MinBloomFilter::new(ENTRY_COUNT, rate);
            for index in 0..ENTRY_COUNT {
                filter.insert(hash(index), geometric_value(hash(index), ratio));
            }

            let mut false_positives = 0_usize;
            let mut exact = 0_usize;
            let mut overestimated = 0_usize;
            let mut excess = 0_u64;
            let mut relative_excess = 0.0;

            for index in 0..QUERY_COUNT {
                let key = index % ENTRY_COUNT;
                let key_hash = hash(key);
                let true_value = geometric_value(key_hash, ratio);
                let estimate = filter.get(key_hash);
                if estimate == true_value {
                    exact += 1;
                } else {
                    overestimated += 1;
                    excess += u64::from(estimate - true_value);
                    relative_excess += f64::from(estimate - true_value) / f64::from(true_value);
                }

                false_positives += usize::from(filter.get(hash(ENTRY_COUNT + index)) > 0);
            }

            println!(
                "| {rate_name} | {ratio:.1} | {:.4}% | {:.4}% | {:.4}% | {:.4} | {:.4}% |",
                false_positives as f64 / QUERY_COUNT as f64 * 100.0,
                exact as f64 / QUERY_COUNT as f64 * 100.0,
                overestimated as f64 / QUERY_COUNT as f64 * 100.0,
                excess as f64 / QUERY_COUNT as f64,
                relative_excess / QUERY_COUNT as f64 * 100.0,
            );
        }
    }
}

fn geometric_value(hash: u64, ratio: f64) -> u8 {
    let uniform = ((hash >> 11) as f64 + 0.5) / ((1_u64 << 53) as f64);
    let level = (uniform.ln() / ratio.ln()).floor() as u32;
    (level + 1).min(15) as u8
}

fn hash(index: usize) -> u64 {
    splitmix64(index as u64)
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
