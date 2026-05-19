# Evaluation

The crate includes an overhead evaluation to measure the efficiency of the set reconciliation algorithm.
Below are instructions on how to run the evaluation.

_Note that all of these should be run from the crate's root directory!_

## Overhead Evaluation

The overhead evaluation measures how many coded symbols are required to successfully decode set differences of various sizes.
The key metric is the **overhead multiplier**: the ratio of coded symbols needed to the actual diff size.

See [overhead results](evaluation/overhead.md) for the predefined configurations.

### Running the Evaluation

1. Ensure R and ImageMagick are installed with necessary packages:

   - Install R from [download](https://cran.r-project.org/) or using your platform's package manager.
   - Start `R` and install packages by executing `install.packages(c('dplyr', 'ggplot2', 'readr', 'stringr', 'scales'))`.
   - Install ImageMagick using the official [instructions](https://imagemagick.org/script/download.php).
   - Install Ghostscript for PDF conversion: `brew install ghostscript` (macOS) or `apt install ghostscript` (Linux).

2. Run the benchmark tool directly to see available options:

       cargo run --release --features bin --bin hriblt-bench -- --help

   Example: run 1000 trials with diff sizes from 1 to 100:

       cargo run --release --features bin --bin hriblt-bench -- \
           --trials 10000 \
           --set-size 1000 \
           --diff-size '1..101' \
           --diff-mode incremental \
           --tsv

3. To generate a plot from TSV output:

       cargo run --release --features bin --bin hriblt-bench -- \
           --trials 10000 --diff-size '1..101' --diff-mode incremental --tsv > overhead.tsv
       
       evaluation/plot-overhead.r overhead.tsv overhead.pdf

### TSV Output Format

When using `--tsv`, the benchmark outputs tab-separated data with the following columns:

| Column | Description |
|--------|-------------|
| `trial` | Trial number (1-indexed) |
| `set_size` | Number of elements in each set |
| `diff_size` | Number of differences between sets |
| `success` | Whether decoding succeeded (`true`/`false`) |
| `coded_symbols` | Number of coded symbols needed to decode |
| `overhead` | Ratio of coded_symbols to diff_size |

### Regenerating Documentation Plots

To regenerate the plots in the repository, ensure ImageMagick is available, and run:

    scripts/generate-overhead-plots

_Note that this will always re-run the benchmark!_

## Understanding the Results

The overhead plot shows percentiles of the overhead multiplier across different diff sizes:

- **Gray region**: Full range (p0 to p99) - min to ~max overhead observed
- **Blue region**: Interquartile range (p25 to p75) - where 50% of trials fall
- **Dark line**: Median (p50) - typical overhead

A lower overhead means more efficient encoding. An overhead of 1.0x would mean the number of coded symbols exactly equals the diff size (theoretical minimum).

Typical results show:
- Small diff sizes (1-10) have higher variance and overhead due to the probabilistic nature of the algorithm
- Larger diff sizes (50+) converge to a more stable overhead around 1.3-1.5x
- The algorithm successfully decodes 100% of trials when given up to 10x the diff size in coded symbols
