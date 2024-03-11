# Evaluation

The crate includes accuracy and performance evaluations.
Below are instructions on how to run those.

_Note that all of these should be run from the crate's root directory!_

## Accuracy

See [accuracy results](evaluation/accuracy.md) for the predefined configurations.

The evaluation tool is available as a CLI to run experiments on predefined and custom configurations.

1. Ensure R and ImageMagick are installed with necssary packages:

   - Install R from [download](https://cran.r-project.org/) or using your platform's package manager.
   - Start `R` and install packages by executing `install.packages(c('dplyr', 'egg', 'ggplot2', 'readr', 'stringr'))`.
   - Install ImageMagick using the official [instructions](https://imagemagick.org/script/download.php).

2. Run the following in the `geo_filters` crate root to see the help text:

       script/accuracy -h

   For example, to compare the predefined diff count configurations, run:

       $ script/accuracy geo_diff_{7,13}
       +--------------------+
       + Running Simulation +
       +--------------------+

       Parameters:

          number of configs = 2
          number of samples = 500
          number of sets    = 120

       Running simulations:

          geo_diff_7 ...  done (0.70 s)
          geo_diff_13 ...  done (1.71 s)

       +--------------------+
       + Writing Results    +
       +--------------------+

          done

          csv file = accuracy.csv

       ...

       Output: accuracy.pdf

   Open `accuracy.pdf` to see the results.

To regenerate the plots in the repository, ensure ImageMagick is available, and run:

    script/generate-accuracy-plots

_Note that this will always re-run the experiments!_

## Performance

See [performance results](evaluation/performance.md) for the predefined configurations.

To reproduce the benchmark (note that this may take quite some time), run:

    script/performance

Open `../../target/criterion/report/index.html` to see the results.

To regenerate the plots in the repository, ensure ImageMagick is available, and run:

    script/generate-performance-plots

_Note that this will always re-run the benchmark!_
