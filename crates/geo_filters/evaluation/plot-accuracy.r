#!/usr/bin/env Rscript

## Plot accuracies from the given CSV to a PDF.

library(dplyr)
library(egg)
library(ggplot2)
library(readr)
library(stringr)

do_plot <- function(data, name) {
  error_n <- data[[paste0(name, ":relative_error.n")]][1]
  error_plot <-
    ggplot(data, aes(
      x = .data$set_size,
      y = .data[[paste0(name, ":relative_error.mean")]],
      ymin = .data[[paste0(name, ":relative_error.mean")]] - .data[[paste0(name, ":relative_error.std")]],
      ymax = .data[[paste0(name, ":relative_error.mean")]] + .data[[paste0(name, ":relative_error.std")]]
    )) +
    geom_line() +
    geom_ribbon(alpha = 0.5) +
    scale_x_log10(
      breaks = scales::breaks_log(),
      labels = scales::label_log()
    ) +
    scale_y_continuous(
      breaks = scales::breaks_extended(7)
    ) +
    ggtitle(paste0("Relative error mean with standard deviation for ", name, " (n = ", error_n, ")")) +
    xlab("set size") +
    ylab("relative error")

  error_std_plot <-
    ggplot(data, aes(
      x = .data$set_size,
      y = .data[[paste0(name, ":relative_error.std")]]
    )) +
    geom_line() +
    scale_x_log10(
      breaks = scales::breaks_log(),
      labels = scales::label_log()
    ) +
    scale_y_continuous(
      breaks = scales::breaks_extended(7),
      limits = c(0, NA)
    ) +
    ggtitle(paste0("Relative error standard deviation for ", name, " (n = ", error_n, ")")) +
    xlab("set size") +
    ylab("relative error")

  bytes_n <- data[[paste0(name, ":bytes_in_memory.n")]][1]
  bytes_plot <-
    ggplot(data, aes(
      x = .data$set_size,
      y = .data[[paste0(name, ":bytes_in_memory.mean")]],
      ymin = .data[[paste0(name, ":bytes_in_memory.min")]],
      ymax = .data[[paste0(name, ":bytes_in_memory.max")]]
    )) +
    geom_line() +
    geom_ribbon(alpha = 0.5) +
    scale_x_log10(
      breaks = scales::breaks_log(),
      labels = scales::label_log()
    ) +
    scale_y_continuous(
      breaks = scales::breaks_extended(7),
      limits = c(0, NA)
    ) +
    ggtitle(paste0("Bytes in memory for ", name, " (n = ", bytes_n, ")")) +
    xlab("set size") +
    ylab("bytes in memory")

  list(ggarrange(error_plot, error_std_plot, bytes_plot))
}

args <- commandArgs(trailingOnly = TRUE)
input <- "accuracy.csv"
if (length(args) > 0) {
  input <- args[1]
  if (length(args) > 1 || input %in% c("-h", "--help")) {
    cat("
Usage: plot-accuracy.r FILENAME

  Plot accuracy and memory footprint for given CSV output of the `accuracy` command.

  Example usage:

      plot-accuracy.r accuracy.csv

")
    quit(status = -1)
  }
}
output <- str_replace(input, "\\.csv$", ".pdf")

data <- read_delim(input, delim = ";", show_col_types = FALSE)
names <- unique(str_replace(str_subset(colnames(data), ".*:.*"), ":.*", ""))
plots <- list()
for (name in names) {
  plots <- c(plots, do_plot(data, name))
}
pdf(file = output, width = 11, height = 11)
plots
dev.off()

cat(paste0("Output: ", output, "\n"))
