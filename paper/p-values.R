rm(list = ls())

library(tidyverse)


n <- 5

read_performance <- function(file) {
  filepath <- paste("../results/avgs_stds/", file, sep="")
  print(filepath)
  if (file.exists(filepath)) {
    data <- read_csv(filepath) %>%
      filter(model == "example_f1")
    return(list(mu = select(data, mu)[1, 1, drop = TRUE],
                sd = select(data, sd)[1, 1, drop = TRUE]))
  } else return(list(mu = 0.0,
                     sd = 0.0))
}
read_performance.native <- function(model, train_dataset, filtered) {
  file <- paste(model, train_dataset, "on", train_dataset, "filtered_w", filtered, "test.txt",
                sep = "_")
  return(read_performance(file))
}
read_performance.non_native <- function(model, train_dataset, test_dataset) {
  directory <- "../results/avgs_stds/"
  file <- paste(model,
                train_dataset,
                ifelse(test_dataset %in% c("bbn", "onto"), "to", "on"),
                test_dataset,
                "test.txt",
                sep = "_")
  return(read_performance(file))
}


calc_confidence <- function(m1, m2, n) {
  if (m1$sd == 0) return(0.5)
  t <- (m1$mu - m2$mu) / (m1$sd / sqrt(n))
  return(pt(t, df = n - 1) / 2)
}

perc <- function(m1, m2) {
  (m2$mu - m1$mu) / m1$mu
}

print_confidence <- function(p) {
  p.abs <- abs(p)
  if (p.abs < 1 - 0.999) return("***")
  if (p.abs < 1 - 0.99)  return("**")
  if (p.abs < 1 - 0.95)  return("*")
  if (p.abs < 1-  0.9)   return(".")
  else                   return("")
}



data <- tibble(dataset = c(),
               filter  = c(),
               model   = c(),
               perc    = c(),
               p       = c(),
               stars   = c())
datasets <- c("bbn", "onto", "figer", "choi")
models <- c("bert_ft_0", "bert_ft_2", "adapter_2", "adapter_16")
for (test_set in datasets)
  for (train_set in datasets) if (train_set != test_set)
    for (model in models) {
      native     <- read_performance.native(model, train_set, test_set)
      non.native <- read_performance.non_native(model, train_set, test_set)
      p <- calc_confidence(non.native, native, n)
      data <- data %>%
        add_row(dataset = train_set,
                filter  = test_set,
                model   = model,
                perc    = round(perc(native, non.native), 2),
                p       = p,
                stars   = print_confidence(p))
    }
