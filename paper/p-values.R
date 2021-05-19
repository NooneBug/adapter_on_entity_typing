rm(list = ls())

library(tidyverse)


n <- 5

read_performance <- function(model, train_dataset, test_dataset, filtered) {
  file <- paste0(model,
                 "_trained_on_", train_dataset,
                 "_tested_on_", test_dataset,
                 ifelse(filtered != test_dataset, paste0("_filtered_with_", filtered), ""),
                 "_test.txt")
  filepath <- paste("../results/avgs_stds/", file, sep="")
  data <- read_csv(filepath) %>%
    filter(model == "example_f1")
  return(list(mu = select(data, mu)[1, 1, drop = TRUE],
              sd = select(data, sd)[1, 1, drop = TRUE]))
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
               score   = c(),
               p       = c(),
               stars   = c())
datasets <- c("bbn", "onto", "figer", "choi")
models <- c("bert_ft_0", "bert_ft_2", "adapter_2", "adapter_16")
for (test_set in datasets)
  for (train_set in datasets) if (train_set != test_set)
    for (model in models) {
      native     <- read_performance(model, test_set,  test_set, train_set)
      non.native <- read_performance(model, train_set, test_set, train_set)
      p <- calc_confidence(non.native, native, n)
      data <- data %>%
        add_row(dataset = train_set,
                filter  = test_set,
                model   = model,
                score   = non.native$mu,
                perc    = round(perc(native, non.native), 2),
                p       = p,
                stars   = print_confidence(p))
    }


prepare_results <- function(score, perc, p, stars)
  paste0(round(score, 2), " (", round(perc, 2), ") ", stars)

prepare_rows <- function(data, model_label) {
  data.out <- data %>% filter(model == model_label) %>%
    mutate(model = prepare_results(score, perc, p, stars)) %>%
    select(-score, -perc, -p, -stars)
  data.colnames <- colnames(data.out)
  data.colnames[data.colnames == "model"] <- model_label
  colnames(data.out) <- data.colnames
  data.out
}

data_table <- unique(select(data, dataset, filter))

for (m in models) {
  data_table <- data_table %>%
    right_join(prepare_rows(data, m))
}



print_row <- function(row) {
  paste(reduce(row, ~paste(.x, .y , sep = " & ")), "\\\\")
}

filename <- "results.tex"
write(print_row(colnames(data_table)), filename)

for (r in 1:nrow(data_table)) {
  row <- print_row(data_table[r, , drop = FALSE])
  write(row, filename, append = TRUE)
}


data_table <- data %>% filter(model == "bert_ft_0") %>%
  mutate("bert_ft_0" = prepare_results(score, perc, p, stars))

data_table <- data %>% filter
