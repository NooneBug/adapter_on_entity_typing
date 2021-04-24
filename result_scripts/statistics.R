rm(list = ls())

library("tidyverse")

parameter_tags <- c("adapter_2",  "adapter_16",  "bert_ft_2")
datasets <- c("bbn", "onto")
indexes <- c("micro", "macro", "example")
N <- 5
confidence <- 0.95

test_dev <- c("test")
dataset = "bbn"


load_model <- function(x, dataset, test_dev) {
  filename <- paste(x, dataset, test_dev, sep = "_")
  read.csv(paste0("../results/avgs_stds/", filename, ".txt"))
  }

analyze_model <- function(data) {
  micro   <- filter(data, model == "micro_f1")
  macro   <- filter(data, model == "macro_f1")
  example <- filter(data, model == "example_f1")
  list(micro       = select(micro, mu)[1, 1],
       micro.sd    = select(micro, sd)[1, 1],
       macro       = select(macro, mu)[1, 1],
       macro.sd    = select(macro, sd)[1, 1],
       example     = select(example, mu)[1, 1],
       example.sd  = select(example, sd)[1, 1])
}


load_and_analyze <- compose(analyze_model, ~load_model(.x, dataset, test_dev))
models <- map(parameter_tags, load_and_analyze)
names(models) <- parameter_tags

get_mu <- function(m) m[[index]]
get_sd <- function(m) m[[paste0(index, ".sd")]]

get_p_value <- function(mod_1, mod_2, index, n = N) {
  t.val <- (get_mu(mod_1) - get_mu(mod_2)) / (get_sd(mod_1) / sqrt(n))
  1 - pt(t.val, df = n - 1)
}

results <- tibble(dataset = c(),
                  train_dev = c(),
                  index = c(),
                  model_1 = c(),
                  model_2 = c(),
                  model_1_mu = c(),
                  model_2_mu = c(),
                  p_value = c())

for (configuration in test_dev)
  for (dataset in datasets)
    for (index in indexes)
      for (m_1 in parameter_tags)
        for (m_2 in parameter_tags)
          if (m_1 != m_2)
            results <- results %>%
              add_row(dataset = dataset,
                      train_dev = configuration,
                      index = index,
                      model_1 = m_1,
                      model_2 = m_2,
                      model_1_mu = get_mu(models[[m_1]]),
                      model_2_mu = get_mu(models[[m_2]]),
                      p_value = get_p_value(models[[m_1]], models[[m_2]], index))


# quali sono gli indici statisticamente significativi?
results %>%
  filter(p_value < 1 - confidence) %>%
  View()
