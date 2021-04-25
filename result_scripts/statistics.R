rm(list = ls())


if (!"tidyverse" %in% installed.packages()) {
  install.packages("tidyverse")
}
library("tidyverse")

dataset = "figer"
parameter_tags <- c("adapter_2",  "adapter_16"
                    # ,  "bert_ft_2"
)
indexes <- c("micro", "example")
N <- 5
confidence <- 0.95
test_dev <- c("test")



load_model <- function(x, dataset, test_dev) {
  filename <- paste(x, dataset, test_dev, sep = "_")
  read_csv(paste0("../results/avgs_stds/", filename, ".txt"))
}

analyze_model <- function(data) {
  micro   <- filter(data, model == "micro_f1")
  macro   <- filter(data, model == "macro_f1")
  example <- filter(data, model == "example_f1")
  list(micro       = select(micro, mu)[1, 1, drop=TRUE],
       micro.sd    = select(micro, sd)[1, 1, drop=TRUE],
       macro       = select(macro, mu)[1, 1, drop=TRUE],
       macro.sd    = select(macro, sd)[1, 1, drop=TRUE],
       example     = select(example, mu)[1, 1, drop=TRUE],
       example.sd  = select(example, sd)[1, 1, drop=TRUE])
}


load_and_analyze_test <- compose(analyze_model, ~load_model(.x, dataset, "test"))
load_and_analyze_dev  <- compose(analyze_model, ~load_model(.x, dataset, "dev"))
models <- map(parameter_tags, load_and_analyze_test)   
names(models) <- parameter_tags

get_mu <- function(m) m[[index]]
get_sd <- function(m) m[[paste0(index, ".sd")]]

get_p_value <- function(mod_1, mod_2, index, n = N) {
  t.val <- (get_mu(mod_1) - get_mu(mod_2)) / (get_sd(mod_1) / sqrt(n))
  1 - pt(t.val, df = n - 1)
}

get_stars <- function(score) {
  if (score < .900) return("")
  if (score < .950) return("*")
  if (score < .990) return("**")
  if (score < .999) return("***")
}

results <- tibble(dataset = c(),
                  train_dev = c(),
                  index = c(),
                  model_1 = c(),
                  model_2 = c(),
                  model_1_mu = c(),
                  model_2_mu = c(),
                  p_value = c(),
                  stars = c())

for (configuration in test_dev)
  for (index in indexes)
    for (m_1 in parameter_tags)
      for (m_2 in parameter_tags)
        if (m_1 != m_2) {
          p_val <- get_p_value(models[[m_1]], models[[m_2]], index)
          results <- results %>%
            add_row(dataset = dataset,
                    train_dev = configuration,
                    index = index,
                    model_1 = m_1,
                    model_2 = m_2,
                    model_1_mu = get_mu(models[[m_1]]),
                    model_2_mu = get_mu(models[[m_2]]),
                    p_value = p_val,
                    stars = get_stars(1 - p_val))
        }

results

# H_0: i modelli sono uguali
# H_1: A > B
results %>%
  filter(p_value < 1 - confidence)
