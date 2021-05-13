rm(list = ls())


metrics  <- c("micro", "macro", "experiment")
datasets <- c("bbn", "choi", "figer", "onto")
datasets.name <- list("bbn" = "BBN",
                      "choi" = "Choi",
                      "figer" = "FIGER",
                      "onto" = "OntoNotes")
training_datasets <- datasets  # c("bbn", "onto")
models   <- c("bert_ft_0", "bert_ft_2", "adapter_2", "adapter_16")
models <- factor(models, models)
models.name <- list("bert_ft_0"  = "Bert finetuned 0",
                    "bert_ft_2"  = "Bert finetuned 2",
                    "adapter_2"  = "Adapters rf = 2",
                    "adapter_16" = "Adapters rf = 16")
confidence <- 0.95
n <- 5


library(tidyverse)
library(gridExtra)


read_performance <- function(model, train_dataset, test_dataset) {
  file <- paste0(model,
                 "_trained_on_", train_dataset,
                 "_tested_on_", test_dataset,
                 ifelse(train_dataset != test_dataset, paste0("_filtered_with_", train_dataset), ""),
                 "_test.txt")
  filepath <- paste("../results/avgs_stds/", file, sep="")
  data <- read_csv(filepath)
  list("micro"         = filter(data, model == "micro_f1")[1, "mu", drop = TRUE],
       "micro.sd"      = filter(data, model == "micro_f1")[1, "sd", drop = TRUE],
       "macro"         = filter(data, model == "macro_f1")[1, "mu", drop = TRUE],
       "macro.sd"      = filter(data, model == "macro_f1")[1, "sd", drop = TRUE],
       "experiment"    = filter(data, model == "example_f1")[1, "mu", drop = TRUE],
       "experiment.sd" = filter(data, model == "example_f1")[1, "sd", drop = TRUE])
}


get_metric_name <- function(m)
  list("micro" = "micro f1",
       "macro" = "macro f1",
       "experiment" = "experimental micro f1")[[m]]



for (metric in metrics) {
  data <- tibble(model = c(),
                 dataset = c(),
                 measure = c(),
                 idc_low = c(),
                 idc_up  = c())
  for (train in training_datasets)
    for (m in models) {
      m.performance <- read_performance(m, train, train)
      z <- qt((1 - confidence) / 2, df = n - 1)
      d <- abs(z) * m.performance[[paste0(metric, ".sd")]] / sqrt(n)
      data <- data %>%
        add_row(model = models.name[[m]],
                dataset = datasets.name[[train]],
                measure = m.performance[[metric]],
                idc_low = m.performance[[metric]] - d,
                idc_up  = m.performance[[metric]] + d)
    }

  ggplot(data, aes(x = dataset,
                   y = measure,
                   fill = model)) +
    geom_bar(stat = "identity",
             position = position_dodge()) +
    geom_errorbar(aes(ymin = idc_low,
                      ymax = idc_up),
                  width = .2,
                  position = position_dodge(.9)) +
    theme_minimal() +
    ylim(0, 1) +
    xlab("Dataset") +
    ylab(get_metric_name(metric)) +
    scale_x_discrete() +
    scale_fill_brewer() -> p

  ggsave(filename = paste0("native_", metric, ".png"), p,
         device = png(),
         dpi = 600,
         height = 10, width = 15)
}


for (metric in metrics) {
  plots <- list()
  for (train in training_datasets) {
    data <- tibble(model = c(),
                   mapped = c(),
                   measure = c())
    for (d in datasets)
      for (m in models)
        data <- data %>%
          add_row(model = m,
                  mapped = d,
                  measure = read_performance(m, train, d)[[metric]])

    plots[[train]] <- ggplot(data, aes(x = mapped,
                                       y = measure,
                                       fill = model)) +
      geom_bar(stat="identity",
               position=position_dodge()) +
      theme_minimal() +
      ylim(0, 1) +
      ggtitle(paste0("Trained on ", train)) +
      xlab("Tested on ") +
      ylab(get_metric_name(metric)) +
      scale_x_discrete() +
      scale_fill_brewer()
  }


  grid.arrange(plots[[1]], plots[[2]],
               plots[[3]], plots[[4]],
               nrow = 2) %>%
    ggsave(filename = paste0(metric, ".png"), png(),
           dpi = 600,
           height = 10, width = 15)
}
