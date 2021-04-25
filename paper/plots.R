rm(list = ls())


metrics  <- c("micro", "macro", "experiment")
datasets <- c("bbn", "choi", "figer", "onto")
training_datasets <- datasets  # c("bbn", "onto")
models   <- c("bert_ft_2", "adapter_2", "adapter_16")



library(tidyverse)
library(gridExtra)


read_performance <- function(model, train_dataset, mapped) {
  directory <- "../results/avgs_stds/"
  file <- ifelse(train_dataset == mapped,
                 paste(model, train_dataset, "test.txt",
                       sep = "_"),
                 paste(model, train_dataset, "on", train_dataset, "filtered_w", mapped, "test.txt",
                       sep = "_"))
  data <- read_csv(paste0(directory, file))
  list("micro"      = filter(data, model == "micro_f1")[1, "mu", drop = TRUE],
       "macro"      = filter(data, model == "macro_f1")[1, "mu", drop = TRUE],
       "experiment" = filter(data, model == "example_f1")[1, "mu", drop = TRUE])
}


get_metric_name <- function(m)
  list("micro" = "micro f1",
       "macro" = "macro f1",
       "experiment" = "experimental micro f1")[[m]]



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
      ggtitle(paste0("Trainied on ", train)) +
      xlab("Tested on ") +
      ylab(get_metric_name(metric)) +
      scale_x_discrete() +scale_fill_brewer()
  }


  grid.arrange(plots[[1]], plots[[2]],
               plots[[3]], plots[[4]],
               nrow = 2) %>%
    ggsave(filename = paste0(metric, ".png"), png(),
           dpi = 600,
           height = 10, width = 15)
}
