library(dplyr)
library(ggplot2)

results_wide <- read.csv("./results.csv", header=F, stringsAsFactors = FALSE)

results_long <- data.frame()
for (row_ix in 1:nrow(results_wide)) {
  experiment_name <- results_wide[row_ix, 1]
  experiment_runtimes <- unname(unlist(results_wide[row_ix, -1]))
  
  experiment_results <- data.frame(runtime = experiment_runtimes,
                                   experiment_name = experiment_name,
                                   stringsAsFactors = FALSE)
  results_long <- rbind(results_long, experiment_results,
                        stringsAsFactors = FALSE)
}

parsed_results <- results_long %>%
  mutate(
    observations = gsub("([0-9]+)\\.svm:[0-9]+_[0-9]+\\.model:[a-zA-Z0-9]+", "\\1", experiment_name),
    n_trees = gsub("[0-9]+\\.svm:([0-9]+)_[0-9]+\\.model:[a-zA-Z0-9]+", "\\1", experiment_name),
    depth = gsub("[0-9]+\\.svm:[0-9]+_([0-9]+)\\.model:[a-zA-Z0-9]+", "\\1", experiment_name),
    lib = gsub("[0-9]+\\.svm:[0-9]+_[0-9]+\\.model:([a-zA-Z0-9]+)", "\\1", experiment_name)
  ) %>%
  mutate_at(c('observations', 'n_trees', 'depth'), 
            as.integer) %>%
  mutate(
    obs_per_milli = observations / runtime * 1000
  )

for (plot_tree_depth in unique(parsed_results$depth)) {
  p <- parsed_results %>%
    filter(depth == plot_tree_depth) %>%
    group_by(n_trees, depth, lib) %>%
    summarize(
      bound_025 = mean(log(obs_per_milli)) - 2 * sd(log(obs_per_milli)),
      bound_975 = mean(log(obs_per_milli)) + 2 * sd(log(obs_per_milli)),
      obs_per_milli = mean(log(obs_per_milli))
    ) %>%
    ggplot(aes(x=n_trees, y=obs_per_milli, color=lib)) +
      geom_line() +
      geom_point() +
      geom_errorbar(aes(ymin=bound_025, ymax=bound_975)) +
      ggtitle(paste0('For depth ', plot_tree_depth))
  ggsave(paste0('plots/depth', as.character(plot_tree_depth), '.png'), p)
}
