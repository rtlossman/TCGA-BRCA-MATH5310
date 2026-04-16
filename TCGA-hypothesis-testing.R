#Deep Learning Models for Classification of RNA-seq Data
#TCGA Model Metric Visualization and Hypothesis Testing

#Rhys Lossman
#Georgetown University
#MATH 5310 Deep Learning
#April 23, 2026

library(car)
library(dplyr)
library(ggplot2)
library(scales)
library(tidyr)

fnn_results <- read.csv('mlp_architecture_search.csv')
fnn_results$arch <- "fnn"
fnn_long <- fnn_results %>%
  pivot_longer(
    cols = c("pca_val_acc", "ae_val_acc"),
    values_to = "val_acc"
  ) %>%
  select(-name)

cnn_results <- read.csv('cnn_architecture_search.csv')
cnn_results$arch <- "cnn"
cnn_long <- cnn_results %>%
  pivot_longer(
    cols = c("pca_val_acc", "ae_val_acc"),
    values_to = "val_acc"
  ) %>%
  select(-name)

rnn_results <- read.csv('rnn_architecture_search.csv')
rnn_results$arch <- "rnn"
rnn_long <- rnn_results %>%
  pivot_longer(
    cols = c("pca_val_acc", "ae_val_acc"),
    values_to = "val_acc"
  ) %>%
  select(-name)

#Hypothesis test 1: Dimensionality Reduction Method

pca <- c(fnn_results$pca_val_acc, cnn_results$pca_val_acc, rnn_results$pca_val_acc)
autoencoder <- c(fnn_results$ae_val_acc, cnn_results$ae_val_acc, rnn_results$ae_val_acc)
t.test(pca, autoencoder)

diff <- pca-autoencoder
shapiro.test(diff)

df_plot <- data.frame(
  group = c("PCA", "Autoencoder"),
  mean = c(mean(pca), mean(autoencoder))
)
ggplot(df_plot, aes(group, mean, fill=group)) +
  geom_col()+
  geom_text(aes(label=label_number(accurary=0.001)(mean), vjust=10))+
  labs(x="Method", y="Average Accuracy", title ="Validation accuracy by method") +
  theme_minimal()

#Hypothesis test 2: Model Class Effect

combined <- bind_rows(cnn_results, fnn_results, rnn_results)
combined <- combined %>% rename(pca_epochs = epochs_pca, ae_epochs=epochs_ae)

combined_long <- combined %>%
  pivot_longer(
    cols = c(pca_val_acc, ae_val_acc, pca_epochs, ae_epochs),
    names_to = c("model", ".value"),
    names_pattern = "(pca|ae)_(val_acc|epochs)"
  )
  
shapiro.test(combined_long$val_acc)
leveneTest(val_acc ~ arch, data=combined_long)

kruskal.test(val_acc ~ arch, data=combined_long)
pairwise.wilcox.test(combined_long$val_acc, combined_long$arch, p.adjust.method = "bonferroni")

mean(fnn_long$val_acc)
mean(cnn_long$val_acc)
mean(rnn_long$val_acc)

df_plot <- data.frame(
  group = c("FNN", "CNN", "RNN"),
  mean = c(mean(fnn_long$val_acc), mean(cnn_long$val_acc), mean(rnn_long$val_acc)))
ggplot(df_plot, aes(group, mean, fill=group)) +
  geom_col()+
  geom_text(aes(label=label_number(accurary=0.001)(mean), vjust=10))+
  labs(x="Model Type", y="Average Accuracy", title = "Validation accuracy by Architecture type" ) +
  theme_minimal()

#Hypothesis test 3: Model size effect

cor.test(combined_long$num_params, combined_long$val_acc, method = "spearman")
ggplot(combined_long, aes(x=num_params, y=val_acc, color = arch))+
  geom_point()+
  scale_x_log10()+
  labs(x="Number of parameters (log10)", 
       y="Validation accuracy",
       title = 'Validation accuracy vs. model size')

#Hypothesis test 4: Training duration effect 

cor.test(combined_long$epochs, combined_long$val_acc, method = "spearman")
ggplot(combined_long, aes(x=epochs, y=val_acc, color = arch))+
  geom_point()+
  labs(x="Training duration (epochs)", 
       y="Validation accuracy",
       title = 'Validation accuracy vs. training duration')
