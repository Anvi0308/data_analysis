#Loading the libraries
library(dplyr)
library(xgboost)
library(caret)
library(pROC)
library(scales)
library(doParallel)
library(plyr)
library(corrplot)
library(ggplot2)
library(gridExtra)
library(ggthemes)
library(MASS)
library(party)
library(ROSE)
library(tidyverse) 

# datasets
data <- read.csv('train.csv')
new_data <- read.csv('test.csv')

# Product Interaction: Clicks by Category
product_interaction <- train_data %>%
  select(starts_with("clicks"))

product_interaction_long <- tidyr::pivot_longer(product_interaction, cols = everything(), names_to = "Product_Category", values_to = "Clicks")

ggplot(product_interaction_long, aes(x = Product_Category, y = Clicks, fill = Product_Category)) +
  geom_bar(stat = "identity") +
  labs(title = "Clicks by Product Category",
       x = "Product Category", y = "Clicks",
       fill = "Product Category") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Temporal Patterns: Time of Day vs. User Activity
ggplot(train_data, aes(x = timeOfDay, y = visits)) +
  geom_line() +
  labs(title = "Temporal Patterns: Time of Day vs. User Activity",
       x = "Time of Day", y = "Visits") +
  theme_minimal()

# Grouping data by daysInactive and calculating average churn rate
engagement_summary <- train_data %>%
  group_by(daysInactive) %>%
  summarise(avg_churn = mean(churn))

# Creating a bar plot
ggplot(engagement_summary, aes(x = factor(daysInactive), y = avg_churn)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "User Engagement: Days Inactive vs. Churn Rate",
       x = "Days Inactive", y = "Average Churn Rate") +
  theme_minimal()



# Basic preprocessing
predictors <- names(data %>% select(-churn, -id) %>% select_if(is.numeric))
data[predictors] <- scale(data[predictors])
new_data[predictors] <- scale(new_data[predictors])

#Target variable-churn
labels <- as.numeric(as.factor(data$churn)) - 1
dataMatrix <- xgb.DMatrix(data = as.matrix(data[predictors]), label = labels)

# Handling class imbalance
class_ratio <- sum(labels == 0) / sum(labels == 1)

# Register parallel backend to use with caret for hyperparameter tuning
registerDoParallel(cores = 4) # Adjust based on your system

# Set up random search
set.seed(42)
search_grid <- expand.grid(
  eta = seq(0.01, 0.3, by = 0.05),
  max_depth = seq(3, 10, by = 1),
  subsample = seq(0.5, 1, by = 0.1),
  colsample_bytree = seq(0.5, 1, by = 0.1),
  gamma = seq(0, 5, by = 0.5),
  min_child_weight = seq(1, 6, by = 1),
  scale_pos_weight = list(class_ratio),
  lambda = seq(0.5, 2, by = 0.5),
  alpha = seq(0, 2, by = 0.5)
)

# Sample a subset of combinations for random search
sampled_indices <- sample(1:nrow(search_grid), 100)
search_grid <- search_grid[sampled_indices, ]

best_model <- NULL
best_auc <- 0
best_params <- list()

# Begin random search
for(i in 1:nrow(search_grid)) {
  params <- as.list(search_grid[i, ])
  params$objective <- "binary:logistic"
  params$eval_metric <- "auc"
  
  cv_results <- xgb.cv(
    params = params,
    data = dataMatrix,
    nrounds = 100,
    nfold = 5,
    showsd = TRUE,
    stratified = TRUE,
    print_every_n = 10,
    early_stopping_rounds = 10,
    maximize = TRUE
  )
  
  max_auc <- max(cv_results$evaluation_log$test_auc_mean)
  if(max_auc > best_auc) {
    best_auc <- max_auc
    best_params <- params
  }
}

# Retrain the model with the best parameters found
best_model <- xgboost(
  params = best_params,
  data = dataMatrix,
  nrounds = 100,
  verbose = 1
)

# Prepare test data
new_dataMatrix <- xgb.DMatrix(as.matrix(new_data[predictors]))

# Predict with the best model
new_predictions <- predict(best_model, new_dataMatrix)

# Create submission file
submission_data <- data.frame(id = new_data$id, churn = new_predictions)
write.csv(submission_data, 'submission.csv', row.names = FALSE)

