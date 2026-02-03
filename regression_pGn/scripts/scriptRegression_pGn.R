rm(list=ls())
library(tidyverse)
library(glmnet)
library(randomForest)
library(xgboost)
library(ranger)
library(caret)
library(gridExtra)

# ==============================================================================
# === REGRESSION PROBLEM
# === HIGH-DIMENSIONAL (p > n)
# ==============================================================================

# load data
library(mixOmics)
data(liver.toxicity)
dim(liver.toxicity$gene)
dim(liver.toxicity$clinic)

X <- liver.toxicity$gene # 64 obs x 3116 genes
y <- liver.toxicity$clinic[,"ALB.g.dL."] # Target: ALB.g/dL 
hist(y)

d <- as.data.frame(X)
d$y <- y
n <- nrow(d)
p <- ncol(X)

# check for NA's
na_get = function(data){
  na_vars = sapply(data, function(col)sum(is.na(col)))
  na_vars = sort(na_vars[na_vars>0])
  na_vars = data.frame(variabile=names(na_vars),freq_assoluta=as.numeric(na_vars),
                       freq_relativa=round(as.numeric(na_vars)/nrow(data),4))
  na_vars
}
na_tab = na_get(d)
na_tab

# check for constant values 
const <- apply(d %>% select(-y), 2, function(x) length(unique(x))==1)
#table(const)
d <- d %>% select(-names(X)[const])

# ============================================================================
# === K-FOLD CROSS-VALIDATION CLASSIFICATION
# ============================================================================

# create 5 balanced folds: 4 train vs 1 test
# evaluate with accuracy, confusion matrix, F1, precision, log loss

# ----------------------------------------------------------------------------
# --- FOLDS AND METRICS
# ----------------------------------------------------------------------------

set.seed(123)
K <- 5
# stratified folds
fold_ids <- sample(rep(1:K, length.out = nrow(d)))

compute_metrics <- function(pred, actual) {
  rmse <- sqrt(mean((pred - actual)^2))
  mae <- mean(abs(pred - actual))
  r2 <- 1 - sum((actual - pred)^2) / sum((actual - mean(actual))^2)
  c(RMSE = rmse, MAE = mae, R2 = r2)
}

# Feature selection via correlations (frequently used in regression problems)
feature_selection_correlation <- function(X_train, y_train, top_n = 100) {
  correlations <- apply(X_train, 2, function(gene) {
    abs(cor(gene, y_train, use = "complete.obs"))
  })
  top_features <- names(sort(correlations, decreasing = TRUE)[1:top_n])
  return(top_features)
}

# Feature selection via variance
feature_selection_variance <- function(X_train, top_n = 100) {
  variances <- apply(X_train, 2, var, na.rm = TRUE)
  top_features <- names(sort(variances, decreasing = TRUE)[1:top_n])
  return(top_features)
}

# initialise results lists
predictions_all <- list()
selected_features_all <- list()

# ----------------------------------------------------------------------------
# --- START
# --- WITH INTERNAL FEATURE SELECTION
# ----------------------------------------------------------------------------

for (k in 1:K) {
  cat("--- FOLD", k, "/", K, "---\n")
  
  # split train/test
  test_idx <- which(fold_ids == k)
  train_idx <- which(fold_ids != k)
  train <- d[train_idx, ]
  test <- d[test_idx, ]

  # X and y  
  y_train <- train$y
  X_train <- train %>% select(-y)
  y_test <- test$y
  X_test <- test %>% select(-y)
  
  # feature selection (p > n)
  # top 100 genes via correlations
  selected_features <- feature_selection_correlation(X_train, y_train, top_n = 100)
  selected_features_all[[k]] <- selected_features
  
  # Riduci dataset
  X_train_selected <- X_train[, selected_features]
  X_test_selected <- X_test[, selected_features]
  X_train_mat <- as.matrix(X_train_selected)
  X_test_mat <- as.matrix(X_test_selected)
  
  # initialise predictions
  fold_predictions <- list()
  fold_predictions$y_true <- y_test
  fold_predictions$fold <- k
  
  # -----------------------------
  # ---   RIDGE REGRESSION    ---
  # -----------------------------
  cv_ridge <- cv.glmnet(X_train_mat, y_train, alpha = 0, nfolds = 5)
  ridge_pred <- predict(cv_ridge, newx = X_test_mat, s = "lambda.min")
  fold_predictions$ridge <- as.vector(ridge_pred)
  
  # ------------------
  # ---   LASSO    ---
  # ------------------
  cv_lasso <- cv.glmnet(X_train_mat, y_train, alpha = 1, nfolds = 5)
  lasso_pred <- predict(cv_lasso, newx = X_test_mat, s = "lambda.min")
  fold_predictions$lasso <- as.vector(lasso_pred)
  
  # ------------------------
  # ---   ELASTIC NET    ---
  # ------------------------
  cv_enet <- cv.glmnet(X_train_mat, y_train, alpha = 0.5, nfolds = 5)
  enet_pred <- predict(cv_enet, newx = X_test_mat, s = "lambda.min")
  fold_predictions$enet <- as.vector(enet_pred)
  
  # --------------------------
  # ---   RANDOM FOREST    ---
  # --------------------------
  train_rf <- data.frame(y = y_train, X_train_selected)
  rf_model <- randomForest(y ~ ., data = train_rf, ntree = 500, importance = FALSE)
  fold_predictions$rf <- predict(rf_model, newdata = X_test_selected)
  
  # -------------------
  # ---   RANGER    ---
  # -------------------
  ranger_model <- ranger(y ~ ., data = train_rf, num.trees = 500)
  fold_predictions$ranger <- predict(ranger_model, data = as.data.frame(X_test_selected))$predictions
  
  # --------------------
  # ---   XGBOOST    ---
  # --------------------
  dtrain <- xgb.DMatrix(data = X_train_mat, label = y_train)
  dtest <- xgb.DMatrix(data = X_test_mat, label = y_test)
  
  params_xgb <- list(
    objective = "reg:squarederror",
    learning_rate = 0.05,
    max_depth = 3,
    subsample = 0.8,
    colsample_bytree = 0.8
  )
  
  cv_xgb <- xgb.cv(
    params = params_xgb,
    data = dtrain,
    nrounds = 200,
    nfold = 5,
    early_stopping_rounds = 20,
    verbose = 0
  )
  
  best_iter <- ifelse(!is.null(cv_xgb$best_iteration) && cv_xgb$best_iteration > 0,
                      cv_xgb$best_iteration, 200)
  
  xgb_model <- xgb.train(
    params = params_xgb,
    data = dtrain,
    nrounds = best_iter,
    verbose = 0
  )
  
  fold_predictions$xgb <- predict(xgb_model, dtest)
  
  predictions_all[[k]] <- fold_predictions
}

# ----------------------------------------------------------------------------
# --- RESULTS AGGREGATION
# ----------------------------------------------------------------------------

# Models list
model_names <- c("ridge", "lasso", "enet", "rf", "ranger", "xgb")

# Calculate metrics combining folds
cv_summary <- map_dfr(model_names, function(model) {
  
  # all predictions
  all_preds <- map(predictions_all, ~ .x[[model]]) %>% unlist()
  all_true <- map(predictions_all, ~ .x$y_true) %>% unlist()
  
  # global metrics
  global_metrics <- compute_metrics(all_preds, all_true)
  
  # metrics per fold
  fold_metrics <- map_dfr(1:K, function(k) {
    preds <- predictions_all[[k]][[model]]
    true <- predictions_all[[k]]$y_true
    metrics <- compute_metrics(preds, true)
    as_tibble_row(metrics)
  })
  
  tibble(
    Model = model,
    RMSE_mean = mean(fold_metrics$RMSE),
    RMSE_sd = sd(fold_metrics$RMSE),
    MAE_mean = mean(fold_metrics$MAE),
    MAE_sd = sd(fold_metrics$MAE),
    R2_mean = mean(fold_metrics$R2),
    R2_sd = sd(fold_metrics$R2),
    RMSE_global = global_metrics["RMSE"]
  )
}) %>%
  arrange(RMSE_mean)

model_labels <- c(
  ridge = "Ridge Regression",
  lasso = "Lasso Regression",
  enet = "Elastic Net",
  rf = "Random Forest",
  ranger = "Ranger RF",
  xgb = "XGBoost"
)

cv_summary <- cv_summary %>%
  mutate(Model_Label = model_labels[Model])

# Results
print(cv_summary %>% 
        select(Model_Label, RMSE_mean, RMSE_sd, MAE_mean, R2_mean) %>%
        rename(Model = Model_Label),
      n = Inf)

# best model
best_model <- cv_summary$Model_Label[1]
cat("\n Best model:", best_model, "\n")
cat("Mean RMSE:", round(cv_summary$RMSE_mean[1], 4), "±", 
    round(cv_summary$RMSE_sd[1], 4), "\n")
cat("Mean R2:", round(cv_summary$R2_mean[1], 4), "\n")

# ----------------------------------------------------------------------------
# --- FEATURE SELECTION ANALYSIS
# ----------------------------------------------------------------------------

# Most stable features 
all_selected <- unlist(selected_features_all)
feature_freq <- table(all_selected)
feature_stability <- sort(feature_freq / K, decreasing = TRUE)

# 20 most stable genes (often selected)
print(head(feature_stability, 20))

# Plot stability
stable_features_df <- tibble(
  Gene = names(feature_stability),
  Frequency = as.vector(feature_stability)
) %>%
  head(30)

p_stability <- ggplot(stable_features_df, aes(x = reorder(Gene, Frequency), y = Frequency)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Gene Selection Stability",
       x = "Gene", y = "Frequency (of 5 folds)") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

print(p_stability)

# ----------------------------------------------------------------------------
# --- VISUALS
# ----------------------------------------------------------------------------

# Plot 1: Confronto RMSE
p1 <- cv_summary %>%
  mutate(Model_Label = fct_reorder(Model_Label, RMSE_mean)) %>%
  ggplot(aes(x = Model_Label, y = RMSE_mean)) +
  geom_col(fill = "steelblue", alpha = 0.7) +
  geom_errorbar(aes(ymin = RMSE_mean - RMSE_sd, 
                    ymax = RMSE_mean + RMSE_sd),
                width = 0.3, color = "darkred") +
  coord_flip() +
  labs(title = "Model comparison - Average RMSE (p >> n)",
       x = NULL, y = "Average RMSE") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

print(p1)

# Plot 2: R2 comparison
p2 <- cv_summary %>%
  mutate(Model_Label = fct_reorder(Model_Label, R2_mean)) %>%
  ggplot(aes(x = Model_Label, y = R2_mean)) +
  geom_col(fill = "darkgreen", alpha = 0.7) +
  geom_errorbar(aes(ymin = R2_mean - R2_sd, 
                    ymax = R2_mean + R2_sd),
                width = 0.3, color = "darkred") +
  coord_flip() +
  labs(title = "R2 Score Comparison",
       x = NULL, y = "Average R2") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

print(p2)

# ----------------------------------------------------------------------------
# --- RE-RUNNING OF BEST MODEL ON ENTIRE DATASET: RIDGE REGRESSION
# ----------------------------------------------------------------------------

# Feature selection on the entire dataset
best_features <- feature_selection_correlation(d %>% select(-y), d$y, top_n = 100)
X_final <- as.matrix(d[, best_features])
y_final <- d$y

best_model_code <- cv_summary$Model[3]

final_model <- switch(
  best_model_code,
  ridge = cv.glmnet(X_final, y_final, alpha = 0, nfolds = 10),
  lasso = cv.glmnet(X_final, y_final, alpha = 1, nfolds = 10),
  enet = cv.glmnet(X_final, y_final, alpha = 0.5, nfolds = 10),
  rf = {
    train_data <- data.frame(y = y_final, as.data.frame(X_final))
    randomForest(y ~ ., data = train_data, ntree = 500, importance = TRUE)
  },
  ranger = {
    train_data <- data.frame(y = y_final, as.data.frame(X_final))
    ranger(y ~ ., data = train_data, num.trees = 500, importance = "impurity")
  },
  xgb = {
    dtrain <- xgb.DMatrix(data = X_final, label = y_final)
    params <- list(objective = "reg:squarederror", learning_rate = 0.05,
                   max_depth = 3, subsample = 0.8, colsample_bytree = 0.8)
    cv_res <- xgb.cv(params = params, data = dtrain, nrounds = 200,
                     nfold = 5, early_stopping_rounds = 20, verbose = 0)
    best_n <- ifelse(!is.null(cv_res$best_iteration), cv_res$best_iteration, 200)
    xgb.train(params = params, data = dtrain, nrounds = best_n, verbose = 0)
  }
)

# save
saveRDS(final_model, paste0("final_highdim_regression_", best_model_code, ".rds"))
saveRDS(best_features, "selected_features_regression.rds")

# ----------------------------------------------------------------------------
# --- PRESENTING THE RESULTS
# ----------------------------------------------------------------------------
library(pheatmap)

# PERFORMANCE

train_pred <- predict(final_model, newx = X_final, s = "lambda.min")
train_pred <- as.vector(train_pred)
train_metrics <- compute_metrics(train_pred, y_final)

cat("RMSE:", round(train_metrics["RMSE"], 4), "\n")
cat("MAE:", round(train_metrics["MAE"], 4), "\n")
cat("R2:", round(train_metrics["R2"], 4), "\n")

cat("Cross-Validation (5-fold):\n")
ridge_cv <- cv_summary %>% filter(Model == "ridge")
cat("Mean RMSE:", round(ridge_cv$RMSE_mean, 4), "±", round(ridge_cv$RMSE_sd, 4), "\n")
cat("Mean MAE:", round(ridge_cv$MAE_mean, 4), "\n")
cat("Mean R2:", round(ridge_cv$R2_mean, 4), "\n")

# OBSERVED vs PREDICTED

pred_df <- tibble(
  Observed = y_final,
  Predicted = train_pred,
  Residual = Observed - Predicted
)

# Scatter plot
p_obs_pred <- ggplot(pred_df, aes(x = Observed, y = Predicted)) +
  geom_point(alpha = 0.6, size = 3, color = "steelblue") +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed", size = 1) +
  geom_smooth(method = "lm", se = TRUE, color = "darkblue", alpha = 0.2) +
  labs(title = "Observed vs Predicted - Ridge Regression",
       subtitle = paste0("R2 = ", round(train_metrics["R2"], 3), 
                         " | RMSE = ", round(train_metrics["RMSE"], 3)),
       x = "Observed Albumin (g/dL)", 
       y = "Predicted Albumin (g/dL)") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(face = "bold", size = 16))

print(p_obs_pred)

# Residual plot
p_residual <- ggplot(pred_df, aes(x = Predicted, y = Residual)) +
  geom_point(alpha = 0.6, size = 3, color = "steelblue") +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed", size = 1) +
  geom_smooth(method = "loess", se = TRUE, color = "darkblue", alpha = 0.2) +
  labs(title = "Residual Plot",
       x = "Predicted Albumin (g/dL)", 
       y = "Residuals") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(face = "bold"))

print(p_residual)

# Residuals histogram
p_hist_res <- ggplot(pred_df, aes(x = Residual)) +
  geom_histogram(bins = 20, fill = "steelblue", alpha = 0.7, color = "black") +
  geom_vline(xintercept = 0, color = "red", linetype = "dashed", size = 1) +
  labs(title = "Distribution of Residuals",
       x = "Residuals", y = "Frequency") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(face = "bold"))

print(p_hist_res)

# Q-Q plot
p_qq <- ggplot(pred_df, aes(sample = Residual)) +
  stat_qq(color = "steelblue", size = 2) +
  stat_qq_line(color = "red", linetype = "dashed", size = 1) +
  labs(title = "Q-Q Plot - Normality of Residuals",
       x = "Theoretical Quantiles", 
       y = "Sample Quantiles") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(face = "bold"))

print(p_qq)

# COEFFICIENTS AND FEATURES IMPORTANCE

coef_matrix <- coef(final_model, s = "lambda.min")
coef_values <- as.vector(coef_matrix)[-1] 
coef_names <- rownames(coef_matrix)[-1]
cat("Intercept:", round(coef_matrix[1,1], 4), "\n")

# ! No sparsity with Ridge: all genes have non-zero coefficients
coef_df <- tibble(
  Gene = coef_names,
  Coefficient = coef_values,
  Abs_Coefficient = abs(coef_values)
) %>%
  arrange(desc(Abs_Coefficient)) %>%
  mutate(
    Direction = ifelse(Coefficient > 0, "Increase Albumin", "Decrease Albumin"),
    Rank = row_number()
  )

# 20 most important genes
print(coef_df %>% select(Gene, Coefficient, Direction) %>% head(20), n = 20)

# Plot coefficients - Top 20
p_coef_top <- coef_df %>%
  head(20) %>%
  ggplot(aes(x = reorder(Gene, Abs_Coefficient), y = Coefficient, fill = Direction)) +
  geom_col() +
  coord_flip() +
  scale_fill_manual(values = c("Increase Albumin" = "darkgreen", 
                               "Decrease Albumin" = "darkred")) +
  labs(title = "Top 20 Gene Importance - Ridge Regression",
       subtitle = paste0("R2 CV = ", round(ridge_cv$R2_mean, 3)),
       x = NULL, y = "Coefficient",
       fill = "Effect") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold", size = 14),
        axis.text.y = element_text(size = 10))

print(p_coef_top)

# Plot coefficients - All non-zero
p_coef_dist <- ggplot(coef_df, aes(x = Coefficient)) +
  geom_histogram(bins = 50, fill = "steelblue", alpha = 0.7, color = "black") +
  geom_vline(xintercept = 0, color = "red", linetype = "dashed", size = 1) +
  labs(title = "Distribution of all coefficients",
       subtitle = paste(nrow(coef_df), "genes"),
       x = "Coefficient", y = "Frequency") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

print(p_coef_dist)

# Coefficients ordered by rank
p_coef_rank <- ggplot(coef_df, aes(x = Rank, y = Coefficient, color = Direction)) +
  geom_point(alpha = 0.6, size = 2) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  scale_color_manual(values = c("Increase Albumin" = "darkgreen", 
                                "Decrease Albumin" = "darkred")) +
  labs(title = "Ridge Coefficients by Rank",
       x = "Rank (by absolute value)", y = "Coefficient") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

print(p_coef_rank)

# REGULARIZATION PATH

# Plot coefficients by lambda values
plot(final_model$glmnet.fit, xvar = "lambda", label = TRUE,
     main = "Ridge Regularization Path")
abline(v = log(final_model$lambda.min), col = "red", lty = 2)
abline(v = log(final_model$lambda.1se), col = "blue", lty = 2)
legend("topright", legend = c("lambda.min", "lambda.1se"), 
       col = c("red", "blue"), lty = 2)

# CV MSE plot
plot(final_model, main = "Cross-Validation MSE vs Lambda")

# FEATURE STABILITY ACROSS FOLDS

all_selected <- unlist(selected_features_all)
feature_freq <- table(all_selected)
feature_stability <- sort(feature_freq / K, decreasing = TRUE)

# 20 most stable genes in feature selection before ridge
print(head(feature_stability, 20))
# comparison with selected features by final_model
stable_genes <- names(feature_stability)[feature_stability >= 0.8] # at least 4/5 folds
cat("Stable genes:", length(stable_genes), "\n")

# Plot stability
stable_df <- tibble(
  Gene = names(feature_stability),
  Stability = as.vector(feature_stability)
) %>%
  head(30)

p_stability <- ggplot(stable_df, aes(x = reorder(Gene, Stability), y = Stability)) +
  geom_col(fill = "steelblue") +
  geom_hline(yintercept = 0.8, color = "red", linetype = "dashed") +
  coord_flip() +
  labs(title = "Top 30 Gene Selection Stability",
       subtitle = "Proportion of CV folds in which the feature was selected",
       x = "Gene", y = "Stability (0-1)") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

print(p_stability)

# HEATMAP: MOST IMPORTANT GENES

# Heatmap - top 30 genes by coefficients
top_genes <- coef_df$Gene[1:min(30, nrow(coef_df))]
heatmap_data <- t(X_final[, top_genes])

if ("treatment" %in% names(liver.toxicity)) {
  annotation_col <- data.frame(
    Treatment = liver.toxicity$treatment$Treatment.Group
  )
  rownames(annotation_col) <- rownames(X_final)
} else {
  annotation_col <- data.frame(
    Albumin = cut(y_final, breaks = 3, labels = c("Low", "Medium", "High"))
  )
  rownames(annotation_col) <- rownames(X_final)
}

# Plot
pheatmap(heatmap_data,
         annotation_col = annotation_col,
         scale = "row",
         clustering_distance_rows = "correlation",
         clustering_distance_cols = "euclidean",
         show_colnames = FALSE,
         main = "30 most influent genes on Albumin",
         fontsize = 10,
         fontsize_row = 8)

# INTERPRETATION

# Genes that increase Albumin
increase_genes <- coef_df %>% filter(Direction == "Increase Albumin") %>% head(10)
for (i in 1:nrow(increase_genes)) {
  cat("  ", i, ". ", increase_genes$Gene[i], " (β = ", 
      round(increase_genes$Coefficient[i], 5), ")\n", sep = "")
}

# Genes that decrease Albumin
decrease_genes <- coef_df %>% filter(Direction == "Decrease Albumin") %>% head(10)
for (i in 1:nrow(decrease_genes)) {
  cat("  ", i, ". ", decrease_genes$Gene[i], " (β = ", 
      round(decrease_genes$Coefficient[i], 5), ")\n", sep = "")
}

# PREDICTION CONFIDENCE

pred_df <- pred_df %>%
  mutate(
    Abs_Error = abs(Residual),
    Rel_Error = abs(Residual) / Observed * 100
  )

cat("MAE:", round(mean(pred_df$Abs_Error), 4), "\n")
cat("Mean relative error:", round(mean(pred_df$Rel_Error), 2), "%\n")
cat("Max absolute error:", round(max(pred_df$Abs_Error), 4), "\n\n")

# Worst predictions - top 5
worst <- pred_df %>%
  arrange(desc(Abs_Error)) %>%
  select(Observed, Predicted, Residual, Abs_Error) %>%
  head(5)
print(worst)

# Best predictions - top 5
best <- pred_df %>%
  arrange(Abs_Error) %>%
  select(Observed, Predicted, Residual, Abs_Error) %>%
  head(5)
print(best)

# SUMMARY

summary_table <- tibble(
  Metric = c(
    "Modello",
    "Alpha (Ridge)",
    "Lambda min",
    "Genes in the dataset",
    "Genes after feature selection",
    "Genes used by Ridge",
    "Training RMSE",
    "Training R2",
    "CV RMSE (mean)",
    "CV RMSE (sd)",
    "CV R2 (mean)",
    "CV MAE (mean)",
    "Most important gene (increase Albumin)",
    "Most important gene (decrease Albumin)"
  ),
  Value = c(
    "Ridge Regression",
    "0 (L2 only)",
    format(final_model$lambda.min, scientific = TRUE, digits = 3),
    as.character(ncol(d) - 1),
    as.character(ncol(X_final)),
    paste0(nrow(coef_df), " (all - no sparsity)"),
    round(train_metrics["RMSE"], 4),
    round(train_metrics["R2"], 4),
    round(ridge_cv$RMSE_mean, 4),
    round(ridge_cv$RMSE_sd, 4),
    round(ridge_cv$R2_mean, 4),
    round(ridge_cv$MAE_mean, 4),
    ifelse(nrow(increase_genes) > 0, increase_genes$Gene[1], "N/A"),
    ifelse(nrow(decrease_genes) > 0, decrease_genes$Gene[1], "N/A")
  )
)

print(summary_table)
