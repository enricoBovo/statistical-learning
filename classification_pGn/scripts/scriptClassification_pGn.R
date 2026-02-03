rm(list=ls())
library(tidyverse)
library(glmnet)
library(randomForest)
library(xgboost)
library(ranger)
library(e1071)
library(caret)
library(pROC)

# ==============================================================================
# === CLASSIFICATION PROBLEM
# === HIGH-DIMENSIONAL (p > n)
# ==============================================================================

# load data
library(colonCA)
data(colonCA)
colonCA
str(colonCA)
head(colonCA$class)

# features
dim(exprs(colonCA)) 
X <- t(exprs(colonCA))
dim(X)
# class
head(pData(colonCA))
y <- pData(colonCA)$class
table(y) # 22 normal, 40 tumour

d <- as.data.frame(X)
d$y <- as.factor(y)

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

n_classes <- nlevels(y)
class_levels <- levels(y)

K <- 5
n <- nrow(d)
# stratified folds
fold_ids <- createFolds(d$y, k = K, list = FALSE)

compute_metrics <- function(pred_class, true_class, pred_prob = NULL) {
  pred_class <- factor(pred_class, levels = class_levels)
  true_class <- factor(true_class, levels = class_levels)
  acc <- mean(pred_class == true_class)
  cm <- table(Predicted = pred_class, Actual = true_class)
  if (n_classes == 2) {
    pos_class <- class_levels[2]
    neg_class <- class_levels[1]
    tp <- cm[pos_class, pos_class]
    fp <- cm[pos_class, neg_class]
    fn <- cm[neg_class, pos_class]
    tn <- cm[neg_class, neg_class]
    precision <- ifelse(tp + fp > 0, tp / (tp + fp), 0)
    recall <- ifelse(tp + fn > 0, tp / (tp + fn), 0)
    f1 <- ifelse(precision + recall > 0, 2 * precision * recall / (precision + recall), 0)
    specificity <- ifelse(tn + fp > 0, tn / (tn + fp), 0)
    return(list(
      accuracy = acc,
      precision = precision,
      recall = recall,
      f1 = f1,
      specificity = specificity,
      confusion_matrix = cm
    ))
  } else {
    return(list(accuracy = acc, confusion_matrix = cm))
  }
}

# Feature selection via t-test (frequently used in genomic)
feature_selection_ttest <- function(X_train, y_train, top_n = 100) {
  p_values <- apply(X_train, 2, function(gene) {
    t.test(gene ~ y_train)$p.value
  })
  top_features <- names(sort(p_values)[1:top_n])
  return(top_features)
}

# Feature selection via variance
feature_selection_variance <- function(X_train, top_n = 100) {
  variances <- apply(X_train, 2, var)
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
  # top 100 genes via t test
  selected_features <- feature_selection_ttest(X_train, y_train, top_n = 100)
  selected_features_all[[k]] <- selected_features
  X_train_selected <- X_train[, selected_features]
  X_test_selected <- X_test[, selected_features]
  X_train_mat <- as.matrix(X_train_selected)
  X_test_mat <- as.matrix(X_test_selected)
  
  # initialise predictions
  fold_predictions <- list()
  fold_predictions$y_true <- y_test
  fold_predictions$fold <- k
  
  # ---------------------------
  # ---   RIDGE LOGISTIC    ---
  # ---------------------------
  cv_ridge <- cv.glmnet(X_train_mat, y_train, family = "binomial", 
                        alpha = 0, nfolds = 5, type.measure = "class")
  ridge_pred <- predict(cv_ridge, newx = X_test_mat, s = "lambda.min", type = "class")
  ridge_prob <- predict(cv_ridge, newx = X_test_mat, s = "lambda.min", type = "response")
  fold_predictions$ridge <- factor(ridge_pred[,1], levels = class_levels)
  fold_predictions$ridge_prob <- as.vector(ridge_prob)
  
  # ---------------------------
  # ---   LASSO LOGISTIC    ---
  # ---------------------------
  cv_lasso <- cv.glmnet(X_train_mat, y_train, family = "binomial", 
                        alpha = 1, nfolds = 5, type.measure = "class")
  lasso_pred <- predict(cv_lasso, newx = X_test_mat, s = "lambda.min", type = "class")
  lasso_prob <- predict(cv_lasso, newx = X_test_mat, s = "lambda.min", type = "response")
  fold_predictions$lasso <- factor(lasso_pred[,1], levels = class_levels)
  fold_predictions$lasso_prob <- as.vector(lasso_prob)
  
  # ------------------------
  # ---   ELASTIC NET    ---
  # ------------------------
  cv_enet <- cv.glmnet(X_train_mat, y_train, family = "binomial", 
                       alpha = 0.5, nfolds = 5, type.measure = "class")
  enet_pred <- predict(cv_enet, newx = X_test_mat, s = "lambda.min", type = "class")
  enet_prob <- predict(cv_enet, newx = X_test_mat, s = "lambda.min", type = "response")
  fold_predictions$enet <- factor(enet_pred[,1], levels = class_levels)
  fold_predictions$enet_prob <- as.vector(enet_prob)
  
  # --------------------------
  # ---   RANDOM FOREST    ---
  # --------------------------
  # RF including only selected features
  rf_model <- randomForest(y ~ ., data = cbind(y = y_train, X_train_selected), 
                           ntree = 500)
  fold_predictions$rf <- predict(rf_model, newdata = X_test_selected, type = "class")
  fold_predictions$rf_prob <- predict(rf_model, newdata = X_test_selected, type = "prob")[, 2]
  
  # -------------------
  # ---   RANGER    ---
  # -------------------
  ranger_model <- ranger(y ~ ., data = cbind(y = y_train, X_train_selected), 
                         num.trees = 500, probability = TRUE)
  ranger_preds <- predict(ranger_model, data = as.data.frame(X_test_selected))
  fold_predictions$ranger <- factor(
    ifelse(ranger_preds$predictions[, 2] > 0.5, class_levels[2], class_levels[1]),
    levels = class_levels
  )
  fold_predictions$ranger_prob <- ranger_preds$predictions[, 2]
  
  # -----------------------
  # ---   LINEAR SVM    ---
  # -----------------------
  svm_model <- svm(y ~ ., data = cbind(y = y_train, X_train_selected), 
                   kernel = "linear", probability = TRUE, cost = 1)
  fold_predictions$svm <- predict(svm_model, newdata = as.data.frame(X_test_selected))
  svm_prob <- predict(svm_model, newdata = as.data.frame(X_test_selected), probability = TRUE)
  svm_prob_matrix <- attr(svm_prob, "probabilities")
  # correct columns (ordered)
  if (class_levels[2] %in% colnames(svm_prob_matrix)) {
    fold_predictions$svm_prob <- svm_prob_matrix[, class_levels[2]]
  } else {
    fold_predictions$svm_prob <- svm_prob_matrix[, 1]  # Fallback
  }
  # -------------------
  # ---   XGBOOST   ---
  # -------------------
  y_train_numeric <- as.numeric(y_train) - 1
  y_test_numeric <- as.numeric(y_test) - 1
  dtrain <- xgb.DMatrix(data = X_train_mat, label = y_train_numeric)
  dtest <- xgb.DMatrix(data = X_test_mat, label = y_test_numeric)
  
  params_xgb <- list(
    objective = "binary:logistic",
    learning_rate = 0.05,
    max_depth = 3,
    eval_metric = "logloss",
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
  xgb_prob_pos <- predict(xgb_model, dtest)
  xgb_pred <- ifelse(xgb_prob_pos > 0.5, class_levels[2], class_levels[1])
  fold_predictions$xgb <- factor(xgb_pred, levels = class_levels)
  fold_predictions$xgb_prob <- xgb_prob_pos
  
  # ------------------------
  # ---   NAIVE BAYES    ---
  # ------------------------
  nb_model <- naiveBayes(y ~ ., data = cbind(y = y_train, X_train_selected))
  fold_predictions$nb <- predict(nb_model, newdata = as.data.frame(X_test_selected))
  nb_prob <- predict(nb_model, newdata = as.data.frame(X_test_selected), type = "raw")
  # correct column (ordered variables
  if (class_levels[2] %in% colnames(nb_prob)) {
    fold_predictions$nb_prob <- nb_prob[, class_levels[2]]
  } else {
    fold_predictions$nb_prob <- nb_prob[, 1]  # Fallback
  }
  
  predictions_all[[k]] <- fold_predictions
  cat("\n")
}

# ----------------------------------------------------------------------------
# --- RESULTS AGGREGATION
# ----------------------------------------------------------------------------

# Models list
model_names <- c("ridge", "lasso", "enet", "rf", "ranger", "svm", "xgb", "nb")

# Calculate metrics combining folds
cv_summary <- map_dfr(model_names, function(model) {
  
  # all predictions
  all_preds <- map(predictions_all, ~ factor(.x[[model]], levels = class_levels)) %>% 
    do.call(c, .) %>%
    factor(levels = class_levels)
  all_true <- map(predictions_all, ~ factor(.x$y_true, levels = class_levels)) %>% 
    do.call(c, .) %>%
    factor(levels = class_levels)
  
  # global metrics
  global_metrics <- compute_metrics(all_preds, all_true)
  
  # metrics per fold
  fold_metrics <- map_dfr(1:K, function(k) {
    preds <- factor(predictions_all[[k]][[model]], levels = class_levels)
    true <- factor(predictions_all[[k]]$y_true, levels = class_levels)
    metrics <- compute_metrics(preds, true)
    
    tibble(
      Accuracy = metrics$accuracy,
      F1 = metrics$f1,
      Precision = metrics$precision,
      Recall = metrics$recall,
      Specificity = metrics$specificity
    )
  })
  
  tibble(
    Model = model,
    Accuracy_mean = mean(fold_metrics$Accuracy),
    Accuracy_sd = sd(fold_metrics$Accuracy),
    F1_mean = mean(fold_metrics$F1),
    Precision_mean = mean(fold_metrics$Precision),
    Recall_mean = mean(fold_metrics$Recall),
    Specificity_mean = mean(fold_metrics$Specificity)
  )
}) %>%
  arrange(desc(Accuracy_mean))

model_labels <- c(
  ridge = "Ridge Logistic",
  lasso = "Lasso Logistic",
  enet = "Elastic Net",
  rf = "Random Forest",
  ranger = "Ranger RF",
  svm = "SVM Linear",
  xgb = "XGBoost",
  nb = "Naive Bayes"
)

cv_summary <- cv_summary %>%
  mutate(Model_Label = model_labels[Model])

# Results
print(cv_summary %>% 
        select(Model_Label, Accuracy_mean, Accuracy_sd, F1_mean, Precision_mean) %>%
        rename(Model = Model_Label),
      n = Inf)

# best model
best_model <- cv_summary$Model_Label[1]
cat("\n Best model:", best_model, "\n")
cat(" Mean Accuracy:", round(cv_summary$Accuracy_mean[1], 4), "±", 
    round(cv_summary$Accuracy_sd[1], 4), "\n")

# ----------------------------------------------------------------------------
# --- FEATURE SELECTION ANALYSIS
# ----------------------------------------------------------------------------

# Selected features for each fold
all_selected <- unlist(selected_features_all)
feature_freq <- table(all_selected)
feature_stability <- sort(feature_freq / K, decreasing = TRUE)
# 10 most stable features (often selected)
print(head(feature_stability, 10))

# Plot
stable_features_df <- tibble(
  Feature = names(feature_stability),
  Frequency = as.vector(feature_stability)
) %>%
  head(20)

p_stability <- ggplot(stable_features_df, aes(x = reorder(Feature, Frequency), y = Frequency)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Feature Selection Stability",
       x = "Feature", y = "Frequency (of 5 folds)") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

print(p_stability)

# ----------------------------------------------------------------------------
# --- VISUALS
# ----------------------------------------------------------------------------

p1 <- cv_summary %>%
  mutate(Model_Label = fct_reorder(Model_Label, Accuracy_mean)) %>%
  ggplot(aes(x = Model_Label, y = Accuracy_mean)) +
  geom_col(fill = "steelblue", alpha = 0.7) +
  geom_errorbar(aes(ymin = Accuracy_mean - Accuracy_sd, 
                    ymax = Accuracy_mean + Accuracy_sd),
                width = 0.3, color = "darkred") +
  coord_flip() +
  labs(title = "Model comparison - Average Accuracy (p >> n)",
       x = NULL, y = "Average Accuracy") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

print(p1)

# ----------------------------------------------------------------------------
# --- RE-RUNNING OF BEST MODEL ON ENTIRE DATASET: ELASTIC NET
# ----------------------------------------------------------------------------

# Feature selection on the entire dataset
best_features <- feature_selection_ttest(d %>% select(-y), d$y, top_n = 100)
X_final <- as.matrix(d[, best_features])
y_final <- d$y

best_model_code <- cv_summary$Model[1]

final_model <- switch(
  best_model_code,
  ridge = cv.glmnet(X_final, y_final, family = "binomial", alpha = 0, nfolds = 5),
  lasso = cv.glmnet(X_final, y_final, family = "binomial", alpha = 1, nfolds = 5),
  enet = cv.glmnet(X_final, y_final, family = "binomial", alpha = 0.5, nfolds = 5),
  rf = randomForest(y ~ ., data = cbind(y = y_final, as.data.frame(X_final)), ntree = 500),
  ranger = ranger(y ~ ., data = cbind(y = y_final, as.data.frame(X_final)), 
                  num.trees = 500, probability = TRUE, importance = "impurity"),
  svm = svm(y ~ ., data = cbind(y = y_final, as.data.frame(X_final)), 
            kernel = "linear", probability = TRUE),
  xgb = {
    y_num <- as.numeric(y_final) - 1
    dtrain <- xgb.DMatrix(data = X_final, label = y_num)
    params <- list(objective = "binary:logistic", learning_rate = 0.05, 
                   max_depth = 3, subsample = 0.8, colsample_bytree = 0.8)
    cv_res <- xgb.cv(params = params, data = dtrain, nrounds = 200, 
                     nfold = 5, early_stopping_rounds = 20, verbose = 0)
    best_n <- ifelse(!is.null(cv_res$best_iteration), cv_res$best_iteration, 200)
    xgb.train(params = params, data = dtrain, nrounds = best_n, verbose = 0)
  },
  nb = naiveBayes(y ~ ., data = cbind(y = y_final, as.data.frame(X_final)))
)

# save
saveRDS(final_model, paste0("final_highdim_", best_model_code, ".rds"))
saveRDS(best_features, "selected_features.rds")

# ----------------------------------------------------------------------------
# --- PRESENTING THE RESULTS
# ----------------------------------------------------------------------------
library(pheatmap)

# PERFORMANCE

train_pred_class <- predict(final_model, newx = X_final, s = "lambda.min", type = "class")
train_pred_prob <- predict(final_model, newx = X_final, s = "lambda.min", type = "response")
train_pred_class <- factor(train_pred_class[,1], levels = class_levels)
cm_train <- confusionMatrix(train_pred_class, y_final)

cat("Accuracy:", round(cm_train$overall["Accuracy"], 4), "\n")
cat("Sensitivity (Recall):", round(cm_train$byClass["Sensitivity"], 4), "\n")
cat("Specificity:", round(cm_train$byClass["Specificity"], 4), "\n")
cat("Precision (PPV):", round(cm_train$byClass["Pos Pred Value"], 4), "\n")
cat("F1-Score:", round(cm_train$byClass["F1"], 4), "\n")

cat("Cross-Validation (5-fold):\n")
enet_cv <- cv_summary %>% filter(Model == "enet")
cat("Mean Accuracy:", round(enet_cv$Accuracy_mean, 4), "±", round(enet_cv$Accuracy_sd, 4), "\n")
cat("Mean F1-Score:", round(enet_cv$F1_mean, 4), "\n")
cat("Mean Precision:", round(enet_cv$Precision_mean, 4), "\n")

# CONFUSION MATRIX

print(cm_train$table)
cm_df <- as.data.frame(cm_train$table) %>%
  rename(Predicted = Prediction, Actual = Reference, Count = Freq)
cm_pct <- prop.table(cm_train$table, margin = 2) * 100
cm_pct_df <- as.data.frame(cm_pct) %>%
  rename(Predicted = Prediction, Actual = Reference, Percentage = Freq)
cm_plot_df <- cm_df %>% left_join(cm_pct_df, by = c("Predicted", "Actual"))

p_cm <- ggplot(cm_plot_df, aes(x = Actual, y = Predicted, fill = Percentage)) +
  geom_tile(color = "white", size = 2) +
  geom_text(aes(label = paste0(Count, "\n(", sprintf("%.1f%%", Percentage), ")")), 
            color = "white", size = 8, fontface = "bold") +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(title = "Confusion Matrix - Elastic Net",
       subtitle = paste0("Colon Cancer (n=", n, ", p=", ncol(X_final), ")"),
       x = "True Class", y = "Predicted Class") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(face = "bold", size = 18))

print(p_cm)

# COEFFICIENTS AND FEATURES IMPORTANCE

coef_matrix <- coef(final_model, s = "lambda.min")
coef_values <- as.vector(coef_matrix)[-1] 
coef_names <- rownames(coef_matrix)[-1]
# selected features
nonzero_idx <- which(coef_values != 0)
selected_genes <- coef_names[nonzero_idx]
selected_coefs <- coef_values[nonzero_idx]

cat("Intercept:", round(coef_matrix[1,1], 4), "\n")
cat("Selected genes:", length(selected_genes), "su", length(coef_names), "\n")
cat("Sparsity percentage:", round(100 * (1 - length(selected_genes)/length(coef_names)), 2), "%\n")

coef_df <- tibble(
  Gene = selected_genes,
  Coefficient = selected_coefs,
  Abs_Coefficient = abs(selected_coefs)
) %>%
  arrange(desc(Abs_Coefficient)) %>%
  mutate(
    Direction = ifelse(Coefficient > 0, "Tumor", "Normal"),
    Rank = row_number()
  )

# 20 most important genes
print(coef_df %>% select(Gene, Coefficient, Direction) %>% head(20), n = 20)

# Plot coefficients - Top 20
p_coef <- coef_df %>%
  head(20) %>%
  ggplot(aes(x = reorder(Gene, Abs_Coefficient), y = Coefficient, fill = Direction)) +
  geom_col() +
  coord_flip() +
  scale_fill_manual(values = c("Tumor" = "darkred", "Normal" = "darkblue")) +
  labs(title = "Top 20 Gene Importance - Elastic Net",
       subtitle = paste0("Accuracy CV = ", round(enet_cv$Accuracy_mean, 3), 
                         " & ", length(selected_genes), " genes selected"),
       x = NULL, y = "Coefficient",
       fill = "Promotes") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold", size = 14),
        axis.text.y = element_text(size = 10))

print(p_coef)

# Plot coefficients - All non-zero
p_coef_all <- ggplot(coef_df, aes(x = Rank, y = Coefficient, color = Direction)) +
  geom_point(size = 3, alpha = 0.7) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  scale_color_manual(values = c("Tumor" = "darkred", "Normal" = "darkblue")) +
  labs(title = paste("All", length(selected_genes), "non-zero coefficients"),
       x = "Rank (absolute value)", y = "Coefficient") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

print(p_coef_all)

# REGULARIZATION

# Plot coefficients by lambda values
plot(final_model$glmnet.fit, xvar = "lambda", label = TRUE,
     main = "Elastic Net Regularization Path")
abline(v = log(final_model$lambda.min), col = "red", lty = 2)
abline(v = log(final_model$lambda.1se), col = "blue", lty = 2)
legend("topright", legend = c("lambda.min", "lambda.1se"), 
       col = c("red", "blue"), lty = 2)

# CV error plot
plot(final_model, main = "Cross-Validation Error vs Lambda")

# FEATURE STABILITY ACROSS FOLDS

all_selected <- unlist(selected_features_all)
feature_freq <- table(all_selected)
feature_stability <- sort(feature_freq / K, decreasing = TRUE)

# 20 most stable features 
print(head(feature_stability, 20))

# comparison with selected features by final_model
stable_genes <- names(feature_stability)[feature_stability >= 0.8]  # at least 4/5 folds
cat("Stable genes:", length(stable_genes), "\n")

# Overlap 
overlap <- intersect(selected_genes, stable_genes)
cat("Overlap stable genes CV with Elastic Net:", length(overlap), "\n")
cat("Percentage:", round(100 * length(overlap) / length(selected_genes), 2), "%\n")

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
  labs(title = "Top 30 Features Selection Stability",
       subtitle = "Proportion of CV folds in which the feature was selected",
       x = "Gene", y = "Stability (0-1)") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

print(p_stability)

# ROC CURVE & AUC

prob_positive <- as.vector(train_pred_prob)
true_binary <- ifelse(y_final == class_levels[2], 1, 0)
roc_obj <- roc(true_binary, prob_positive, quiet = TRUE)
auc_value <- auc(roc_obj)

cat("AUC:", round(auc_value, 4), "\n")
cat("95% CI:", paste(round(ci.auc(roc_obj)[c(1,3)], 4), collapse = " - "), "\n")
cat("Optimal threshold:", round(coords(roc_obj, "best", ret = "threshold")$threshold, 4), "\n")

# Plot ROC
roc_df <- tibble(
  Specificity = roc_obj$specificities,
  Sensitivity = roc_obj$sensitivities
)
p_roc <- ggplot(roc_df, aes(x = 1 - Specificity, y = Sensitivity)) +
  geom_line(color = "darkblue", size = 1.5) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
  annotate("text", x = 0.6, y = 0.2, 
           label = paste0("AUC = ", round(auc_value, 3)), 
           size = 6, fontface = "bold") +
  labs(title = "ROC Curve - Elastic Net",
       x = "1 - Specificity (FPR)",
       y = "Sensitivity (TPR)") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(face = "bold"))

print(p_roc)

# HEATMAP SELECTED GENES

# Top 30 genes
top_genes <- coef_df$Gene[1:min(30, nrow(coef_df))]
heatmap_data <- t(X_final[, top_genes])
annotation_col <- data.frame(Class = y_final)
rownames(annotation_col) <- rownames(X_final)
ann_colors <- list(
  Class = setNames(c("darkblue", "darkred"), class_levels)
)
pheatmap(heatmap_data,
         annotation_col = annotation_col,
         annotation_colors = ann_colors,
         scale = "row",
         clustering_distance_rows = "correlation",
         clustering_distance_cols = "euclidean",
         show_colnames = FALSE,
         main = "Top 30 genes",
         fontsize = 10,
         fontsize_row = 8)

# PREDICTION CONFIDENCE

pred_confidence <- tibble(
  True_Class = y_final,
  Predicted_Class = train_pred_class,
  Probability = prob_positive,
  Correct = True_Class == Predicted_Class
)

cat("Mean:", round(mean(prob_positive), 4), "\n")
cat("Median:", round(median(prob_positive), 4), "\n\n")

# Distribuzione
p_conf <- ggplot(pred_confidence, aes(x = Probability, fill = Correct)) +
  geom_histogram(bins = 30, alpha = 0.7, position = "identity") +
  scale_fill_manual(values = c("FALSE" = "red", "TRUE" = "green"),
                    labels = c("Misclassified", "Correctly classified")) +
  labs(title = "Distribution - Predicted probabilities",
       x = "Probability Tumor", y = "Frequency",
       fill = "Prediction") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

print(p_conf)

# Misclassified samples
misclass <- pred_confidence %>% filter(!Correct)
cat("Misclassified samples:", nrow(misclass), "/", nrow(d), "\n")
if (nrow(misclass) > 0) {
  cat("Samples with less confidence:\n")
  misclass_sorted <- misclass %>%
    mutate(Distance_from_0.5 = abs(Probability - 0.5)) %>%
    arrange(Distance_from_0.5) %>%
    head(5)
  print(misclass_sorted %>% select(True_Class, Predicted_Class, Probability))
}

# INTERPRETATION

# Genes that increase tumour
tumor_genes <- coef_df %>% filter(Direction == "Tumor") %>% head(10)
for (i in 1:nrow(tumor_genes)) {
  cat("  ", i, ". ", tumor_genes$Gene[i], " (coef = ", 
      round(tumor_genes$Coefficient[i], 4), ")\n", sep = "")
}

# Genes that increas normal
normal_genes <- coef_df %>% filter(Direction == "Normal") %>% head(10)
for (i in 1:nrow(normal_genes)) {
  cat("  ", i, ". ", normal_genes$Gene[i], " (coef = ", 
      round(normal_genes$Coefficient[i], 4), ")\n", sep = "")
}

# SUMMARY

summary_table <- tibble(
  Metric = c(
    "Model",
    "Alpha (Elastic Net)",
    "Lambda min",
    "Genes in the dataset",
    "Genes after features selection",
    "Genes selected by Elastic Net",
    "Sparsity (%)",
    "Training Accuracy",
    "CV Accuracy (mean)",
    "CV Accuracy (sd)",
    "CV F1-Score",
    "AUC",
    "Most important gene (Tumor)",
    "Most important gene (Normal)"
  ),
  Value = c(
    "Elastic Net",
    "0.5",
    format(final_model$lambda.min, scientific = TRUE, digits = 3),
    as.character(ncol(d) - 1),
    as.character(ncol(X_final)),
    as.character(length(selected_genes)),
    paste0(round(100 * (1 - length(selected_genes)/ncol(X_final)), 1), "%"),
    round(cm_train$overall["Accuracy"], 4),
    round(enet_cv$Accuracy_mean, 4),
    round(enet_cv$Accuracy_sd, 4),
    round(enet_cv$F1_mean, 4),
    round(auc_value, 4),
    ifelse(nrow(tumor_genes) > 0, tumor_genes$Gene[1], "N/A"),
    ifelse(nrow(normal_genes) > 0, normal_genes$Gene[1], "N/A")
  )
)

print(summary_table)
