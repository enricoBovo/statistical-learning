rm(list=ls())
library(tidyverse)
library(glmnet)
library(rpart)
library(randomForest)
library(xgboost)
library(ranger)
library(nnet)
library(e1071)
library(caret)
library(pROC)
library(ROCR)
library(gridExtra)

# ==============================================================================
# === CLASSIFICATION PROBLEM 
# ==============================================================================

# load data
setwd("C:/Users/enric/OneDrive/Desktop/NewJob/Python_projects/data/wineQuality")
wine <- read.csv("winequality-white.csv", sep = ";")
str(wine); summary(wine)
# quality from 0 to 10, here 3 to 9

# features
X <- wine[,-12]

# target
summary(wine$quality)
par(mfrow = c(1,2))
hist(wine$quality, breaks = 10, main = "", xlab = "Original scale", ylab = "")
hist(log(wine$quality), breaks = 10, main = "", xlab = "Log scale", ylab = "")
par(mfrow = c(1,1))
# - classification: low (0-5), medium (6), high (7-10)
y <- cut(wine$quality,
         breaks = quantile(wine$quality, probs = c(0, 1/3, 2/3, 1)),
         include.lowest = TRUE,
         labels = c("low", "medium", "high")
)
summary(y)
table(wine$quality, y)

d <- cbind(y, X)
if (!is.factor(d$y)) {
  d$y <- as.factor(d$y)
}

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

n_classes <- nlevels(d$y)
class_levels <- levels(d$y)

K <- 5
n <- nrow(d)
# stratified folds
fold_ids <- createFolds(d$y, k = K, list = FALSE)

compute_metrics <- function(pred_class, true_class, pred_prob = NULL) {
  acc <- mean(pred_class == true_class) 
  cm <- table(Predicted = pred_class, Actual = true_class) 
  metrics_by_class <- sapply(class_levels, function(cls) {
    tp <- sum(pred_class == cls & true_class == cls)
    fp <- sum(pred_class == cls & true_class != cls)
    fn <- sum(pred_class != cls & true_class == cls)
    tn <- sum(pred_class != cls & true_class != cls)
    precision <- ifelse(tp+fp>0, tp/(tp+fp), 0)
    recall <- ifelse(tp+fn>0, tp/(tp+fn), 0)
    f1 <- ifelse(precision + recall > 0, 2*precision*recall/(precision+recall), 0)
    c(Precision = precision, Recall = recall, F1 = f1)
  })
  # macro
  macro_precision <- mean(metrics_by_class["Precision", ])
  macro_recall <- mean(metrics_by_class["Recall", ])
  macro_f1 <- mean(metrics_by_class["F1", ])
  # log-loss (if probabilities available)
  logloss <- NA
  if (!is.null(pred_prob)) {
    pred_prob <- pmax(pred_prob, 1e-15)
    pred_prob <- pmin(pred_prob, 1 - 1e-15)
    true_matrix <- model.matrix(~ true_class - 1)
    colnames(true_matrix) <- class_levels
    logloss <- -mean(rowSums(true_matrix * log(pred_prob)))
  }
  list(
    accuracy = acc,
    macro_precision = macro_precision,
    macro_recall = macro_recall,
    macro_f1 = macro_f1,
    logloss = logloss,
    confusion_matrix = cm
  )
}

# initialise results lists
cv_results <- list()
predictions_all <- list()
confusion_matrices <- list()

# ----------------------------------------------------------------------------
# --- START
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
  
  # glmnet matrix
  X_train_mat <- model.matrix(~ . - 1, data = X_train)
  X_test_mat <- model.matrix(~ . - 1, data = X_test)
  
  # initialise predictions
  fold_predictions <- list()
  fold_predictions$y_true <- y_test
  fold_predictions$fold <- k
  
  # --------------------------------------------
  # ---   MULTINOMIAL LOGISTIC REGRESSION    ---
  # --------------------------------------------
  multinom_model <- multinom(y ~ ., data = train, trace = FALSE)
  fold_predictions$multinom <- predict(multinom_model, newdata = test, type = "class")
  fold_predictions$multinom_prob <- predict(multinom_model, newdata = test, type = "probs")
  
  # ---------------------------
  # ---   RIDGE LOGISTIC    ---
  # ---------------------------
  cv_ridge <- cv.glmnet(X_train_mat, y_train, family = "multinomial", 
                        alpha = 0, nfolds = 5, type.measure = "class")
  ridge_pred <- predict(cv_ridge, newx = X_test_mat, s = "lambda.min", type = "class")
  ridge_prob <- predict(cv_ridge, newx = X_test_mat, s = "lambda.min", type = "response")
  fold_predictions$ridge <- as.factor(ridge_pred[,1])
  fold_predictions$ridge_prob <- ridge_prob[,,1]
  
  # ---------------------------
  # ---   LASSO LOGISTIC    ---
  # ---------------------------
  cv_lasso <- cv.glmnet(X_train_mat, y_train, family = "multinomial", 
                        alpha = 1, nfolds = 5, type.measure = "class")
  lasso_pred <- predict(cv_lasso, newx = X_test_mat, s = "lambda.min", type = "class")
  lasso_prob <- predict(cv_lasso, newx = X_test_mat, s = "lambda.min", type = "response")
  fold_predictions$lasso <- as.factor(lasso_pred[,1])
  fold_predictions$lasso_prob <- lasso_prob[,,1]
  
  # ------------------------
  # ---   ELASTIC NET    ---
  # ------------------------
  cv_enet <- cv.glmnet(X_train_mat, y_train, family = "multinomial", 
                       alpha = 0.5, nfolds = 5, type.measure = "class")
  enet_pred <- predict(cv_enet, newx = X_test_mat, s = "lambda.min", type = "class")
  enet_prob <- predict(cv_enet, newx = X_test_mat, s = "lambda.min", type = "response")
  fold_predictions$enet <- as.factor(enet_pred[,1])
  fold_predictions$enet_prob <- enet_prob[,,1]
  
  # -----------------------------
  # ---   CLASSIFICATION TREE ---
  # -----------------------------
  tree_model <- rpart(y ~ ., data = train, method = "class",
                      control = rpart.control(cp = 0.001, xval = 10))
  cp_opt <- tree_model$cptable[which.min(tree_model$cptable[, "xerror"]), "CP"]
  tree_pruned <- prune(tree_model, cp = cp_opt)
  fold_predictions$tree <- predict(tree_pruned, newdata = test, type = "class")
  fold_predictions$tree_prob <- predict(tree_pruned, newdata = test, type = "prob")
  
  # --------------------------
  # ---   RANDOM FOREST    ---
  # --------------------------
  rf_model <- randomForest(y ~ ., data = train, ntree = 500)
  fold_predictions$rf <- predict(rf_model, newdata = test, type = "class")
  fold_predictions$rf_prob <- predict(rf_model, newdata = test, type = "prob")
  
  # -------------------
  # ---   RANGER    ---
  # -------------------
  ranger_model <- ranger(y ~ ., data = train, num.trees = 500, 
                         probability = TRUE)
  fold_predictions$ranger <- predict(ranger_model, data = test)$predictions %>% 
    apply(1, which.max) %>% 
    factor(levels = 1:n_classes, labels = class_levels)
  fold_predictions$ranger_prob <- predict(ranger_model, data = test)$predictions
  
  # -------------------
  # ---   XGBOOST   ---
  # -------------------
  # classes become 0,1,2
  y_train_numeric <- as.numeric(y_train) - 1
  y_test_numeric <- as.numeric(y_test) - 1
  dtrain <- xgb.DMatrix(data = X_train_mat, label = y_train_numeric)
  dtest <- xgb.DMatrix(data = X_test_mat, label = y_test_numeric)
  params_xgb <- list(
    objective = "multi:softprob",
    num_class = n_classes,
    learning_rate = 0.1,
    max_depth = 6,
    eval_metric = "mlogloss"
  )
  cv_xgb <- xgb.cv(
    params = params_xgb,
    data = dtrain,
    nrounds = 300,
    nfold = 5,
    early_stopping_rounds = 20,
    verbose = 0
  )
  best_iter_xgb <- ifelse(!is.null(cv_xgb$best_iteration) && cv_xgb$best_iteration > 0,
                          cv_xgb$best_iteration, 300)
  xgb_model <- xgb.train(
    params = params_xgb,
    data = dtrain,
    nrounds = best_iter_xgb,
    verbose = 0
  )
  xgb_prob <- predict(xgb_model, dtest)
  xgb_prob_matrix <- matrix(xgb_prob, ncol = n_classes, byrow = TRUE)
  colnames(xgb_prob_matrix) <- class_levels
  xgb_pred <- class_levels[apply(xgb_prob_matrix, 1, which.max)]
  
  fold_predictions$xgb <- as.factor(xgb_pred)
  fold_predictions$xgb_prob <- xgb_prob_matrix
  
  # ------------------------
  # ---   NAIVE BAYES    ---
  # ------------------------
  nb_model <- naiveBayes(y ~ ., data = train)
  fold_predictions$nb <- predict(nb_model, newdata = test, type = "class")
  fold_predictions$nb_prob <- predict(nb_model, newdata = test, type = "raw")
  
  # -----------------------
  # ---   LINEAR SVM    ---
  # -----------------------
  svm_model <- svm(y ~ ., data = train, kernel = "linear", probability = TRUE)
  fold_predictions$svm <- predict(svm_model, newdata = test)
  svm_pred_prob <- predict(svm_model, newdata = test, probability = TRUE)
  fold_predictions$svm_prob <- attr(svm_pred_prob, "probabilities")
  
  # ----------------------
  # ---   SVM (RBF)    ---
  # ----------------------
  svm_rbf_model <- svm(y ~ ., data = train, kernel = "radial", probability = TRUE)
  fold_predictions$svm_rbf <- predict(svm_rbf_model, newdata = test)
  svm_rbf_pred_prob <- predict(svm_rbf_model, newdata = test, probability = TRUE)
  fold_predictions$svm_rbf_prob <- attr(svm_rbf_pred_prob, "probabilities")
  
  # save predictions
  predictions_all[[k]] <- fold_predictions
  cat("\n")
}

# ----------------------------------------------------------------------------
# --- RESULTS AGGREGATION
# ----------------------------------------------------------------------------

# Models list
model_names <- c("multinom", "ridge", "lasso", "enet", "tree", 
                 "rf", "ranger", "xgb", "nb", "svm", "svm_rbf")

# Calculate metrics combining folds
cv_summary <- map_dfr(model_names, function(model) {
  
  # all predictions
  all_preds <- map(predictions_all, ~ .x[[model]]) %>% 
    unlist() %>% 
    factor(levels = class_levels)
  all_true <- map(predictions_all, ~ .x$y_true) %>% 
    unlist() %>% 
    factor(levels = class_levels)
  
  # probabilities (if available)
  prob_name <- paste0(model, "_prob")
  all_probs <- NULL
  if (prob_name %in% names(predictions_all[[1]])) {
    all_probs <- map(predictions_all, ~ .x[[prob_name]]) %>% 
      do.call(rbind, .)
  }
  
  # global metrics
  global_metrics <- compute_metrics(all_preds, all_true, all_probs)
  
  # metrics per fold
  fold_metrics <- map_dfr(1:K, function(k) {
    preds <- predictions_all[[k]][[model]]
    true <- predictions_all[[k]]$y_true
    probs <- predictions_all[[k]][[prob_name]]
    metrics <- compute_metrics(preds, true, probs)
    
    tibble(
      Accuracy = metrics$accuracy,
      Macro_F1 = metrics$macro_f1,
      Macro_Precision = metrics$macro_precision,
      Macro_Recall = metrics$macro_recall,
      LogLoss = metrics$logloss
    )
  })
  
  # confusion matrix
  confusion_matrices[[model]] <<- global_metrics$confusion_matrix
  
  tibble(
    Model = model,
    Accuracy_mean = mean(fold_metrics$Accuracy),
    Accuracy_sd = sd(fold_metrics$Accuracy),
    F1_mean = mean(fold_metrics$Macro_F1),
    F1_sd = sd(fold_metrics$Macro_F1),
    Precision_mean = mean(fold_metrics$Macro_Precision),
    Recall_mean = mean(fold_metrics$Macro_Recall),
    LogLoss_mean = mean(fold_metrics$LogLoss, na.rm = TRUE),
    Accuracy_global = global_metrics$accuracy
  )
}) %>%
  arrange(desc(Accuracy_mean))

model_labels <- c(
  multinom = "Multinomial Logistic",
  ridge = "Ridge Logistic",
  lasso = "Lasso Logistic",
  enet = "Elastic Net",
  tree = "Classification Tree",
  rf = "Random Forest",
  ranger = "Ranger RF",
  xgb = "XGBoost",
  nb = "Naive Bayes",
  svm = "SVM (Linear)",
  svm_rbf = "SVM (RBF)"
)

cv_summary <- cv_summary %>%
  mutate(Model_Label = model_labels[Model])

# Results
print(cv_summary %>% 
        select(Model_Label, Accuracy_mean, Accuracy_sd, F1_mean, LogLoss_mean) %>%
        rename(Model = Model_Label, 
               `Accuracy (mean)` = Accuracy_mean,
               `Accuracy (sd)` = Accuracy_sd,
               `F1 (mean)` = F1_mean,
               `LogLoss (mean)` = LogLoss_mean),
      n = Inf)

# best model
best_model <- cv_summary$Model_Label[1]
best_acc <- cv_summary$Accuracy_mean[1]
best_sd <- cv_summary$Accuracy_sd[1]

cat("\n Best model:", best_model, "\n")
cat(" Mean Accuracy:", round(best_acc, 4), "±", round(best_sd, 4), "\n")

# ----------------------------------------------------------------------------
# --- VISUALS
# ----------------------------------------------------------------------------

# Plot 1: Accuracy and errors comparison
p1 <- cv_summary %>%
  mutate(Model_Label = fct_reorder(Model_Label, Accuracy_mean)) %>%
  ggplot(aes(x = Model_Label, y = Accuracy_mean)) +
  geom_col(fill = "steelblue", alpha = 0.7) +
  geom_errorbar(aes(ymin = Accuracy_mean - Accuracy_sd, 
                    ymax = Accuracy_mean + Accuracy_sd),
                width = 0.3, color = "darkred") +
  coord_flip() +
  labs(title = "Model comparison - Average Accuracy (5-fold CV)",
       subtitle = "Errors = ± 1 SD",
       x = NULL, y = "Average Accuracy") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

print(p1)

# Plot 2: F1-Score vs Accuracy
p2 <- cv_summary %>%
  ggplot(aes(x = Accuracy_mean, y = F1_mean, label = Model_Label)) +
  geom_point(size = 3, color = "steelblue") +
  geom_text(vjust = -0.5, size = 3) +
  labs(title = "F1-Score vs Accuracy",
       x = "Accuracy", y = "Macro F1-Score") +
  theme_minimal()

print(p2)

# Plot 3: Best model Confusion Matrix
best_model_code <- cv_summary$Model[1]
cm_best <- confusion_matrices[[best_model_code]]
cm_pct <- prop.table(cm_best, margin = 1) * 100
cm_df <- as.data.frame(cm_pct) %>%
  rename(Predicted = Predicted, Actual = Actual, Percentage = Freq)
p3 <- ggplot(cm_df, aes(x = Actual, y = Predicted, fill = Percentage)) +
  geom_tile() +
  geom_text(aes(label = sprintf("%.1f%%", Percentage)), color = "white", size = 5) +
  scale_fill_gradient(low = "steelblue", high = "darkblue") +
  labs(title = paste("Confusion Matrix -", best_model),
       subtitle = "Percentuali per riga (classe vera)") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

print(p3)

# ----------------------------------------------------------------------------
# --- RE-RUNNING OF BEST MODEL ON ENTIRE DATASET: RANDOM FOREST
# ----------------------------------------------------------------------------

X_full <- d %>% select(-y)
y_full <- d$y
X_full_mat <- model.matrix(~ . - 1, data = X_full)

final_model <- switch(
  best_model_code,
  multinom = multinom(y ~ ., data = d, trace = FALSE),
  ridge = cv.glmnet(X_full_mat, y_full, family = "multinomial", alpha = 0, nfolds = 5),
  lasso = cv.glmnet(X_full_mat, y_full, family = "multinomial", alpha = 1, nfolds = 5),
  enet = cv.glmnet(X_full_mat, y_full, family = "multinomial", alpha = 0.5, nfolds = 5),
  tree = {
    t <- rpart(y ~ ., data = d, method = "class", 
               control = rpart.control(cp = 0.001, xval = 10))
    prune(t, cp = t$cptable[which.min(t$cptable[, "xerror"]), "CP"])
  },
  rf = randomForest(y ~ ., data = d, ntree = 500),
  ranger = ranger(y ~ ., data = d, num.trees = 500, 
                  probability = TRUE, importance = "impurity"),
  xgb = {
    y_full_numeric <- as.numeric(y_full) - 1
    dtrain_full <- xgb.DMatrix(data = X_full_mat, label = y_full_numeric)
    params_full <- list(objective = "multi:softprob", num_class = n_classes,
                        learning_rate = 0.1, max_depth = 6, eval_metric = "mlogloss")
    cv_xgb_full <- xgb.cv(params = params_full, data = dtrain_full, 
                          nrounds = 300, nfold = 5, early_stopping_rounds = 20, verbose = 0)
    best_iter_final <- ifelse(!is.null(cv_xgb_full$best_iteration) && 
                                cv_xgb_full$best_iteration > 0,
                              cv_xgb_full$best_iteration, 300)
    xgb.train(params = params_full, data = dtrain_full, 
              nrounds = best_iter_final, verbose = 0)
  },
  nb = naiveBayes(y ~ ., data = d),
  svm = svm(y ~ ., data = d, kernel = "linear", probability = TRUE),
  svm_rbf = svm(y ~ ., data = d, kernel = "radial", probability = TRUE)
)

final_model
# save
saveRDS(final_model, paste0("final_model_", best_model_code, ".rds"))


# ----------------------------------------------------------------------------
# --- PRESENTING THE RESULTS
# ----------------------------------------------------------------------------

# PERFORMANCE

cat("Trees:", final_model$ntree, "\n")
cat("Variables per split (mtry):", final_model$mtry, "\n")
cat("OOB Error Rate:", round(mean(final_model$err.rate[final_model$ntree, ]), 4), "\n\n")
cat("OOB Error per class\n")
oob_errors <- final_model$err.rate[final_model$ntree, ]
for (i in 1:length(oob_errors)) {
  cat("  ", names(oob_errors)[i], ": ", round(oob_errors[i], 4), "\n", sep = "")
}

cat("Cross-Validation (5-fold):\n")
cat("Mean Accuracy:", round(cv_summary$Accuracy_mean[cv_summary$Model == "rf"], 4), "±", 
    round(cv_summary$Accuracy_sd[cv_summary$Model == "rf"], 4), "\n")
cat("Mean F1-Score:", round(cv_summary$F1_mean[cv_summary$Model == "rf"], 4), "\n")
cat("Mean Precision:", round(cv_summary$Precision_mean[cv_summary$Model == "rf"], 4), "\n")
cat("Mean Recall:", round(cv_summary$Recall_mean[cv_summary$Model == "rf"], 4), "\n")

# CONFUSION MATRIX

cm <- final_model$confusion[, 1:n_classes]
print(cm)

# metrics per class
metrics_by_class <- sapply(class_levels, function(cls) {
  tp <- cm[cls, cls]
  fp <- sum(cm[cls, ]) - tp
  fn <- sum(cm[, cls]) - tp
  tn <- sum(cm) - tp - fp - fn
  precision <- tp / (tp + fp)
  recall <- tp / (tp + fn)
  f1 <- 2 * precision * recall / (precision + recall)
  specificity <- tn / (tn + fp)
  c(TP = tp, FP = fp, FN = fn, TN = tn,
    Precision = precision, Recall = recall, 
    F1_Score = f1, Specificity = specificity)
})
print(round(t(metrics_by_class), 4))

# visualise confusion matrix
cm_pct <- prop.table(cm, margin = 2) * 100
cm_df <- as.data.frame(as.table(cm)) %>%
  rename(Predicted = Var1, Actual = Var2, Count = Freq) %>%
  left_join(
    as.data.frame(as.table(cm_pct)) %>%
      rename(Predicted = Var1, Actual = Var2, Percentage = Freq),
    by = c("Predicted", "Actual")
  )
p_cm <- ggplot(cm_df, aes(x = Actual, y = Predicted, fill = Percentage)) +
  geom_tile(color = "white", size = 1) +
  geom_text(aes(label = paste0(Count, "\n(", sprintf("%.1f%%", Percentage), ")")), 
            color = "white", size = 5, fontface = "bold") +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(title = "Confusion Matrix - Random Forest (OOB)",
       x = "True Class", y = "Predicted Class") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(face = "bold", size = 16),
        legend.position = "right")

print(p_cm)

# IMPORTANCE

varImpPlot(final_model, main = "Variable Importance Plot")

importance_matrix <- final_model$importance
importance_df <- tibble(
  Variable = rownames(importance_matrix),
  MeanDecreaseGini = importance_matrix[,"MeanDecreaseGini"]
) %>%
  arrange(desc(MeanDecreaseGini)) %>%
  mutate(
    Gini_Pct = 100 * MeanDecreaseGini / sum(MeanDecreaseGini),
    Cumulative_Pct = cumsum(Gini_Pct)
  )

print(importance_df, n = Inf)

# Plot importance - MeanDecreaseGini
p_imp_gini <- ggplot(importance_df, aes(x = reorder(Variable, MeanDecreaseGini), 
                                        y = MeanDecreaseGini)) +
  geom_col(fill = "steelblue", alpha = 0.8) +
  coord_flip() +
  labs(title = "Variable Importance - Mean Decrease Gini",
       subtitle = paste0("Wine Quality Classification (Accuracy = ", 
                         round(1 - mean(final_model$err.rate[final_model$ntree, ]), 3), ")"),
       x = NULL, y = "Mean Decrease Gini") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

print(p_imp_gini)

# CLASS PROBABILITIES ANALYSIS

# OOB predictions with probability
oob_preds <- predict(final_model, type = "prob")
oob_class <- predict(final_model, type = "class")
max_prob <- apply(oob_preds, 1, max)

pred_analysis <- tibble(
  True_Class = d$y,
  Predicted_Class = oob_class,
  Max_Probability = max_prob,
  Correct = True_Class == Predicted_Class
)

for (i in 1:n_classes) {
  pred_analysis[[paste0("Prob_", class_levels[i])]] <- oob_preds[, i]
}

p_prob_dist <- ggplot(pred_analysis, aes(x = Max_Probability, fill = Correct)) +
  geom_histogram(bins = 50, alpha = 0.7, position = "identity") +
  scale_fill_manual(values = c("FALSE" = "red", "TRUE" = "green"),
                    labels = c("Misclassified", "Correctly classified")) +
  labs(title = "Max probabilities distribution (OOB)",
       x = "Max probability", y = "Frequency",
       fill = "Prediction") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

print(p_prob_dist)

# Box plot probability for true class
prob_long <- pred_analysis %>%
  pivot_longer(cols = starts_with("Prob_"), 
               names_to = "Predicted_For", 
               values_to = "Probability") %>%
  mutate(Predicted_For = str_remove(Predicted_For, "Prob_"))

p_prob_box <- ggplot(prob_long, aes(x = True_Class, y = Probability, fill = Predicted_For)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Probability distribution for true class",
       x = "True Class", y = "Predicted Probability",
       fill = "Prob. per class") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

print(p_prob_box)

# Confidence analysis

cat("Mean:", round(mean(max_prob), 4), "\n")
cat("Median:", round(median(max_prob), 4), "\n")
cat("Min:", round(min(max_prob), 4), "\n")
cat("Max:", round(max(max_prob), 4), "\n\n")

confidence_stats <- pred_analysis %>%
  group_by(Correct) %>%
  summarise(
    N = n(),
    Mean_Prob = mean(Max_Probability),
    Median_Prob = median(Max_Probability),
    SD_Prob = sd(Max_Probability)
  )

print(confidence_stats)

# ROC CURVES (One-vs-Rest)

# ROC per class
roc_list <- list()
auc_values <- numeric(n_classes)

for (i in 1:n_classes) {
  cls <- class_levels[i]
  
  # One vs rest
  true_binary <- ifelse(d$y == cls, 1, 0)
  pred_prob <- oob_preds[, cls]
  
  # ROC curve
  roc_obj <- roc(true_binary, pred_prob, quiet = TRUE)
  roc_list[[cls]] <- roc_obj
  auc_values[i] <- auc(roc_obj)
  
  cat("Class", cls, "- AUC:", round(auc_values[i], 4), "\n")
}

cat("Mean AUC (macro):", round(mean(auc_values), 4), "\n")

# Plot ROC curves
roc_data <- map_dfr(class_levels, function(cls) {
  roc_obj <- roc_list[[cls]]
  tibble(
    Class = cls,
    Specificity = roc_obj$specificities,
    Sensitivity = roc_obj$sensitivities,
    AUC = as.numeric(auc(roc_obj))
  )
})

p_roc <- ggplot(roc_data, aes(x = 1 - Specificity, y = Sensitivity, 
                              color = paste0(Class, " (AUC=", round(AUC, 3), ")"))) +
  geom_line(size = 1.2) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
  labs(title = "ROC Curves - One-vs-Rest",
       x = "1 - Specificity (False Positive Rate)",
       y = "Sensitivity (True Positive Rate)",
       color = "Class") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"),
        legend.position = "bottom")

print(p_roc)

# MISCLASSIFICATION ANALYSIS (ERROS)

misclass <- pred_analysis %>%
  filter(!Correct) %>%
  select(True_Class, Predicted_Class, Max_Probability, 
         starts_with("Prob_"))

cat("Number of errors:", nrow(misclass), "/", nrow(d), 
    "(", round(100 * nrow(misclass) / nrow(d), 2), "%)\n")

# Confusion matrix errors
error_matrix <- table(
  True = misclass$True_Class,
  Predicted = misclass$Predicted_Class
)
print(error_matrix)

top_errors <- misclass %>%
  arrange(Max_Probability) %>%
  head(10) %>%
  select(True_Class, Predicted_Class, Max_Probability)
print(top_errors)

# PARTIAL DEPENDENCE PLOTS (top 3-4 variables)

top_vars <- importance_df$Variable[1:4]
# wrapper
pred_wrapper <- function(object, newdata) {
  predict(object, newdata = newdata, type = "prob")
}
# PDP per variable and per class
pdp_plots <- list()
for (var in top_vars) {
  pd_data <- partial(final_model, pred.var = var, 
                     train = d, pred.fun = pred_wrapper,
                     prob = TRUE)
    pd_long <- pd_data %>%
    as_tibble() %>%
    pivot_longer(cols = -all_of(var), 
                 names_to = "Class", 
                 values_to = "Probability")
  p <- ggplot(pd_long, aes_string(x = var, y = "Probability", color = "Class")) +
    geom_line(size = 1.2) +
    labs(title = paste("Partial Dependence:", var),
         x = var, y = "Predicted Probability") +
    theme_minimal() +
    theme(plot.title = element_text(face = "bold"))
  
  pdp_plots[[var]] <- p
}

grid.arrange(grobs = pdp_plots, ncol = 2)

# LEARNING CURVES (Error Rate vs Number of Trees)

error_df <- as.data.frame(final_model$err.rate) %>%
  mutate(Trees = 1:n()) %>%
  pivot_longer(cols = -Trees, names_to = "Class", values_to = "Error")
p_learning <- ggplot(error_df, aes(x = Trees, y = Error, color = Class)) +
  geom_line(size = 1) +
  labs(title = "OOB Error Rate vs Number of Trees",
       x = "Number of Trees", y = "OOB Error Rate") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

print(p_learning)

# SUMMARY TABLE

summary_table <- tibble(
  Metric = c(
    "Model", 
    "Number of trees", 
    "mtry",
    "Number of classes",
    "OOB Error Rate",
    "OOB Accuracy",
    "CV Accuracy (mean)", 
    "CV Accuracy (sd)",
    "CV F1-Score (mean)",
    "Macro Precision",
    "Macro Recall",
    "AUC (mean)",
    "Most important variable (Gini)",
    "Second most important variable (Gini)"
  ),
  Value = c(
    "Random Forest", 
    as.character(final_model$ntree),
    as.character(final_model$mtry),
    as.character(n_classes),
    round(mean(final_model$err.rate[final_model$ntree, ]), 4),
    round(1 - mean(final_model$err.rate[final_model$ntree, ]), 4),
    round(cv_summary$Accuracy_mean[cv_summary$Model == "rf"], 4),
    round(cv_summary$Accuracy_sd[cv_summary$Model == "rf"], 4),
    round(cv_summary$F1_mean[cv_summary$Model == "rf"], 4),
    round(cv_summary$Precision_mean[cv_summary$Model == "rf"], 4),
    round(cv_summary$Recall_mean[cv_summary$Model == "rf"], 4),
    round(mean(auc_values), 4),
    importance_df$Variable[1],
    importance_df$Variable[2]
  )
)

print(summary_table)

# EXAMPLE: PREDICTION ON NEW DATA

# Example: wine with average values
new_obs <- d %>% 
  select(-y) %>% 
  summarise(across(everything(), mean))
pred_new <- predict(final_model, newdata = new_obs, type = "prob")
pred_class_new <- predict(final_model, newdata = new_obs, type = "class")
for (i in 1:n_classes) {
  cat("  ", class_levels[i], ": ", 
      round(pred_new[1, i] * 100, 2), "%\n", sep = "")
}
cat("Predicted quality with average characteristics:", as.character(pred_class_new), "\n")
