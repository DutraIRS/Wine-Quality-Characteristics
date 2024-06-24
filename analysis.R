library(rcompanion)
library(ordinal)
library(MASS)
library(readr)
library(brant)
library(AER)

# Define a function for 10-fold cross-validation
TenFoldCV <- function(data, formula_str, method = "probit") {
  # Number of folds
  K <- 10
  n <- nrow(data)
  
  # Create indices for stratified k-fold cross-validation
  fold_indices <- cut(1:n, breaks = K, labels = FALSE)
  
  # Initialize accuracy vector
  accuracies <- numeric(K)
  
  # Perform k-fold cross-validation
  for (k in 1:K) {
    # Split data into training and validation sets
    train_indices <- which(fold_indices != k)
    valid_indices <- which(fold_indices == k)
    
    train_data <- data[train_indices, ]
    valid_data <- data[valid_indices, ]
    
    # Fit the model on training data
    model <- polr(formula_str, data = train_data, method = method, Hess="TRUE")
    
    # Predict on validation data
    predicted_probs <- predict(model, newdata = valid_data, type = "probs")
    
    # Determine the predicted category with highest probability
    pred_category <- colnames(predicted_probs)[apply(predicted_probs, 1, which.max)]
    
    # Compute accuracy
    correct <- sum(pred_category == as.character(valid_data$quality))
    accuracies[k] <- correct / length(valid_indices)
  }
  
  # Return mean accuracy across folds
  return(mean(accuracies))
}

# Load the data
wine <- read_csv("normal_wine.csv")

wine$quality <- as.ordered(wine$quality)

# Fit the model
model_null <- polr(quality ~ 1, data = wine, Hess = TRUE, method = "probit")
model_all <- polr(quality ~ ., data = wine, Hess = TRUE, method = "probit")
summary(model_all)

# Ten-fold cross-validation accuracy
print(TenFoldCV(wine, "quality ~ .", method = "probit"))

# Wald test
coeftest(model_all)

# Model Selection
# We'll be looking at AIC and CV accuracy
# We'll include the next most relevant variable (measured by z value) and see if it improves the model

model1 <- polr(quality ~ alcohol, data = wine, Hess = TRUE, method = "probit")
summary(model1)
print(TenFoldCV(wine, "quality ~ alcohol", method = "probit"))
coeftest(model1)

model2 <- polr(quality ~ alcohol + volatile_acidity, data = wine, Hess = TRUE, method = "probit")
summary(model2)
print(TenFoldCV(wine, "quality ~ alcohol + volatile_acidity", method = "probit"))
coeftest(model2)

model3 <- polr(quality ~ alcohol + volatile_acidity + sulphates, data = wine, Hess = TRUE, method = "probit")
summary(model3)
print(TenFoldCV(wine, "quality ~ alcohol + volatile_acidity + sulphates", method = "probit"))
coeftest(model3)

model4 <- polr(quality ~ alcohol + volatile_acidity + sulphates + chlorides, data = wine, Hess = TRUE, method = "probit")
summary(model4)
print(TenFoldCV(wine, "quality ~ alcohol + volatile_acidity + sulphates + chlorides", method = "probit"))
coeftest(model4)

model5 <- polr(quality ~ alcohol + volatile_acidity + sulphates + chlorides + total_sulfur_dioxide, data = wine, Hess = TRUE, method = "probit")
summary(model5)
print(TenFoldCV(wine, "quality ~ alcohol + volatile_acidity + sulphates + chlorides + total_sulfur_dioxide", method = "probit"))
coeftest(model5)

model6 <- polr(quality ~ alcohol + volatile_acidity + sulphates + chlorides + total_sulfur_dioxide + residual_sugar*total_sulfur_dioxide, data = wine, Hess = TRUE, method = "probit")
summary(model6)
print(TenFoldCV(wine, "quality ~ alcohol + volatile_acidity + sulphates + chlorides + total_sulfur_dioxide + residual_sugar*total_sulfur_dioxide", method = "probit"))
coeftest(model6)

model7 <- polr(quality ~ alcohol + volatile_acidity + sulphates + chlorides + total_sulfur_dioxide + residual_sugar*total_sulfur_dioxide + pH*citric_acid, data = wine, Hess = TRUE, method = "probit")
summary(model7)
print(TenFoldCV(wine, "quality ~ alcohol + volatile_acidity + sulphates + chlorides + total_sulfur_dioxide + residual_sugar*total_sulfur_dioxide + pH*citric_acid", method = "probit"))
coeftest(model7)

# Brant test
brant(model7)

# Likelihood ratio test
lrtest(model7)

# Anova test
anova(model_null, model7)

# Confusion matrix
table(predict(model7), wine$quality)
write.csv(table(predict(model7), wine$quality), "confusion_matrix.csv")
