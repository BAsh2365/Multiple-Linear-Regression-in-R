library(readxl)
Atlanta_Falcons_data <- read_excel("Atlanta_Falcons_data.xlsx")
View(Atlanta_Falcons_data)

library(tidyr)

vars_needed <- c("TD", "Tgt", "Rec", "Ctch%", "Yds")

clean_df <- Atlanta_Falcons_data %>%
  drop_na(all_of(vars_needed))

set.seed(123)  # for reproducibility
n        <- nrow(clean_df)
train_i  <- sample(seq_len(n), size = 0.8 * n)
train_df <- clean_df[train_i, ]
test_df  <- clean_df[-train_i, ]

# 2. Fit your model on TRAIN only
MLR_Model_2 <- lm(
  TD   ~ Tgt + Rec + `Ctch%` + Yds,
  data = train_df
)
summary(MLR_Model_2)

histogram_TTS <- MLR_Model_2$residuals

hist(histogram_TTS)

qqnorm(histogram_TTS)
qqline(histogram_TTS)

# 3. Predict on TEST and evaluate
test_df$predicted_TD <- predict(MLR_Model_2, newdata = test_df)

# Basic scatter of Pred vs Actual on test set
plot(
  test_df$predicted_TD, test_df$TD,
  xlab = "Predicted TD (test)", ylab = "Actual TD (test)",
  main = "Test-set: Predicted vs Actual",
  pch = 19
)
abline(0, 1, col = "red")

# 5b. Test-set residual histogram & Q-Q
resid_test <- test_df$TD - test_df$predicted_TD
hist(resid_test, main="Test Residuals", xlab="Residual")
qqnorm(resid_test); qqline(resid_test)

#Checking Multicollinearity
library(car)
vif(MLR_Model_2)

#Targets, Rec and Catch % seem to be a bit on the larger side based on variable knowledge
#so we may have to cross validate and use lasso regression 
library(glmnet)

# 1) Build model matrix
x_train <- model.matrix(TD ~ Tgt + Rec +  `Ctch%` + Yds, data=train_df)[,-1]
y_train <- train_df$TD
x_test  <- model.matrix(TD ~ Tgt + Rec +  `Ctch%` + Yds, data=test_df)[,-1] # the -1 gets predictor column since intercept is present
y_test  <- test_df$TD

# 2) Ridge regression (alpha=0)
cv_ridge <- cv.glmnet(x_train, y_train, alpha=0)
best_lr  <- cv_ridge$lambda.min
ridge_mod <- glmnet(x_train, y_train, alpha=0, lambda=best_lr)

# 3) Lasso (alpha=1)
cv_lasso <- cv.glmnet(x_train, y_train, alpha=1)
best_ls  <- cv_lasso$lambda.min
lasso_mod <- glmnet(x_train, y_train, alpha=1, lambda=best_ls)

library(Metrics) #gets RMSE 

# 4) Compare test RMSE
pred_ridge <- predict(ridge_mod, x_test)
pred_lasso <- predict(lasso_mod, x_test)
cat("Ridge RMSE:", rmse(y_test, pred_ridge), 
    "  Lasso RMSE:", rmse(y_test, pred_lasso), "\n")

#We can see that the ridge Model outperforms the lasso model for TD prediction (3.52 TDs vs 3.95 TDs)

#Graphing Ridge Regression Data
plot(cv_ridge)
abline(v = log(cv_ridge$lambda.min), col = "red", lty = 2)

#Fitting final Ridge regression plot
final_ridge <- glmnet(
  x_train, y_train,
  alpha  = 0,
  lambda = cv_ridge$lambda.min
)

# Make test-set predictions
preds_ridge <- predict(final_ridge, x_test)

#Plot the final ridge regression
plot(
  as.numeric(preds_ridge), y_test,
  xlab = "Predicted TD (ridge)", 
  ylab = "Actual TD",
  main = "Ridge: Predicted vs Actual",
  pch = 19
)
abline(0,1,col="blue",lwd=2)

#Comparison plot for Ridge vs Regular Trained MLR model
# OLS predictions
test_df$pred_ols <- predict(MLR_Model_2, newdata = test_df)

# Ridge predictions (make sure preds_ridge is a plain numeric vector)
test_df$pred_ridge <- as.numeric(predict(
  final_ridge,      
  newx = x_test
))

#Plot of comparison
plot(
  test_df$pred_ols, test_df$TD,
  xlim = range(c(test_df$pred_ols, test_df$pred_ridge)),
  ylim = range(c(test_df$pred_ols, test_df$pred_ridge)),
  xlab = "Predicted TD",
  ylab = "Actual TD",
  main = "OLS (blue) vs Ridge (red)",
  pch  = 19,
  col  = rgb(0,0,1,0.6)   
)
points(
  test_df$pred_ridge, test_df$TD,
  pch  = 17,
  col  = rgb(1,0,0,0.6) 
)
abline(0,1, lty=2)       
legend(
  "topleft",
  legend = c("OLS","Ridge"),
  pch    = c(19,17),
  col    = c("blue","red"),
  bty    = "n"
)
