library(readxl)
Atlanta_Falcons_data <- read_excel("Atlanta_Falcons_data.xlsx")
View(Atlanta_Falcons_data)

set.seed(123)  # for reproducibility
n        <- nrow(Atlanta_Falcons_data)
train_i  <- sample(seq_len(n), size = 0.8 * n)
train_df <- Atlanta_Falcons_data[train_i, ]
test_df  <- Atlanta_Falcons_data[-train_i, ]

# 2. Fit your model on TRAIN only
MLR_Model <- lm(
  TD   ~ Tgt + Rec + `Ctch%` + Yds,
  data = train_df
)
summary(MLR_Model)

histogram_TTS <- MLR_Model_2$residuals

hist(histogram_TTS)

qqnorm(histogram_TTS)
qqline(histogram_TTS)

# 3. Predict on TEST and evaluate
test_df$predicted_TD <- predict(MLR_Model, newdata = test_df)

# Basic scatter of Pred vs Actual on test set
plot(
  test_df$predicted_TD, test_df$TD,
  xlab = "Predicted TD (test)", ylab = "Actual TD (test)",
  main = "Test-set: Predicted vs Actual",
  pch = 19
)
abline(0, 1, col = "red")
