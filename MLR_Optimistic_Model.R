library(readxl)
Atlanta_Falcons_data <- read_excel("Atlanta_Falcons_data.xlsx")
View(Atlanta_Falcons_data)

MLR_Model <-  lm(`TD` ~ Tgt + Rec + `Ctch%` +`Yds`, 
               data = Atlanta_Falcons_data)

summary(MLR_Model)

MLR_Residuals <- MLR_Model$residuals

hist(MLR_Residuals)

qqnorm(MLR_Residuals)
qqline(MLR_Residuals)

Atlanta_Falcons_data$predicted_TD <- predict(
  MLR_Model,
  newdata = Atlanta_Falcons_data
)


plot(
  Atlanta_Falcons_data$predicted_TD,
  Atlanta_Falcons_data$TD,
  xlab = "Predicted TD",
  ylab = "Actual TD",
  main = "Predicted vs Actual Touchdowns",
  pch  = 19
)
abline(0, 1, col = "red")

plot(MLR_Model, which=1)       
