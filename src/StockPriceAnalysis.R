library(readxl)
library(lmtest) # for durbin watson test
library(forecast) # for linear modelling of time-series
library(DIMORA) # for Bass Model, Generalized Bass Model, GGM model
library(tidyr) # for filling missing values
library(tsfknn) # for applying knn on univariate time-series
library(sm) # for local regression and loess
library(splines) # for splines
library(gam) # for generalized additive models
library(lubridate) # for date data types
library(tidyverse)
library(ggplot2) # for plotting EDA
library(dplyr) # for checking numerical columns
library(gbm) # for boosting methods


# importing the datasets
FDX_stock_prices = read.csv("../data/FDX.csv")
UPS_stock_prices = read.csv("../data/UPS.csv")

# converting date column to date data type
FDX_stock_prices$Date <- ymd(FDX_stock_prices$Date)
FDX_stock_prices$month <- month(FDX_stock_prices$Date)
FDX_stock_prices$year <- year(FDX_stock_prices$Date)



# deriving binary variables from the data
FDX_stock_prices$is_first_month <- FDX_stock_prices$month == 1
FDX_stock_prices$is_last_month <- FDX_stock_prices$month == 12
FDX_stock_prices$is_winter <- (FDX_stock_prices$month == 1 | FDX_stock_prices$month == 2 | FDX_stock_prices$month == 12)
FDX_stock_prices$is_spring <- (FDX_stock_prices$month == 3 | FDX_stock_prices$month == 4 | FDX_stock_prices$month == 5)
FDX_stock_prices$is_summer <- (FDX_stock_prices$month == 6 | FDX_stock_prices$month == 7 | FDX_stock_prices$month == 8)
FDX_stock_prices$is_fall <- (FDX_stock_prices$month == 9 | FDX_stock_prices$month == 10 | FDX_stock_prices$month == 11)

# EDA ----
str(FDX_stock_prices)

# missing values check
sum(is.na(FDX_stock_prices))
sum(is.na(UPS_stock_prices))

# plotting close, open and high prices of the stock over time
tt <- 1:NROW(FDX_stock_prices)
tt1<- (NROW(FDX_stock_prices) - NROW(UPS_stock_prices) + 1):NROW(FDX_stock_prices)

# plot for showing the trend of the time series
par(mfrow=c(1,1))
plot(tt, FDX_stock_prices$Close, xlab="time", ylab="closing stock price", main="distribution of closing stock prices over time", col=1,type="l", xaxt="n")
lines(tt, FDX_stock_prices$Open,col=2)
axis(1, at = seq(1,length(FDX_stock_prices$Close),by=48), labels=FDX_stock_prices$Date[seq(1,length(FDX_stock_prices$Close),by=48)])
legend(1, 310, legend=c("Close prices", "open prices"),col=c("black", "red"), lty=c(1,1), cex=1)

# seasonal plot (fix the labels of years)
# time series object of past 10 years
FDX_close_prices_ts <- ts(FDX_stock_prices$Close[-c(1:264)], frequency=12, start=c(2013, 1))
ggseasonplot(FDX_close_prices_ts, year.labels=TRUE, year.labels.left=TRUE) +
ylab("dollars($)") +
ggtitle("Seasonal plot: FedX close prices")

# scatter plot of close prices and volume
plot(log(FDX_stock_prices$Volume), FDX_stock_prices$Close, type="p", xlab="Volume", ylab="Close prices",main="Volume vs Close prices")


# barchart for volumes of each month in an year
# Create a new column that combines the month and year into a single factor
FDX_stock_prices$month_year <- as.factor(paste(FDX_stock_prices$month, FDX_stock_prices$year, sep = "-"))

# Create the bar chart
FDX_stock_prices$log_volume <- log(FDX_stock_prices$Volume)

ggplot(FDX_stock_prices[-c(1:264),], aes(x = month_year, y = log_volume, fill=month_year)) +
  geom_col() +
  labs(x = "Month-Year", y = "Volume")

# correlogram
Acf(FDX_stock_prices$Close)

# converting years to integer 1=start year 
FDX_stock_prices$year <- FDX_stock_prices$year - 1990

# adding the variables by replacing true with one and false with zero
FDX_stock_prices$is_first_month_num <- ifelse(FDX_stock_prices$is_first_month, 1, 0)
FDX_stock_prices$is_last_month_num <- ifelse(FDX_stock_prices$is_last_month, 1, 0)
FDX_stock_prices$is_winter_num <- ifelse(FDX_stock_prices$is_winter, 1, 0)
FDX_stock_prices$is_spring_num <- ifelse(FDX_stock_prices$is_spring, 1, 0)
FDX_stock_prices$is_summer_num <- ifelse(FDX_stock_prices$is_summer, 1, 0)
FDX_stock_prices$is_fall_num <- ifelse(FDX_stock_prices$is_fall, 1, 0)

# splitting the data into last year for testing and remaining for training
FDX_train <- FDX_stock_prices[1:372,]
FDX_test <- FDX_stock_prices[373:385,]

UPS_train <- UPS_stock_prices[1:372,]
UPS_test <- UPS_stock_prices[373:385,]

# taking close price of the day as output variable
FDX_Xtrain <- select_if(FDX_train, is.numeric)
UPS_Xtrain <- select_if(UPS_train, is.numeric)

FDX_Xtest <- select_if(FDX_test, is.numeric)
UPS_Xtest <- select_if(UPS_test, is.numeric)


#plotting
# col - color of the points
# cex - size of the points
# type - to plot the line
# pch - shape of the point (0 to 25)
# lwd - line width
# lty - line type(0 to 6)



# models
# Parametric ----

# correlation plot 
library(corrplot)
corrplot(cor(FDX_Xtrain)) # no significant correlation between the predictors as well as on the output

# removing unnecessary columns
removed_columns <- c("Open", "High", "Low", "Adj.Close", "month", "year", "log_volume")
FDX_Xtrain <- select(FDX_Xtrain, -one_of(removed_columns))

FDX_Xtest <- select(FDX_Xtest, -one_of(removed_columns))

# minmax scaling volume and close prices
min_close <- min(FDX_Xtrain$Close)
max_close <- max(FDX_Xtrain$Close)
min_volume <- min(FDX_Xtrain$Volume)
max_volume <- max(FDX_Xtrain$Volume)
  
FDX_Xtrain$Close <- (FDX_Xtrain$Close - min_close)/(max_close - min_close)
FDX_Xtrain$Volume <- (FDX_Xtrain$Volume - min_volume)/(max_volume - min_volume)

FDX_Xtest$Close <- (FDX_Xtest$Close - min_close)/(max_close - min_close)
FDX_Xtest$Volume <- (FDX_Xtest$Volume - min_volume)/(max_volume - min_volume)

# complete transformed dataset
complete_data <- rbind(FDX_Xtrain, FDX_Xtest)


## 1) Linear Regression ----
par(mfrow=c(1,1))
Acf(FDX_Xtrain$Close) # we can see the uptrend present in the prices

# multiple linear regression 
lm_multi <- lm(Close~., data = FDX_Xtrain)
summary(lm_multi) # NA for is_fall_num is due to multi-collinearity problem and None of the derived variables affect the close prices

# time series values
tt_train <- 1:NROW(FDX_Xtrain)
tt_test <- 373:385

# linear model with trend
lm <- lm(FDX_Xtrain$Close~tt_train)
summary(lm) # the prices are significantly dependent on time

# plotting regression fit
plot(tt_train, FDX_Xtrain$Close, type='l')
abline(lm,col=2)

# durbin watson test for auto-correlation significance in residuals
dwtest(lm) # DW is close to zero which means positive correlation

# residual analysis
res <- residuals(lm)
plot(res)
lines(res, type='l', col=2)

Acf(res) # until lag 17 there is a significant positive correlation


# creating a time series object
FDX_close_prices_ts <- ts(FDX_Xtrain$Close, frequency=12)

# linear model with trens and seasonality
lm_ts <- tslm(FDX_close_prices_ts~trend+season)
summary(lm_ts) # no seasonal component is significant 

# linear model with trend
lm_ts_trend <- tslm(FDX_close_prices_ts~trend)

# predictions for linear model
pred <- forecast(lm_ts_trend, h=13)

# plotting the predictions
plot(tt_test, FDX_Xtest$Close, type="l", xlab="Date", ylab="FDX closing prices",main="LR predictions for test set", xaxt="n")
axis(1, at = seq(373,385,by=1), labels=FDX_stock_prices$Date[seq(373,385,by=1)])
lines(tt_test, pred$mean, col=2)
legend(374, 0.55, legend=c("Actual values", "Predictions"),col=c("black", "red"), lty=c(1,1), cex=1)


## 2) Bass model ----
bm_prices <- BM(FDX_Xtrain$Close, display=T)
summary(bm_prices) # significant estimates: we can see lower and upper bounds have same signs

# residual analysis
res_BM <- residuals(bm_prices)
plot(res_BM)
lines(res_BM, type='l', col=2, lwd=2) # we can clearly see pattern in the residuals

acf(res_BM)

# prediction
pred_bm_prices <- predict(bm_prices, newx = c(1:385))
pred_inst_prices <- make.instantaneous(pred_bm_prices)
plot(complete_data$Close, type="b", xlab="Year", ylab="Closing price", pch=16, lty=3, xaxt="n", cex=0.6, main="BM predictions for test set")
axis(1, at = seq(1,385,by=48), labels=FDX_stock_prices$Date[seq(1,385,by=48)])
lines(pred_inst_prices, lwd=2, col=3)
legend(2, 0.8, legend=c("Actual values", "Predictions"),col=c("black", "green"), pch=c(21,NA), pt.bg = c("black",NA),lty=c(NA,1), cex=1)


## 3) Generalized Bass Model ----
# single shock model
GMBr1ps <- GBM(FDX_Xtrain$Close, shock = "rett", nshock = 1, prelimestimates = c(2.789169e+02, 1.139732e-04, 1.055485e-02, 280, 310, -0.1 ))
summary(GMBr1ps)

# residual analysis
res_GBMr1ps <- residuals(GMBr1ps)
plot(res_GBMr1ps)
lines(res_GBMr1ps, type='l', col=2, lwd=2) # we can clearly see pattern in the residuals

acf(res_GBMr1ps)

# prediction
pred_gbmr1_prices <- predict(GMBr1ps, newx = c(1:385))
pred_gbmr1_inst_prices <- make.instantaneous(pred_gbmr1_prices)
plot(complete_data$Close, type="b", xlab="Year", ylab="Closing price", pch=16, lty=3, xaxt="n", cex=0.6, main="BM predictions for test set")
axis(1, at = seq(1,385,by=48), labels=FDX_stock_prices$Date[seq(1,385,by=48)])
lines(pred_gbmr1_inst_prices, lwd=2, col=3)
legend(2, 0.8, legend=c("Actual values", "Predictions"),col=c("black", "green"), pch=c(21,NA), pt.bg = c("black",NA),lty=c(NA,1), cex=1)

# two rectangular shocks
GMBr2ps <- GBM(FDX_Xtrain$Close, shock = "rett", nshock = 2, prelimestimates = c(2.789169e+02, 1.139732e-04, 1.055485e-02, 280, 310, -0.1, 330, 350, -0.1))
summary(GMBr2ps)

# residual analysis
res_GBMr2ps <- residuals(GMBr2ps)
plot(res_GBMr2ps)
lines(res_GBMr2ps, type='l', col=2, lwd=2) # we can clearly see pattern in the residuals

acf(res_GBMr2ps)

# prediction
pred_gbmr2_prices <- predict(GMBr2ps, newx = c(1:385))
pred_gbmr2_inst_prices <- make.instantaneous(pred_gbmr2_prices)
plot(complete_data$Close, type="b", xlab="Year", ylab="Closing price", pch=16, lty=3, xaxt="n", cex=0.6, main="BM predictions for test set")
axis(1, at = seq(1,385,by=48), labels=FDX_stock_prices$Date[seq(1,385,by=48)])
lines(pred_gbmr2_inst_prices, lwd=2, col=3)
legend(2, 0.8, legend=c("Actual values", "Predictions"),col=c("black", "green"), pch=c(21,NA), pt.bg = c("black",NA),lty=c(NA,1), cex=1)

# exponential shock (a1, b1 and c1 have different meaning compared with rectangular shock)
GBMe1ps <- GBM(FDX_Xtrain$Close, shock = "exp", nshock = 1, prelimestimates = c(2.789169e+02, 1.139732e-04, 1.055485e-02, 200, -0.1, 0.3))
summary(GBMe1ps) # not better than rect shocks that why no point of predictions

# residual analysis
res_GBMe1ps <- residuals(GBMe1ps)
plot(res_GBMe1ps)
lines(res_GBMe1ps, type='l', col=2, lwd=2) # we can clearly see pattern in the residuals

acf(res_GBMe1ps)

## 4) GGM ----
GGM_ps <- GGM(FDX_Xtrain$Close, prelimestimates = c(2.789169e+02, 0.001, 0.01, 1.139732e-04, 1.055485e-02))
summary(GGM_ps)

# residual analysis
res_GGM <- residuals(GGM_ps)
plot(res_GGM)
lines(res_GGM, type='l', col=2, lwd=2) # we can clearly see pattern in the residuals

acf(res_GGM)


## 5) competition modelling ----
plot(tt, FDX_avgprice, xlab="time", ylab="avg stock price", main="distribution of avg stock prices over time", col=1,type="l", xaxt="n")
lines(tt1, UPS_avgprice,col=2)

ucrcdFEDUPS <- UCRCD(FDX_avgprice, UPS_avgprice)
summary(ucrcdFEDUPS)

## 6) ARIMA ----
first_diff <- diff(FDX_Xtrain$Close)
acf(first_diff)
acf(diff(FDX_Xtrain$Close, differences = 2))

# checking stationarity using ADF test or unit root test
library(tseries)
adf.test(first_diff) # stationary series

# for checking first significant lag to pick p of AR
pacf(first_diff)

fit1 <- arima(FDX_Xtrain$Close, order=c(7,1,0))

arima(FDX_close_prices, order=c(7,1,4))

# Generate forecasts for the next 5 time steps
forecasts <- predict(fit1, n.ahead = 13)


print(forecasts$pred)

plot(tt, complete_data$Close, xlab="time", ylab="closing stock price", main="distribution of closing stock prices over time", col=1,type="l", xaxt="n")
lines(forecasts$pred, col=2)


auto.a <- auto.arima(FDX_Xtrain$Close)
auto.a

autoplot(forecast(auto.a))
checkresiduals(auto.a)

## 7) ARIMAX ----
auto.a<- auto.arima(FDX_Xtrain$Close, xreg=FDX_Xtrain$Volume) 
AIC(auto.a)

auto.a<- auto.arima(FDX_Xtrain$Close, xreg=FDX_Xtrain$is_first_month_num) 
AIC(auto.a)

auto.a<- auto.arima(FDX_Xtrain$Close, xreg=FDX_Xtrain$is_last_month_num) 
AIC(auto.a)

auto.a<- auto.arima(FDX_Xtrain$Close, xreg=FDX_Xtrain$is_winter_num) 
AIC(auto.a)

auto.a<- auto.arima(FDX_Xtrain$Close, xreg=FDX_Xtrain$is_spring_num) 
AIC(auto.a)

auto.a<- auto.arima(FDX_Xtrain$Close, xreg=FDX_Xtrain$is_summer_num)
AIC(auto.a)

auto.av <- auto.arima(FDX_Xtrain$Close, xreg=FDX_Xtrain$is_fall_num)
AIC(auto.av)

checkresiduals(auto.a)

## 9) Exponential smoothing

# simple smoothing - series with no trend or seasonality
ses.prices <- ses(FDX_Xtrain$Close, alpha=0.2, h=20)
autoplot(ses.prices)
summary(ses.prices)

ses.prices <- ses(FDX_Xtrain$Close, alpha=0.5, h=20)
autoplot(ses.prices)
summary(ses.prices)

ses.prices <- ses(FDX_Xtrain$Close, alpha=0.8, h=20)
autoplot(ses.prices)
summary(ses.prices)

# since we have the trend removing trend and applying simple smoothing
ses.prices <- ses(first_diff, alpha=0.2, h=20)
autoplot(ses.prices)
summary(ses.prices)

ses.prices <- ses(first_diff, alpha=0.5, h=20)
autoplot(ses.prices)
summary(ses.prices)

ses.prices <- ses(first_diff, alpha=0.8, h=20)
autoplot(ses.prices)
summary(ses.prices)


# removing trend from test set
fdx.dif <- diff(FDX_Xtrain$Close)
fdx.dif.test <- diff(FDX_Xtest$Close)

# comparing our model
alpha <- seq(.01, .99, by = .01)
RMSE <- NA
for(i in seq_along(alpha)) {
  fit <- ses(fdx.dif, alpha = alpha[i],
             h = 12)
  RMSE[i] <- accuracy(fit,
                      fdx.dif.test)[2,2]
}

# convert to a data frame and
# identify min alpha value
alpha.fit <- data_frame(alpha, RMSE)
alpha.min <- filter(alpha.fit,
                    RMSE == min(RMSE))

# plot RMSE vs. alpha
ggplot(alpha.fit, aes(alpha, RMSE)) +
  geom_line() +
  geom_point(data = alpha.min,
             aes(alpha, RMSE),
             size = 2, color = "red")

# refit model with alpha = .75
ses.fdx.opt <- ses(fdx.dif,
                    alpha = .01,
                    h = 50)


# Non-parametric ----
## 1) KNN ----
# timeS : the time series to be forecast.
# h : the forecast horizon, that is, the number of future values to be predicted.
# lags : an integer vector indicating the lagged values of the target used as features in the examples (for instance, 1:2 means that lagged values 1 and 2 should be used).
# k : the number of nearest neighbors used by the KNN model.
pred <- knn_forecasting(FDX_close_prices_ts, h = 10, lags = 1:2, k = 5, transform = "none")
knn_examples(pred)

pred$prediction

plot(pred)

## 2) local regression ----

x <- FDX_Xtrain$Volume
y <- FDX_Xtrain$Close

plot(x, y)
sm.regression(x, y,   h =0.2, add = T, col=1, ngrid=300) # add is to add intercept, ngrid is for optimization
sm.regression(x, y,   h =0.5, add = T, col=2, ngrid=300) # h is to consider no of points for fitting
sm.regression(x, y,   h =0.6, add = T, col=3, ngrid=300)
sm.regression(x, y,   h =0.9, add = T, col=4, ngrid=300)


# loess
# Local regression is a type of nonparametric regression that is 
# used to fit a smooth curve to a scatterplot of data. 
# It is particularly useful for data that may have a nonlinear relationship or for data with a lot of noise.

plot(x, y, xlab="Volume", ylab="Close price")
lo1 <- loess.smooth(x,y) #default span= 0.75 where span decides points taken locally for fitting
lines(lo1, col=2)

lo2 <- loess.smooth(x,y,span=0.9) # span is like smoothing parameter h
lines(lo2,col=3)

lo3 <- loess.smooth(x,y,span=0.2)
lines(lo3,col=4)

## 3) regression splines ----

plot(x, y, xlab="Volume", ylab="Close price")
#we select and identify the knots 'equispaced'
xi<-seq(min(x), max(x), length=4)

m1<-lm(y ~ bs(x, knots=xi[2:(length(xi)-1)], degree=2))

###---- for graphical reasons select 200 points where to evaluate the model
xxx<-seq(min(x),max(x),length=200)

#Make predictions by using the 'xxx' points
fit1<-predict(m1, data.frame(x=xxx))

plot(x,y,xlab="engine size", ylab="distance")
lines(xxx,fit1,col=2)

######vertical lines to indicate the knots
abline(v=xi[2], lty=3)
abline(v=xi[3], lty=3)

# first model with 2 internal knots
m1<-lm(y~bs(x, df=5, degree=3)) # no of knots = df-degree
fit1<-predict(m1, data.frame(x=xxx))
lines(xxx,fit1,col=4)

# second model with no internal knots 
m2 <- lm(y ~ bs(x, df=3, degree=3)) 
fit2<-predict(m2,data.frame(x=xxx))
lines(xxx,fit2,col=3)

# Third model with 17 knots 
m3<-lm(y~bs(x,df=20,degree=3))
fit3<-predict(m3,data.frame(x=xxx))
lines(xxx,fit3,col=2)

## 4) smoothing splines ----
plot(x, y)
s <- smooth.spline(x,y)
lines(s)

s1 <- smooth.spline(x,y, lambda=0.0001)
lines(s1, col=2)

p1<- predict(s1, x=xxx)
lines(p1, col=4)

s2 <- smooth.spline(x,y, lambda=0.00001)
p2<- predict(s2, x=xxx)
lines(p2, col=3)

# Model 3
s3 <- smooth.spline(x,y, lambda=0.01)
p3<- predict(s3, x=xxx)
lines(p3, col=4)

# Model 4
s4 <- smooth.spline(x,y, lambda=1)
p4<- predict(s4, x=xxx)
lines(p4, col=5)

# Model 5
s5 <- smooth.spline(x,y, lambda=0.00000001)
p5<- predict(s5, x=xxx)
lines(p5, col=6)


## 5) Generative Additive Models ----

#Show the linear effects 
g1 <- gam(Close~tt_train+Volume, data=FDX_Xtrain)
par(mfrow=c(1,2))
plot(g1, se=T)
summary(g1)


####GAM with splines performs better###

# non-linear effect of smoothing basis on time and linear effect of volume
g2 <- gam(Close~s(tt_train)+Volume, data=FDX_Xtrain)
par(mfrow=c(1,2))
plot(g2, se=T)
summary(g2)

# non-linear effect of both the smoothed parameters basis
g3 <- gam(Close~s(tt_train)+s(Volume), data=FDX_Xtrain)
par(mfrow=c(1,2))
plot(g3, se=T)
summary(g3)

# non-linear effect of both the loess parameters basis
g4 <- gam(Close~lo(tt_train)+lo(Volume), data=FDX_Xtrain)
par(mfrow=c(1,2))
plot(g4, se=T)
summary(g4)

# linear effect of all the predictors
g5 <- gam(Close~tt_train+., data=FDX_Xtrain)
par(mfrow=c(1,8))
plot(g5, se=T)
summary(g5)

#######perform analysis of residuals
par(mfrow=c(1,1))
tsdisplay(residuals(g1))
aar1<- auto.arima(residuals(g1))

plot(FDX_Xtrain$Close, type="l")
lines(fitted(aar1)+ fitted(g1), col=4)
summary(aar1)

# predictions
for1 <- forecast(aar1)
plot(for1)


## 6) Gradient Boosting ----

summary(FDX_Xtrain)

# making binary variables to factors
FDX_Xtrain[,c(3:8)]= lapply(FDX_Xtrain[,c(3:8)],factor)
FDX_Xtest[,c(3:8)]= lapply(FDX_Xtest[,c(3:8)],factor)

str(FDX_Xtrain)

# 1 Boosting
boost.movies=gbm(Close~., data=FDX_Xtrain, 
                 distribution="gaussian", n.trees=5000, interaction.depth=1)
boost.movies
#
#for the plot
par(mfrow=c(1,1))
#
#plot of training error
plot(boost.movies$train.error, type="l", ylab="training error")

#always decreasing with increasing number of trees
#
#
#relative influence plot
summary(boost.movies) 
#let us modify the graphical parameters to obtain a better plot
#
#more space on the left
#
# default vector of parameters
mai.old<-par()$mai
mai.old
#new vector
mai.new<-mai.old
#new space on the left
mai.new[2] <- 2.5 
mai.new
#modify graphical parameters
par(mai=mai.new)
summary(boost.movies, las=1) 
#las=1 horizontal names on y
summary(boost.movies, las=1, cBar=7) 
#cBar defines how many variables
#back to orginal window
par(mai=mai.old)



# test set prediction for every tree (1:5000)
yhat.boost=predict(boost.movies, newdata=FDX_Xtest, n.trees=1:5000)

# calculate the error for each iteration
#use 'apply' to perform a 'cycle for' 
# the first element is the matrix we want to use, 2 means 'by column', 
#and the third element indicates the function we want to calculate

err = apply(yhat.boost, 2, function(pred) mean((FDX_Xtest$Close - pred)^2))
#
plot(err, type="l")

# error comparison (train e test)
plot(boost.movies$train.error, type="l", ylim = c(0,0.2))
lines(err, type="l", col=2)

#minimum error in test set
best=which.min(err)
abline(v=best, lty=2, col=4)
#
min(err) #minimum error


# 2 Boosting - Deeper trees
boost.movies=gbm(Close~., data=FDX_Xtrain, 
                 distribution="gaussian", n.trees=5000, interaction.depth=4)

plot(boost.movies$train.error, type="l")

#par(mai=mai.new)

summary(boost.movies, las=1, cBar=7)  

#par(mai=mai.old)

yhat.boost=predict(boost.movies ,newdata=FDX_Xtest,n.trees=1:5000)
err = apply(yhat.boost,2,function(pred) mean((FDX_Xtest$Close-pred)^2))
plot(err, type="l")


plot(boost.movies$train.error, type="l", ylim = c(0,0.2))
lines(err, type="l", col=2)
best=which.min(err)
abline(v=best, lty=2, col=4)
min(err)


# 3 Boosting - Smaller learning rate 

boost.movies=gbm(Close~., data=FDX_Xtrain, 
                 distribution="gaussian", n.trees=5000, interaction.depth=1, shrinkage=0.01)
plot(boost.movies$train.error, type="l")

par(mai=mai.new)

summary(boost.movies, las=1, cBar=7) 
par(mai=mai.old)

yhat.boost=predict(boost.movies ,newdata=FDX_Xtest,n.trees=1:5000)
err = apply(yhat.boost,2,function(pred) mean((FDX_Xtest$Close-pred)^2))
plot(err, type="l")


plot(boost.movies$train.error, type="l", ylim=c(0, 0.2))
lines(err, type="l", col=2)
best=which.min(err)
abline(v=best, lty=2, col=4)
min(err)


# 4 Boosting - combination of previous models
boost.movies=gbm(Close~., data=FDX_Xtrain, 
                 distribution="gaussian",n.trees=5000, interaction.depth=4, shrinkage=0.01)

plot(boost.movies$train.error, type="l")
#

par(mai=mai.new)

summary(boost.movies, las=1, cBar=7) 

par(mai=mai.old)

yhat.boost=predict(boost.movies ,newdata=FDX_Xtest,n.trees=1:5000)
err = apply(yhat.boost, 2, function(pred) mean((FDX_Xtest$Close-pred)^2))
plot(err, type="l")


plot(boost.movies$train.error, type="l", ylim=c(0,0.2))
lines(err, type="l", col=2)
best=which.min(err)
abline(v=best, lty=2, col=4)
err.boost= min(err)


##Comparison of models in terms of residual deviance
predictions <- predict.gbm(boost.movies, newdata = FDX_Xtest)
dev.gbm<- (sum((predictions-FDX_Xtest$Close)^2))

dev.gbm


boost.movies
# partial dependence plots
plot(boost.movies, i.var=1, n.trees = best)
plot(boost.movies, i.var=2, n.trees = best)
plot(boost.movies, i.var=5, n.trees = best)
plot(boost.movies, i.var=c(1,5), n.trees = best) #bivariate (library(viridis) may be necessary)
#
plot(boost.movies, i.var=3, n.trees = best) # categorical
plot(boost.movies, i.var=6, n.trees = best)

plot(boost.movies, i=23, n.trees = best)# categorical
plot(boost.movies, i=17, n.trees = best) #no effect


## 7) Stepwise Regression ----

m1 <- lm(Close~., data=FDX_Xtrain)

summary(m1)

m2 <- step(m1, direction="both")
summary(m2)



#Prediction
p.lm <- predict.lm(m2, newdata=FDX_Xtest)
dev.lm <- sum((p.lm-FDX_Xtest$Close)^2)
dev.lm

AIC(m2)

## 8) stepwise GAM ----
g3 <- gam(Close~., data=FDX_Xtrain)

#Show the linear effects 
par(mfrow=c(2,4))
plot(g3, se=T) 

#Perform stepwise selection using gam swscope
#Values for df should be greater than 1, with df=1 implying a linear fit

sc = gam.scope(FDX_Xtrain[,-1], response=1, arg=c("df=2","df=3","df=4")) # degrees of freedom for specifying polynomial degrees
g4<- step.Gam(g3, scope=sc, trace=T) # it avoids vote_classes using the model that we specify
summary(g4)

AIC(g4)

par(mfrow=c(2,4))
plot(g4, se=T)

#Prediction

# make some variables factor
p.gam <- predict(g4,newdata=FDX_Xtest[,c(2,5,6,7,8)])     
dev.gam <- sum((p.gam-FDX_Xtest$Close)^2)
dev.gam

# 9) LSTM ----
library(keras)
library(tensorflow)

scale_factors <- c(mean(FDX_train$Close), sd(FDX_train$Close))

scaled_train <- (FDX_train$Close - scale_factors[1]) / scale_factors[2]

prediction <- 12
lag <- prediction

scaled_train <- as.matrix(scaled_train)

# we lag the data 11 times and arrange that into columns
x_train_data <- t(sapply(
  1:(length(scaled_train) - lag - prediction + 1),
  function(x) scaled_train[x:(x + lag - 1), 1]
))

# now we transform it into 3D form
x_train_arr <- array(
  data = as.numeric(unlist(x_train_data)),
  dim = c(
    nrow(x_train_data),
    lag,
    1
  )
)

y_train_data <- t(sapply(
  (1 + lag):(length(scaled_train) - prediction + 1),
  function(x) scaled_train[x:(x + prediction - 1)]
))

y_train_arr <- array(
  data = as.numeric(unlist(y_train_data)),
  dim = c(
    nrow(y_train_data),
    prediction,
    1
  )
)

x_test <- FDX_Xtest$Close
# scale the data with same scaling factors as for training
x_test_scaled <- (x_test - scale_factors[1]) / scale_factors[2]

# this time our array just has one sample, as we intend to perform one 12-months prediction
x_pred_arr <- array(
  data = x_test_scaled,
  dim = c(
    1,
    lag,
    1
  )
)

lstm_model <- keras_model_sequential()

lstm_model %>%
  layer_lstm(units = 50, # size of the layer
             batch_input_shape = c(1, 12, 1), # batch size, timesteps, features
             return_sequences = TRUE,
             stateful = TRUE) %>%
  # fraction of the units to drop for the linear transformation of the inputs
  layer_dropout(rate = 0.5) %>%
  layer_lstm(units = 50,
             return_sequences = TRUE,
             stateful = TRUE) %>%
  layer_dropout(rate = 0.5) %>%
  time_distributed(keras::layer_dense(units = 1))


lstm_model %>%
  compile(loss = 'mae', optimizer = 'adam', metrics = 'accuracy')

summary(lstm_model)

lstm_model %>% fit(
  x = x_train_arr,
  y = y_train_arr,
  batch_size = 1,
  epochs = 20,
  verbose = 0,
  shuffle = FALSE
)

lstm_forecast <- lstm_model %>%
  predict(x_pred_arr, batch_size = 1) %>%
  .[, , 1]

# we need to rescale the data to restore the original values
lstm_forecast <- lstm_forecast * scale_factors[2] + scale_factors[1]


fitted <- predict(lstm_model, x_train_arr, batch_size = 1) %>%
  .[, , 1]

if (dim(fitted)[2] > 1) {
  fit <- c(fitted[, 1], fitted[dim(fitted)[1], 2:dim(fitted)[2]])
} else {
  fit <- fitted[, 1]
}

# additionally we need to rescale the data
fitted <- fit * scale_factors[2] + scale_factors[1]


# I specify first forecast values as not available
fitted <- c(rep(NA, lag), fitted)

lstm_forecast <- ts(lstm_forecast, start=c(2022,1), end=c(2022,12), frequency = 12)
input_ts <- ts(FDX_train$Close, start=c(1991,1), end=c(2021,12), frequency = 12)

forecast_list <- list(
  model = NULL,
  method = "LSTM",
  mean = lstm_forecast,
  x = input_ts,
  fitted = fitted,
  residuals = as.numeric(input_ts) - as.numeric(fitted)
)

class(forecast_list) <- "forecast"


forecast::autoplot(forecast_list)

