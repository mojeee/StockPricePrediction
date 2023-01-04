library(readxl)
library(lmtest) # for durbin watson test
library(forecast) # for linear modelling of time-series
library(DIMORA) # for Bass Model, Generalized Bass Model, GGM model
library(tidyr) # for filling missing values
library(tsfknn) # for applying knn on univariate time-series
library(sm) # for local regression and loess
library(splines) # for splines
library(gam)
library(lubridate) # for date data types
library(tidyverse)
library(ggplot2) # for plotting EDA
library(dplyr) # for checking numerical columns


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
stock_volume <- FDX_Xtrain$Volume
plot(stock_volume) 

plot(FDX_Xtrain$Close, type="l")
lines(stock_volume, col=2)

plot(stock_volume, FDX_Xtrain$Close)

x <- stock_volume
y <- FDX_Xtrain$Close

sm.regression(x, y,   h = 0.3, add = T, ngrid=200, col=2, display="se")

# loess
lo1 <- loess.smooth(x,y) #default span= 0.75 where span decides points taken locally for fitting
lines(lo1, col=2)

lo2 <- loess.smooth(x,y,span=0.9) # span is like smoothing parameter h
lines(lo2,col=3)

lo3 <- loess.smooth(x,y,span=0.2)
lines(lo3,col=4)
## 3) regression splines ----

#we select and identify the knots 'equispaced'
xi<-seq(min(x), max(x), length=4)

m1<-lm(y ~ bs(x, knots=xi[2:(length(xi)-1)], degree=3))

###---- for graphical reasons select 200 points where to evaluate the model
xxx<-seq(min(x),max(x),length=200)

#Make predictions by using the 'xxx' points
fit1<-predict(m1, data.frame(x=xxx))
#########
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
lines(p1, col=2)

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

str(FDX_stock_prices)
#transform variable Date in format "data"
FDX_stock_prices$Date <- as.Date(FDX_stock_prices$Date)

#transform Date in numeric 
FDX_stock_prices$Date <-as.numeric(FDX_stock_prices$Date)
summary(FDX_stock_prices$Date)

str(FDX_stock_prices)

# Set train and test
set.seed(1)
train = sample (1:nrow(FDX_stock_prices), 0.7*nrow(FDX_stock_prices))
data.train=FDX_stock_prices[train ,]
data.test=FDX_stock_prices[-train ,]

data.train = data.train[,-c(3,4)]
data.test=FDX_stock_prices[,-c(3,4)]

m1 <- lm(avgprice~., data=data.train)

summary(m1)

m2 <- step(m1, direction="both")
summary(m2)

# library(corrplot)
# corrplot(cor(FDX_stock_prices))

#Prediction
p.lm <- predict(m2, newdata=data.test)
dev.lm <- sum((p.lm-data.test$avgprice)^2)
dev.lm

AIC(m2)

g3 <- gam(avgprice~., data=data.train)

#Show the linear effects 
par(mfrow=c(1,5))
plot(g3, se=T)

g3 <- gam(avgprice~Date, data=data.train)

#Show the linear effects 
par(mfrow=c(1,1))
plot(g3, se=T)

sc = gam.scope(data.train[,c(1,6)], response=2, arg=c("df=2","df=3","df=4")) # degrees of freedom for specifying polynomial degrees
g4<- step.Gam(g3, scope=sc, trace=T) # it avoids vote_classes using the model that we specify
summary(g4)

AIC(g4)

par(mfrow=c(1,2))
plot(g4, se=T)

par(mfrow=c(1,1))
plot(g4, se=T, ask=T)



## 6) Gradient Boosting ----








