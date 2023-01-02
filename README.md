# Daily oil prices in Ecuador
- Daily oil price. Includes values during both the train and test data timeframes. (Ecuador is an oil-dependent country and it's economical health is highly vulnerable to shocks in oil prices.

## To convert a column into time-series object of daily data
```{r}
prices.ts <- ts(prices.filled, frequency=365)
```

## To plot a time-series object
```{r}
ts.plot(prices.ts, type="o", col=2)
```

## To fit linear-regression model on a time-series object with trend and seasonality parameters
```{r}
tslm(prices.ts~trend+season)
```

## To plot fitted line on time-series object
```{r}
lines(fitted(lm.ts), col=1)
```

## finding the residuals of the time-series linear regression model
```{r}
res.trend <- residuals(lm.ts)
```
Need for visualizing cumulative sum of the data points
## Bass Model
```{r}
bm_prices <- BM(prices.filled, display=T)
```