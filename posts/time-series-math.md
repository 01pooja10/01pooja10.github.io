Time series forecasting is an important aspect of machine learning and data science which hasn't been given its due. It deals with making predictions for the future using historical data which includes *time component* as a separate feature in the dataset. Time series forecasting is important because we need to identify the measure of significance that needs to be assigned to data from the past, i.e., record of data values from 2 decades ago can't be given equal weightage as compared to data values obtained from 2 years ago.

So this blog is part one of a two-part series and will cover the intuition and math behind the concept of time series forecasting. In the [second part](post.html?post=time-series-code), we'll cover some more math and finally delve into some Python code.

First of all, let's conceptualize what a time series means; a time series is represented by a mathematical relationship between an output or target variable and time.

> In a graphical sense, a time series is basically a set of points that represents a quantity (target variable) and how it changes with time.

These points vary with respect to time in the X-axis and their values in the Y-axis. For example, the monthly beer production [dataset](https://www.kaggle.com/shenba/time-series-datasets?select=monthly-beer-production-in-austr.csv) from Kaggle shows data points plotted over a span of 40 years where the beer production value keeps changing with respect to time.

Now let's dive into the components of a time series and familiarize ourselves with the details.

---

## Components of a time series

### Trend

Trend is referred to as the general tendency of the target variable to increase or decrease with respect to time. Trends basically depict the variance of the target variable in a given time interval.

> Trends can be both linear or non-linear.

How do we distinguish them? When we plot the data points over a given time period, the cluster formed by them will trace a rough outline. If this rough outline represents a straight line, the trend is said to be **linear**; if it resembles a parabola or any other curve, it is said to possess a **non-linear** trend.

### Seasonal variations (seasonality)

Seasonal variations refer to the repeating patterns of change in data points with respect to time. These variations keep occurring in regular intervals and are observed for a period less than a full year.

For example, in India, mangoes are sold the most in the summer season — hence is a good example of a data/target variable (mango sales) which possesses seasonality. Seasonality may be hourly, weekly, monthly or quarterly.

### Cyclic variations

The variations observed in data points over a period of one year constitute cyclic variations. They follow a standard pattern of **peak, recession, trough and recovery**. In simpler terms, let's say that the sales of a given product hits peak during a given month, slowly recedes and hits rock bottom in the next 2 months. However, if the sales value recovers and reaches a decent amount, comparable to its former peak value, in the coming months, it is said to possess a component called cyclic variation.

### Irregular fluctuations

The sudden or unexpected fluctuations in data points at certain time intervals are referred to as irregular or random variations in the time series. These changes are controlled by external or independent forces and can't be predicted. The adverse impact of the Covid-19 pandemic on various businesses and the global economy as a whole, is a good example of an irregular fluctuation.

> To represent irregular fluctuations, we use a term called "white noise", which is just a sequence of random numbers related to our time series in order to take into consideration the occurrence of such unforeseen circumstances. We also forecast errors in an ideal time series using white noise. **It is usually represented by epsilon_t.**

---

## Stationary vs. Non-stationary time series

Quite simply put, a time series with no trend or seasonal variations is said to be **stationary** in nature. This means that our data points possess no change in mean or variance and that the covariance between 2 data points in a given time interval is constant.

Why are stationary time series preferred by most algorithms to make predictions? This is because the data becomes easier to analyze over long periods of time as it won't necessarily keep varying and so, the algorithms can assume that stationary data has been readily served on a plate.

But as we all know, the whole point of a time series' existence is the fact that the target variable is going to keep changing with respect to time. A **non-stationary** time series clearly depicts variations showing that it does in fact contain the trend and seasonality components. But such a non-stationary time series makes it difficult for algorithms to make fair predictions and needs to be made stationary.

So, before moving on, how can we, without graphically plotting a time series, figure out if it is stationary or not?

---

## Test for stationarity

The **Augmented Dickey-Fuller test** (ADF test) is used to tackle this problem and falls under the category of a unit root test. A given time series has one unique characteristic called the unit root which is the coefficient of the data point from a previous time period (lag order). This unit root defines how strong the trend component of a time series is.

> The lag value or lag order of a time series is the value that denotes how many previous time steps need to be traversed.

All we need to know is that if the coefficient of Y(t-1) (value of the data point in the previous time period) i.e. **alpha = 1**, this means that a unit root (alpha) exists and so, the time series is said to be non-stationary.

The ADF test is a modified version of the Dickey-Fuller test and takes into account lag values from many time steps in the past unlike the original Dickey-Fuller test.

> The ADF test churns out an important variable called the p-value and if this value is **less than 0.05**, then the time series is stationary. A p-value greater than 0.05 indicates a non-stationary time series.

Now, let's deal with some statistical algorithms for forecasting.

Note that we'll be covering 3 algorithms in this blog that assume stationarity of the input time series. We will deal with algorithms that allow us to input non-stationary time series in the [second part](post.html?post=time-series-code) of this blog.

---

## 3 algorithms for (stationary) time series forecasting

### Autoregression (AR)

This statistical algorithm uses a dependency or relationship between **observations from the present as compared to their values in the past**. To make it clearer, it uses data values from the past (using lag order) to predict values for the future.

It is represented as **AR(p)** where p represents the lag order of the model. This means that we consider data points from *p* previous time steps.

Let's break this down by decoding the meanings of all the variables:

1. **y_t** — Value of the data point to be predicted
2. **C** — A constant
3. **Phi** — The coefficient of each data point from a previous time period
4. **p** — The lag order used for autoregression
5. **epsilon_t** — The white noise added to the expression to compensate for any random/irregular variations

Let's just say that Autoregression uses *value of past data points* to forecast the future.

### Moving Average (MA)

This is yet another statistical algorithm that, unlike AR, uses the dependency between a **given observation and the residual errors** calculated from observations in previous time steps. Note: the errors refer to the difference between the actual data value at time *t* and the average (moving average) of the preceding data values taken in subsets.

It is represented as **MA(q)** where q is the order of the moving average and represents how many previous time steps to traverse through, in order to obtain their error values.

The values of y_t, c, epsilon_t and theta represent exactly the same quantities they did in the AR equation. The only different variables are the **epsilon(t-1), epsilon(t-2), ..., epsilon(t-q)** terms. These terms are the *error terms obtained from previous data points* and they distinguish the MA algorithm from that of AR.

### Autoregressive Moving Average (ARMA)

As you might've guessed, Autoregressive Moving Average is a combination of both Autoregression and Moving Average algorithms. Hence, it uses both the dependencies i.e. between data values and error values from the past to optimize the predictions.

It is represented as **ARMA(p, q)** where p is the lag order of Autoregression and q is the moving average order.

We are already familiar with all the terms in the equation. The two sigma signs are loops from 1 to p and 1 to q. They are used to sum up all observations (AR) and errors (MA) respectively. So in conclusion, ARMA forecasts future values **using the past data values and the errors calculated** — it combines the methodologies used in AR and MA for making better predictions.

---

## Conclusion

> In this blog we: defined a time series, studied its 4 components, categorized it into stationary/non-stationary types and finally, tried to decode the working of 3 statistical algorithms that forecast the future, only using stationary time series i.e. time series without trend or seasonality.

In the [next blog](post.html?post=time-series-code) (part 2 of this series), we'll study some more algorithms like ARIMA, Seasonal ARIMA (SARIMA) and SARIMAX that don't necessarily assume stationarity of a time series. We'll also delve into some Python code and explore the monthly beer production dataset to successfully implement time series forecasting in Python. Stay tuned for part 2!
