In the last blog, we discussed three statistical algorithms that necessarily required a univariate stationary time series as input, namely; Autoregression (AR), Moving Average (MA) and Autoregressive Moving Average (ARMA). For more details regarding a time series and such statistical models, refer to the [Math Edition blog](post.html?post=time-series-math). Now let's dive into the remaining 3 algorithms.

> Note: Algorithms like ARIMA, SARIMA etc. actually make use of a stationary time series in their calculations. Wait, what? So how will components like trend or seasonality get included when forecasting a time series? This is why we generally pre-process the time series that is to be fed as an input to the algorithm. This preprocessing step ensures that the required component (increasing/decreasing trend, seasonality, etc.) is included as an influencing factor and that the time series is forecasted accordingly.

---

## 3 more algorithms for forecasting a time series

### Autoregressive Integrated Moving Average (ARIMA)

ARIMA is a modified version of ARMA with a **differencing parameter**. This means that any time series with a trend component can be used as an input to make this algorithm work. This also means that algorithms don't have to deal with stationary data anymore.

> The trend component denotes the general tendency of the output variable to increase or decrease with time.

To ensure that this is included as a factor in the mathematical equation, we simply subtract the previous time step's value of the output variable from the current value. This is referred to as **differencing**. It can also hold true for multiple subtractions from the current time step.

> ARIMA is usually represented as **ARIMA(p,d,q)**.

We are already familiar with the terms p and q which are the lag and moving average orders for AR and MA algorithms respectively. So the new variable **d** is nothing but the differencing parameter. This variable is going to represent how many previous time steps need to be subtracted from the current time step.

### Seasonal Autoregressive Integrated Moving Average (SARIMA)

ARIMA is superseded by SARIMA in the fact that SARIMA can also accept a seasonal time series. So what other parameters can achieve such a feat?

> SARIMA is represented as **SARIMA(p,d,q)(P,D,Q,m)**

These 4 new capital letters are used to represent an entirely new dimension exclusively for taking seasonality into consideration.

1. **P** is the seasonal Autoregressive lag order
2. **D** is the seasonal differencing parameter
3. **Q** is the seasonal moving average order
4. **m** is the number of time steps for one single seasonal period

So this can be a lot to take in but when stripped down, it is just another model that can accurately take into consideration, the fact that a time series has repeating patterns.

### Seasonal Autoregressive Integrated Moving Average with Exogenous Variables (SARIMAX)

So what does exogenous mean? Exogenous means something that comes from outside or **an external time series** in our case. When we want to include a separate parallel time series, we use SARIMAX. It is also considered to be a slightly more efficient model for forecasting a time series with accuracy.

> SARIMAX is represented as **SARIMAX(p,d,q)(P,D,Q,m)**

This model is represented in a way similar to that of SARIMA and consists of the same 7 parameters. But with respect to Python, a specific hyperparameter called **exog** can be set to point to the name of a new, parallel time series in case we need to include one or it can be set to `None` to ignore the consideration of another time series.

Now let's explore the **beer production** [dataset](https://www.kaggle.com/shenba/time-series-datasets?select=monthly-beer-production-in-austr.csv) from Kaggle and forecast the time series with some Python code!

---

## Time to get our hands dirty with some code

The Python library called [statsmodels](https://www.statsmodels.org/stable/index.html) contains all the necessary algorithms needed for performing time series forecasting. Our dataset consists of a month column which represents the time dimension and a column to depict the beer production value which will inevitably be our dynamic target variable.

Let's begin by exploring and modifying our dataset. If we take a closer look at the dataset, we can clearly notice that the timeline, consisting of the year and the day for the specific month (YYYY-DD), isn't in the format that is generally accepted by Python's models and so we convert it into an acceptable format (YYYY-DD-MM) before proceeding.

Firstly, we check for null values in the dataset and upon finding none, we move on to find the maximum amount of beer produced and it turns out to be: **217.8 million litres**. This was done using the `max()` functionality.

Further, we make a simple plot using matplotlib. The value of beer production on the Y-axis is plotted with respect to time on the X-axis. In the resulting plot, we can notice a clear pattern in the target variable:

- Beer production remains somewhat low during the first 10 years (till 1965)
- Then shows an increasing trend
- Hits peak by 1978-79 (which is 217.8 in value)
- Only to decrease again

The values visibly keep increasing and decreasing with respect to time. **Thus we can conclude that our dataset isn't stationary.**

We then make a few more plots to gain more insight into what our data represents and how it contains different components that come together to form a time series.

### Auto-correlation plot

A correlation plot is used to represent how interconnected 2 variables are. If they rise/fall in the same direction, it means that they are positively correlated else if they rise/fall in opposite directions, they are negatively correlated. An auto-correlation plot works in the same way except that it compares a given variable with itself (using values from previous time steps) to display repetitive patterns or randomness in a time series.

> In its essence, an auto-correlation plot uses the target variable's current value in comparison to its value from a previous time step. The number of time steps traversed is represented as "Lag" in the X-axis.

### Decomposition plot

The decomposition plot is significant since it represents the different components of a time series in our dataset namely: **trend, seasonality, random errors (residue) and cyclic variations**. This can be used to check the stationarity of a time series.

None of the plots seem to be a straight line. This indicates that the dataset possesses all the components and hence, is a non-stationary time series.

### Augmented Dickey-Fuller test

The augmented Dickey-Fuller test, as explained in the [Math Edition blog](post.html?post=time-series-math), is another useful criterion that determines stationarity. Since the output or the p-value is > 0.05, the dataset (time series) is **non-stationary**.

---

## Fitting the SARIMAX model

Now, let's split our dataset into training and testing sets before fitting it to the SARIMAX statistical model from Python's [statsmodels](https://www.statsmodels.org/stable/index.html) library. The *exog* parameter is set to `None` so that the model won't demand an extra time series. **We use the SARIMAX model since our dataset contains both trend and seasonality.**

The lag and moving average order values, seasonality, m, etc. can be set using either the grid search algorithm or the [auto_arima](https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html) function provided by the [pmdarima](https://pypi.org/project/pmdarima/) library in Python, to find the most optimal values of each parameter. Finding the optimal values of the different orders/parameters helps us achieve the best results.

The output obtained is a summary of the working of our algorithm with its optimal parameter values (chosen by auto_arima):

- The **coef** column denotes the weights of each term
- The **P>|z|** or the p-value column determines the significance of each weight term
- A p-value < 0.05 is ideal (as it indicates a stationary time series) and since that is the case for almost all the terms, the model has fit our dataset successfully

The residual and density plots represent the residual errors and the density distribution of the results. Such plots must ideally contain **uniform variance and zero mean**.

---

## Predictions and results

Now it is time to explore the results and the performance of our model. To further explain the model and interpret its working, we can make use of the `predict()` function to forecast the time series for a given set of inputs obtained from testing data.

To make predictions, we start from the end of the training set (which is the beginning of the test set) and visit each value till we reach the end of the entire dataset. These predictions correspond to the test set's output values.

The results of the prediction are very similar to the actual values of the target variable obtained from the test set and this wraps up the implementation of time series forecasting in Python.

---

## Conclusion

In this blog, we explored a new algorithm called ARIMA and 2 of its variants: SARIMA and SARIMAX. We successfully performed time series forecasting on a beer production [dataset](https://www.kaggle.com/shenba/time-series-datasets?select=monthly-beer-production-in-austr.csv) from Kaggle in Python. We used various important plots to visualize what a time series is and how different components are embedded into it. We then fit the SARIMAX model to our dataset and also plotted the results of the model after fitting it. We made predictions on testing data and compared the predicted values to the actual values of the target variable by plotting each of the values with respect to time on the X-axis.
