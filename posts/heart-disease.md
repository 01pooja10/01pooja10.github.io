We all know that machine learning is currently used widely in various sectors like healthcare, automotive, financial services, retail, education, transportation, etc.

When talking about the healthcare sector, machine learning has disrupted the domain in terms of the vast number of applications it can be used for. It is used in disease identification/diagnosis, robotic surgery, predicting epidemic outbreaks or even in the latest clinical trials/research.

> One such important application is predicting the presence of heart disease in human beings, given certain features or parameters. The model we are going to be dealing with in this blog will be a "classification model" that makes predictions using binary values (0 or 1).

The dataset used for building our classification model has been obtained from Kaggle and can be found [here](https://www.kaggle.com/ronitf/heart-disease-uci). Let's get started with the process by exploring our data. Since many of us here including myself aren't from a medical background, decoding the meaning of various features in the dataset could help us understand it better.

---

## Understanding the dataset

The heart disease dataset contains the following features:

- **Age** — Age of the person
- **Sex** — Sex or the gender of the person
- **cp** — Type of chest pain, represented by 4 values (0, 1, 2 and 3)
- **trestbps** — Resting blood pressure
- **chol** — Serum cholesterol is the combined measurement of HDL and LDL (high and low density lipoproteins). HDL is often referred to as the good cholesterol and indicates lower risk of heart disease whereas LDL is considered to be bad cholesterol and indicates a higher risk of heart disease or increased plaque formation in your blood vessels and arteries.
- **fbs** — Fasting blood sugar indicates the level of diabetes and is considered to be a risk factor if found to be above 120 mg/dl.
- **restecg** — Resting electrocardiographic results measure the electrical activity of the heart. This factor can diagnose irregular heart rhythms, abnormally slow heart rhythms, evidences of evolving/acute heart attack possibilities etc.
- **thalach** — Maximum heart rate achieved is the average maximum number of times our heart beats per minute. It is calculated as: 220 - age of the person.
- **exang** — Exercise induced angina (AP) is a common concern among cardiac patients. Angina is usually stable but is triggered when we do physical activity especially in cold conditions.
- **oldpeak** — ST depression induced by exercise relative to rest. Not all ST depressions represent an emergency condition.
- **slope** — The slope of the peak exercise ST segment.
- **ca** — The number of major vessels.
- **thal** — Thalach: 3 = normal; 6 = fixed defect; 7 = reversible defect.
- **Target variable** — Tells us whether the person has heart disease (1) or not (0).

---

## Data visualization

Visualizations help us familiarize ourselves with the data that we are dealing with. Let's look at three important visualization techniques that tell us a lot more about the data than any tabular column ever could.

### Heatmap

The heatmap is plotted using Python's seaborn library. It tells us the relation between various variables in our dataset by indicating how they affect each other. It uses a color scheme as well as decimal values. Negative values indicate relatively less correlation between the 2 specific variables whereas values closer to 1 indicate highly correlated variables. Here, `df.corr()` is used to find the pair-wise correlation of all columns in the dataframe. Read more about seaborn's [heatmap](https://seaborn.pydata.org/generated/seaborn.heatmap.html).

### Countplots

The countplots represent the count of the categorical values with respect to the target variable (0 or 1). These plots show the category (x-axis) vs. the count of the categorical values (y-axis). The presence or absence of heart disease i.e. the target variable is differentiated using colors. Also, the most common category for a given feature can be inferred from these plots. Read more about [countplot](https://seaborn.pydata.org/generated/seaborn.countplot.html).

### Distplots

The distplots are basically histograms that represent the range of values that the continuous numerical variables possess. These plots are useful for visualization when we are dealing with vastly different ranges of data. The plots represent a univariate distribution. The x-axis represents the feature which is separated into different bins. The bins consist of the frequency of occurrence of each variable in that range. The most common value for each variable can be inferred by looking at the bin with the highest y value. Read more about distplots [here](https://seaborn.pydata.org/generated/seaborn.distplot.html).

---

## Data manipulation

Now we'll be performing some data manipulation. The values of some categorical variables cause ambiguity while fitting our model and training it. So, we transform them into binary form i.e. convert categories consisting of alphabets, integers from 0-3, 0-4, etc. to 1s and 0s by adding separate columns for each category. We do this using `pd.get_dummies` from the pandas library. The documentation can be found [here](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html).

We can achieve this by specifying which columns need to be encoded and thus we get a dataframe with the original columns replaced by new columns consisting of each of our encoded variables and their binary values.

### Feature creation

Now, we can go ahead and create an extra feature. This process is called **feature engineering** and is used to either modify an existing feature to get a new column or create an entirely new column.

An extremely simple version of feature engineering is performed using the pre-existing age column. A well-known fact is that adults over the age of 60 are more likely to suffer from heart disease as compared to younger adults. So, we create a separate column to filter the entries in which the person is either 60 years or older. We assign 0s to those below 60 years of age and 1s to people over the age of 60. We name the column "seniors" which refers to senior citizens.

### Train-test split

Our data now needs to be split into `xtrain`, `xtest`, `ytrain` and `ytest` for ensuring that our model can fit itself to the training set and predict values using the test set.

The split is performed in the ratio of **80:20** for train:test using sklearn's [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) which takes 80% of our training data and splits it into input features and labels.

### Data scaling

The next logical step would be data scaling. We need to necessarily scale our data before proceeding with the model creation since our dataset contains different features in various ranges. If the data isn't scaled, it will be difficult for the model to assign equal importance to all the features as it will be biased towards features with higher values.

So, the data is scaled down using sklearn's [StandardScaler()](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html). This functionality transforms our data such that it is scaled down to unit variance. The mean of the distribution will equal 0 and the deviation will equal 1. This makes it easier for the model to assign equal priority to all the features in spite of their range of values.

---

## Model creation

We can finally go ahead with model creation. Classification models basically take in the input values and make predictions by classifying various inputs into their respective labels/categories.

The model that we will be creating will use **Logistic Regression** which is a supervised learning classification algorithm. A single-class binary logistic regression model gives outputs in the form of 0 or 1 and it employs the **sigmoid function** which is, graphically, an S-shaped curve. This function is the crux of the computations performed.

The sigmoid function binds various values into a range between 0 and 1 and maps the predictions to probabilities. A threshold value (for example, 0.5) is set and any probability equal to or above the threshold is assigned to the presence (1) class. Any value below the threshold is assigned to the absence (0) class.

Refer to the documentation for [logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) to learn more about the parameters used.

> This model churns out an accuracy of roughly **90.16%** which is very decent.

---

## Evaluation

The accuracy values from 5 different models, including logistic regression, were compared. For better understanding of what exactly our model has predicted and why our accuracy is what it is, we can view 2 important metrics provided by sklearn — **confusion_matrix** and **classification_report**.

### Confusion matrix

The [confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) is basically a representation of our true negatives, true positives, false negatives and false positives. This matrix tells us how many correct and incorrect predictions our model made.

From this we can infer that our model predicted **27+28 values** (true positives and negatives) correctly, whereas it predicted **2+4 values** (false positives and negatives) incorrectly.

### Classification report

The [classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) uses 3 different aspects: precision, recall and f1-score.

- **Precision** is defined as the number of correct positives predicted with respect to the total number of positive predictions: `TP / (TP+FP)`. In our model, precision for presence of heart disease = 27 / (27+2) = **0.93**

- **Recall** is defined as the number of actual positives predicted with respect to the total number of positive predictions: `TP / (TP+FN)`. In our model, recall for presence of heart disease = 27 / (27+4) = **0.87**

- **F1 score** is the solution to balancing precision and recall: `(2 x precision x recall) / (precision + recall)`. Applying the formula: (2 x 0.93 x 0.87) / (0.93 + 0.87) = **0.90**

---

## Conclusion

Thus our model has been created and evaluated successfully. Our model can make predictions if we provide values for all the input features. The aforementioned steps constitute the process involved in building a classification model for heart disease prediction to get a very decent accuracy. Though there are other ways to improve model accuracy which you are free to try, this blog specifically deals with the structure and process involved in building/understanding an end-to-end machine learning project.

Feel free to refer to my [kernel on Kaggle](https://www.kaggle.com/poojaravi01/predicting-the-presence-of-heart-disease#Classification) for the entire code.
