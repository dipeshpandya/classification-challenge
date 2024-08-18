# Project to demonstrate and compare performance of different machine learning models.

## Detecting spam emails using supervised learning models 

The project demonstrates methods used for supervised learning and compares accuracy score of two methods used in the code. We will go through following steps to evaluate the performance

1. [Import required libraries](#import-required-libraries)
2. [Define end goal](#define-end-goal)
3. [Split the Data into Training and Testing Sets](#split-the-data-into-training-and-testing-sets)
4. [Scaling the data](#scaling-the-data)
5. [Create and Fit and predict a Logistic Regression Model](#create-and-fit-and-predict-a-logistic-regression-model)
6. [Create and Fit and predict Random Forest Classifier Model](#create-and-fit-and-predict-random-forest-classifier-model)
7. [Evaluating and analyzing model differences](#evaluating-and-analyzing-model-differences)

### Import required libraries
```python 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```
The code imports an existing scaled dataset to use for this project

```python
# Import the data
data = pd.read_csv("https://static.bc-edx.com/ai/ail-v-1-0/m13/challenge/spam-data.csv") # pull the data in a dataframe
data.head() # to view top 5 records of 
```

### Define end goal
We will be creating and comparing two models on this data: a Logistic Regression, and a Random Forests Classifier. In order to train and test the models, we will create two variable, viz. X and y. Variable X are the features (aka dimensions) while variable y is the columns representing the results we expect or wish to predict.

### Split the Data into Training and Testing Sets
```python
# Create the labels set `y` and features DataFrame `X`
y = data['spam'] # identifying the result column
# creating a copy of dataframe
X = data.copy()
# Dropping the column we identified as y to get columns (features) to use as X
X.drop('spam', axis=1, inplace=True)
```
Now, we will want to check how balanced is our data. To do that, we will want to see the distribution of y column
```python
# Check the balance of the labels variable (`y`) by using the `value_counts` function.
y.value_counts()
```
This will give us the values as below which tells us the data is marginally skewed which is not outside the expectation.

spam
0    2788
1    1813
Name: count, dtype: int64

Now will split the data in order to train the model and test the accuracy of the model
```python
# Split the data into X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1) 
# one can use a random state value of their choice. The random state number will have a marginal impact on the results
```
### Scaling the data

Next we will want to scale the data using StandardScaler so that scaled data is used for learning (training) for the models. We will initiate the model by setting a variable

```python
from sklearn.preprocessing import StandardScaler

# Create the StandardScaler instance
scaler = StandardScaler()
```
Remember, we split the data into training and testing. Now, we will fit and transform training data, We will scale and transform X_train and X_test data
```python
# Fit the Standard Scaler with the training data
scaler.fit(X_train)

# Scale the training data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Create and Fit and predict a Logistic Regression Model

We will create a Logistic Regression model and fit it to the training data. Then will make predictions with the testing data, and print the model's accuracy score. 

```python
# Train a Logistic Regression model and print the model score
from sklearn.linear_model import LogisticRegression
lgrm = LogisticRegression(random_state=1)
lgrm.fit(X_train_scaled, y_train)
print(f"Logistic Regression Model Training Data Score: {lgrm.score(X_train_scaled, y_train):.3f}") 
print(f"Logistic Regression Model Testing Data Score: {lgrm.score(X_test_scaled, y_test):.3f}")
# .3f is used to limit values to 3 decimal points.
```
The logistics Regression training score is as below
Logistic Regression Model Training Data Score: 0.930
Logistic Regression Model Testing Data Score: 0.928

Make and save testing predictions with the saved logistic regression model using the test data
```python
test_predictions = lgrm.predict(X_test_scaled)
# Calculate the accuracy score by evaluating `y_test` vs. `testing_predictions`.
logistic_regression_accuracy_score = accuracy_score(y_test, test_predictions)
round(logistic_regression_accuracy_score,3)
```
The Logistics Regression model accuracy score is 0.928

-----------

### Create and Fit and predict Random Forest Classifier Model
Next we will create a Random Forest Classifier Model, fit it to the training data, make predictions with the testing data, and print the model's accuracy score

```python
# Train a Random Forest Classifier model and print the model score
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=1)
rfc.fit(X_train_scaled, y_train)

# Evaluate the model
print(f'Random Forest Training Score: {rfc.score(X_train_scaled, y_train):.3f}')
print(f'Random Forest Testing Score: {rfc.score(X_test_scaled, y_test):.3f}')
```
Random Forest Training Score: 1.000
Random Forest Testing Score: 0.967
```python
# Make and save testing predictions with the saved logistic regression model using the test data
rfc_predictions = rfc.predict(X_test_scaled)

# Calculate the accuracy score by evaluating `y_test` vs. `testing_predictions`.

random_forest_model_accuracy_score= accuracy_score(y_test, rfc_predictions)
round(random_forest_model_accuracy_score,3)
```
The Random Forest model accuracy score 0.967

### Evaluating and analyzing model differences

As expected, the Random Forest Classifier model performed better compared to Logistics Regression Model. Random Forest Classifier model performed better by nearly 4%. Having said that, you can notice that the train and testing scores for Logistics regression are more aligned as compared with Random Forest model.

Model accuracy score comparison

Accuracy score for Random Forest Classifier is : 0.967
Accuracy score for Logistics Regression Model is : 0.928


