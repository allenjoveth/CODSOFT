## TASK 1: Credit Card Fraud Detection ##

# Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings('ignore')


# Load the data

train_data = pd.read_csv('fraudTrain.csv')
test_data = pd.read_csv('fraudTest.csv')

# Check the structure and info of the data

train_data.info()


# Check for missing values in the training dataset

print(train_data.isnull().sum())


# Preview the first few rows of training data

print(train_data.head(3))


# Checking the test data structure and null values

test_data.info()
print(test_data.isnull().sum())


# Merging training and test data

combined_data = pd.concat([train_data, test_data])
print(test_data.head(3))


# Checking the shape of the combined data

print(combined_data.shape)


# Correlation with target variable 'is_fraud'

correlations = {}
for col in combined_data.columns:
    if combined_data[col].dtype != 'object' and col != 'is_fraud':
        correlations[col] = combined_data[col].corr(combined_data['is_fraud'])

print(correlations)


# Dropping unnecessary columns for prediction

columns_to_drop = ['Unnamed: 0', 'trans_date_trans_time', 'first', 'last', 'gender', 'street', 'job', 'dob', 'trans_num']
combined_data.drop(columns=columns_to_drop, inplace=True)


# Check if the necessary columns are left

print(combined_data.info())


# Visualizing the class distribution

plt.bar(combined_data['is_fraud'].unique(), combined_data['is_fraud'].value_counts(), width=0.2)
plt.xlabel('Legitimate/Fraud')
plt.ylabel('Number of Transactions')
plt.show()


# Splitting the data into fraudulent and legitimate transactions

legitimate_data = combined_data[combined_data['is_fraud'] == 0]
fraud_data = combined_data[combined_data['is_fraud'] == 1]


# Balancing the data by sampling an equal number of legitimate transactions

legitimate_data = legitimate_data.sample(n=len(fraud_data))


# Checking the new shape of both datasets

print(legitimate_data.shape)
print(fraud_data.shape)


# Combining balanced data

balanced_data = pd.concat([legitimate_data, fraud_data])


# Visualize the class distribution again after balancing

plt.bar(balanced_data['is_fraud'].unique(), balanced_data['is_fraud'].value_counts(), width=0.2)
plt.xlabel('Legitimate/Fraud')
plt.ylabel('Number of Transactions')
plt.show()


# Encoding categorical variables

label_encoder = LabelEncoder()
balanced_data['merchant'] = label_encoder.fit_transform(balanced_data['merchant'])
balanced_data['category'] = label_encoder.fit_transform(balanced_data['category'])
balanced_data['city'] = label_encoder.fit_transform(balanced_data['city'])
balanced_data['state'] = label_encoder.fit_transform(balanced_data['state'])


# Ensure all features are now numerical

print(balanced_data.info())


# Prepare the feature set (X) and target variable (y)

X = balanced_data.iloc[:, :-1].values
y = balanced_data.iloc[:, -1].values


# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)


# Logistic Regression

logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
logistic_predictions = logistic_model.predict(X_train)

print("Logistic Regression - Accuracy on Training Data:", round(accuracy_score(y_train, logistic_predictions), 3))
print("Classification Report:\n", classification_report(y_train, logistic_predictions))


# Decision Tree Classifier

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_train)

print("Decision Tree - Accuracy on Training Data:", round(accuracy_score(y_train, dt_predictions), 3))
print("Classification Report:\n", classification_report(y_train, dt_predictions))


# Random Forest Classifier

rf_model = RandomForestClassifier(n_estimators=50)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_train)

print("Random Forest - Accuracy on Training Data:", round(accuracy_score(y_train, rf_predictions), 3))
print("Classification Report:\n", classification_report(y_train, rf_predictions))


# Evaluating on Test Data

test_predictions = logistic_model.predict(X_test)
print("Logistic Regression - Accuracy on Test Data:", round(accuracy_score(y_test, test_predictions), 3))
print("Classification Report:\n", classification_report(y_test, test_predictions))

