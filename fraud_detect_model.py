import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, recall_score
sns.set()

df = pd.read_csv('creditcard.csv')

class_names = {0:'Not Fraud', 1:'Fraud'}
'''
print(df.Class.value_counts().rename(index = class_names))
Not Fraud    284315
Fraud           492
Name: Class, dtype: int64
'''
feature_names = df.iloc[:, 1:30].columns
target = df.iloc[:1, 30:].columns
'''
print(feature_names)
print(target)
Index(['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11',
       'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21',
       'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'],
      dtype='object')
Index(['Class'], dtype='object')
'''
data_features = df[feature_names]
data_target = df[target]

X_train, X_test, y_train, y_test = train_test_split(data_features, data_target, train_size=0.7, test_size=0.3, random_state=1)
'''
print("Length of X_train is: {X_train}".format(X_train = len(X_train)))
print("Length of X_test is: {X_test}".format(X_test = len(X_test)))
print("Length of y_train is: {y_train}".format(y_train = len(y_train)))
print("Length of y_test is: {y_test}".format(y_test = len(y_test)))
Length of X_train is: 199364
Length of X_test is: 85443
Length of y_train is: 199364
Length of y_test is: 85443
'''

model = LogisticRegression()
# ravel is a numpy function that flattens out nested arrays into a 1D array
# https://www.geeksforgeeks.org/numpy-ravel-python/
model.fit(X_train, y_train.values.ravel())
pred = model.predict(X_test)

# prepare confusion matrix
matrix = confusion_matrix(y_test, pred)
'''
print(matrix)
[[85292    16]
[   61    74]]
class_names = ['not fraud', 'fraud']
dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)
sns.heatmap(dataframe, annot=True, cbar=None, cmap='Blues', fmt = 'g')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.ylabel('True Class'), plt.xlabel('Predicted Class')
plt.show()
'''

# prepare Sensitivity and F1 Score metrics
f1_score = round(f1_score(y_test, pred), 2)
recall_score = round(recall_score(y_test, pred), 2)
print("Sensitivity/Recall for Logistic Regression Model 1 : {recall_score}".format(recall_score = recall_score))
print("F1 Score for Logistic Regression Model 1 : {f1_score}".format(f1_score = f1_score))
'''
Performance metrics
Accuracy - ratio of correctly labeled subjects to the whole pool of subjects
 -- Here, there were 85292+74 correctly identified cases, out of total 85443; giving an accuracy of 99.9% Sounds good right? The majority of these came from identifying "not fraud". Not exactly the goal of this model
Precision - the ratio of True Positive to all Positives
 -- Here, there were 74 cases of fraud correctly identified out of a total 74+16 cases identified as fraud; giving a precision rate of 82.2%. Fairly good but this does not take into account the cases where we did not successfully identify a fraud.
Recall (Sensitivity) - the ratio of True Positives to all Positives
 -- Here, there were 74 cases of True Positive out of an actual total of 74+61 cases of positive. Giving a Sensitivty ratio of 54.8%. Yikes! We missed 45.2% of fraud.
F1-Score - considers both precision and recall; formula... 2*(Recall*Precision)/(Recall+Precision)
 -- Here... 2*(.548*.822)/.548+.822 = .6576
Specificity - the ratio of True Negatives to all Negatives
 -- Here, there were 85292 correctly identified Negatives, out of a total 85292+16 Negative cases; giving a specificity rate of 99.9%. Again, looks good, but this measure does not consider False Negatives, which are the most important measurement.
'''
