'''
Random Forest is a machine-learning method that uses many decision trees instead of just one.
Each tree gives its own answer, and the forest picks the answer that most trees agree on. 
It adds randomness in two ways to make the trees different from each other:

1) Random rows of data are used to build each tree.
2) Random features are used when splitting nodes.
This randomness helps the forest make better predictions and avoid overfitting.

Working:

1) Take your dataset (many rows + many features).
2) Create many Decision Trees, but each tree is trained on:
    Random rows (bootstrap sampling)
    Random features (feature selection at each split)
3) Each tree makes a prediction
    For classification → predicts a class
    For regression → predicts a number
4) Combine all tree predictions
    Classification → the class with the most votes wins
    Regression → the average of all tree outputs
5) Final output = the "wisdom" of all trees, not just one.

'''

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

data = pd.read_csv("Titanic-Dataset.csv")

data = data.dropna(subset = ['Survived'])

# Define X and Y:
x = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = data['Survived']

x['Sex'] = x['Sex'].map({'female': 0, 'male':1})
x['Age'] = x['Age'].fillna(x['Age'].median())


# split the data:
x_train, x_test, y_train, y_test =  train_test_split(x,
                                                     y,
                                                     test_size = 0.2,
                                                     random_state = 42)

# initialize the random forest classifier:
rf_classifier =RandomForestClassifier(n_estimators=100,
                                     random_state=42)

# fit the model:
rf_classifier.fit(x_train, y_train)

# make predictions:
y_pred = rf_classifier.predict(x_test)

accuracy  = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", report)

'''
Classification Report:
1) Recall:
Of all the actual positive items, how many did the model successfully find?
👉How well the model catches positives.

2) F1-Score:
The balance between precision and recall.
👉Good when you need both accuracy and completeness.

3) Support:
How many actual samples are in each class.

4) Accuracy:
Total correct predictions / total samples.
👉Overall correctness.
'''