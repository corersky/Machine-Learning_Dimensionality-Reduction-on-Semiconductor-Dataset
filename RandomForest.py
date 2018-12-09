# Feature importances using Random Forest Classifier
# importing libraries
import numpy as np
import pandas as pd

# importing dataset
# create matrix of independent variables(features)
data_X = pd.read_csv('secom.data.txt', sep = ' ')
X = data_X.values

#create dependent variable vector
data_y = pd.read_csv('secom_labels.data.txt', sep = ' ')
y = data_y.iloc[:, 0].values

# handling missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X)
X = imputer.transform(X)

# Applying Random Forest classifier
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
forest.fit(X, y)
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

#print feature importances in descending order
#for f in range(X_train.shape[1]):
for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot bar graph of feature importances
import matplotlib.pyplot as plt
plt.figure()
plt.title("Feature importances using Random Forest Classifier")
plt.xlabel('Number of features')
plt.ylabel('Importances')
plt.bar(range(X.shape[1]), importances[indices],
       color="b", align="center")

plt.show()
 
