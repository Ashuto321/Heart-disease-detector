import pandas as pd
df = pd.read_csv('C:\\Users\\Ashutosh Pandey\\Desktop\\heart.csv')
# print(df)

from sklearn.model_selection import train_test_split

x,y=df.drop('target', axis=1), df['target']

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.4, random_state=9)#test size is 0.4 or 40% of data

### scale-Insensitive models

#model 1
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier()
forest.fit(x_train, y_train)
#model 2
from sklearn.naive_bayes import GaussianNB

nb_clf = GaussianNB()
nb_clf.fit(x_train, y_train)

#model 3
from sklearn.ensemble import GradientBoostingClassifier

gb_clf = GradientBoostingClassifier()
gb_clf.fit(x_train, y_train)

###scale-sensetive models

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

#model-2
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train_scaled, y_train)
#model-3
from sklearn.linear_model import LogisticRegression
log = KNeighborsClassifier()
log.fit(x_train_scaled, y_train)
#model-4
from sklearn.svm import SVC
SVC = KNeighborsClassifier()
SVC.fit(x_train_scaled, y_train)

#checkig and testing 
c= forest.score(x_test, y_test)
d=nb_clf.score(x_test, y_test)
e=gb_clf.score(x_test, y_test)
f=knn.score(x_test_scaled, y_test)
g=log.score(x_test_scaled, y_test)
h=SVC.score(x_test_scaled, y_test)
# print(c,d,e,f,g,h)

#evaluation----
from sklearn.metrics import recall_score

y_preds = forest.predict(x_test)
# print('forest:', recall_score(y_test, y_preds))

y_preds = nb_clf.predict(x_test)
# print('NB:', recall_score(y_test, y_preds))

y_preds = gb_clf.predict(x_test)
# print('GB:', recall_score(y_test, y_preds))

y_preds = knn.predict(x_test_scaled)
# print('knn:', recall_score(y_test, y_preds))

y_preds = log.predict(x_test_scaled)
# print('log:', recall_score(y_test, y_preds))

y_preds = SVC.predict(x_test_scaled)
# print('svc:', recall_score(y_test, y_preds))

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

#getting probablities
y_probs= forest.predict_proba(x_test)[:, 1]
#getting rates like false positive rates, true positive rates and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

plt.plot(fpr, tpr)
plt.xlabel('False Positive Rates')
plt.ylabel('True Positive Rates(Recall)')
plt.title('ROC Curve')
# plt.show()

##----------if we try with log

#getting probablities
# y_probs= log.predict_proba(x_test)[:, 1]
# #getting rates like false positive rates, true positive rates and thresholds
# fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# plt.plot(fpr, tpr)
# plt.xlabel('False Positive Rates')
# plt.ylabel('True Positive Rates(Recall)')
# plt.title('ROC Curve')
# plt.show()

##----------if we try with knn
#getting probablities
# y_probs= knn.predict_proba(x_test)[:, 1]
# #getting rates like false positive rates, true positive rates and thresholds
# fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# plt.plot(fpr, tpr)
# plt.xlabel('False Positive Rates')
# plt.ylabel('True Positive Rates(Recall)')
# plt.title('ROC Curve')
# plt.show()


##----------if we try with gb
# #getting probablities
# y_probs= gb_clf.predict_proba(x_test)[:, 1]
# #getting rates like false positive rates, true positive rates and thresholds
# fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# plt.plot(fpr, tpr)
# plt.xlabel('False Positive Rates')
# plt.ylabel('True Positive Rates(Recall)')
# plt.title('ROC Curve')
# plt.show()

# a= roc_auc_score(y_test, y_probs)
# print(a)


##-Hyper parameter tuning
from sklearn.model_selection import GridSearchCV

param_grid={
'n_estimators': [10, 20, 50],
'max_depth': [None, 10, 20, 30],
'min_samples_split':[2,5,10],
'min_samples_leaf':[1,2,4],
'max_features': ['sqrt', 'log2', None]

}

forest = RandomForestClassifier(n_jobs=-1, random_state=1)
grid_search= GridSearchCV(forest, param_grid, cv=2, n_jobs=-1, verbose=2)#three fold cross validation

grid_search.fit(x_train, y_train)


best_forest = grid_search.best_estimator_
best_forest


#feature importances:
import numpy as np

feature_importances = best_forest.feature_importances_
features= best_forest.feature_names_in_

sorted_idx = np.argsort(feature_importances)
sorted_features = features[sorted_idx]
sorted_importances = feature_importances[sorted_idx]

colors = plt.cm.YlGn(sorted_importances / max(sorted_importances))

plt.barh(sorted_features, sorted_importances, color=colors)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importances')
# plt.show()


import seaborn as sns

plt.figure(figsize=(12,10))
z=sns.heatmap(df.corr(), annot= True, cmap='YlGn')
# plt.show()





