import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# load the data set from week3
df = pd.read_csv("../week3_preprocessing_features/final_dataset.csv")

# Seperate features and target

X = df.drop('Churn', axis=1)

y = df['Churn']

#split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

#Logistic Regression

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

acc_lr = accuracy_score(y_test, y_pred_lr)

# Random Forest

rf = RandomForestClassifier()

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

acc_rf = accuracy_score(y_test, y_pred_rf)

# Support Vector Machine

svm =  SVC()

svm.fit(X_train, y_train)

y_pred_svm = svm.predict(X_test)

acc_svm = accuracy_score(y_test, y_pred_svm)

print(f"Lofistic Regression Accuracy: {acc_lr:.2f}")
print(f"Random Forest Accuracy : {acc_rf:.2f}")
print(f"SVM Accuracy: {acc_svm:.2f}")