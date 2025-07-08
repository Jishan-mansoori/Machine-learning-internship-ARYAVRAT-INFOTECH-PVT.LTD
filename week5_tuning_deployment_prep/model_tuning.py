import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load dataset 

df = pd.read_csv("../week3_preprocessing_features/final_dataset.csv")
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# Grid search

param_grid = {
    'C': [0.01,0.1,1,10,100],
    'solver':['liblinear', 'lbfgs'],
    'penalty': ['l1','l2']
}

grid = GridSearchCV(LogisticRegression(max_iter=1000),param_grid,cv=5,scoring='f1')
grid.fit(X_train, y_train)
# Best model

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)


print(f" Best Params: {grid.best_params_}")
print("\n Classification Report (Tuned Model):")
print(classification_report(y_test, y_pred))