import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
import joblib

# load dataset from week3

df = pd.read_csv("../week3_preprocessing_features/final_dataset.csv")
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train,y_test= train_test_split(X,y,test_size=0.2, random_state=42)

# best model from tuning

best_model = LogisticRegression(C=100, penalty="l1", solver='liblinear', max_iter=1000)
best_model.fit(X_train,y_train)

#save model
joblib.dump(best_model,'Churn_model.pkl')
print("model saved as Churn_model.pkl")