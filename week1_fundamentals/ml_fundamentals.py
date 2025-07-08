import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# used to load the data in df block so we can use it 
df = pd.read_csv("churn0.csv")
print(df.head())
# preprocess basic 
# in this we are changing the value of Totalcharges in numeric becouse it give as dtype object  
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

# converting the churn yes/no in 0 nad 1 

df["Churn"] = df["Churn"].map({'Yes': 1, "No": 0})

# Select Features 
x = df[['tenure', 'MonthlyCharges', 'TotalCharges']]
y = df[['Churn']]

# train-test split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train.values.ravel())

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Accuracy: {acc:.2f}")