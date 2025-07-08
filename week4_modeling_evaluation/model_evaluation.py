import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df = pd.read_csv("../week3_preprocessing_features/final_dataset.csv")

# Load data from week3
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

#Models

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n {name} Evaluation:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test,y_pred))

    disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=["No Churn", "Churn"])
    disp.plot(cmap="Blues")
    plt.title(f"{name} – Confusion Matrix")
    plt.savefig(f"{name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.clf()