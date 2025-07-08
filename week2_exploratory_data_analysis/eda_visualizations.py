import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 

df = pd.read_csv("data/churn1.csv")

# clean the data null values or correct the dtype 

df["TotalCharges"]= pd.to_numeric(df['TotalCharges'], errors= 'coerce')

df.dropna(inplace= True)

# convert target 

df["Churn"] = df["Churn"].map({"Yes": 1, 'No': 0})

# set plot style 
sns.set(style="whitegrid")
# plot churn distribution 

sns.countplot(data=df, x='Churn')
plt.title("Churn Distribution")
plt.savefig("Chrun_distribution.png")
plt.clf()


# churn by gender 

sns.countplot(data=df, x='gender', hue='Churn')
plt.title("Churn by gender")
plt.savefig("Churn_by_gender.png")
plt.clf()


# Churn by contract type
sns.countplot(data=df, x="Contract", hue = 'Churn')
plt.title("Churn by contract type")
plt.xticks(rotation=20)
plt.savefig("churn_by_contract.png")
plt.clf()

# Boxplot : MonthlyCharges
sns.boxplot(data=df, x='Churn', y="MonthlyCharges")
plt.title("Monthaly chatges by churn")
plt.savefig("monthly_charges_boxplot.png")
plt.clf()

print("Eda plots saved!")