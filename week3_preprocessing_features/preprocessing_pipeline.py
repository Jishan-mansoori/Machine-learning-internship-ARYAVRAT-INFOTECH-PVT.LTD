import pandas as pd 
from sklearn.preprocessing import LabelEncoder, StandardScaler

# load dataser
df = pd.read_csv("../week2_exploratory_data_analysis/data/churn1.csv")

#fix total charges
df['TotalCharges']= pd.to_numeric(df['TotalCharges'], errors = 'coerce')
df.dropna(inplace=True)

# drop customer id (not usefull)
df.drop(['customerID'], axis = 1, inplace = True)


# Encode binary columns
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df[col]= df[col].map({'Yes': 1, "No": 0})


# Encode multiple-category columns
multi_cols = ['InternetService', 'Contract', 'PaymentMethod', 'gender', 'MultipleLines',
              'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

df = pd.get_dummies(df, columns=multi_cols)

# scaler numeric features
scaler = StandardScaler()
num_cols = ['tenure','MonthlyCharges','TotalCharges']
df[num_cols] = scaler.fit_transform(df[num_cols])

# export processed dataset

df.to_csv("final_dataset.csv", index= False)

print("preprocessing done. final_dataset.csv saved.")