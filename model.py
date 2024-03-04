import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data=pd.read_csv("final_credit_data.csv")

featre=data[['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
       'Num_Credit_Card', 'Num_of_Loan', 'Delay_from_due_date',
       'Num_of_Delayed_Payment', 'Num_Credit_Inquiries', 'Outstanding_Debt',
       'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance',
       'Credit_Score', 'Occupation_encoded', 'Credit_Mix_encoded',
       'Payment_of_Min_Amount_encoded', 'Payment_Behaviour_encoded']]
X=data.drop(['Credit_Score'],axis=1)
y=data['Credit_Score']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)
from imblearn.over_sampling import SMOTE
# Assuming X_train contains your feature vectors and y_train contains the corresponding labels

# Instantiate SMOTE
smote = SMOTE(random_state=42)

# Resample the dataset
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Check the distribution of classes after resampling
unique, counts = np.unique(y_train_resampled, return_counts=True)
print(dict(zip(unique, counts)))
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)

# Define the Random Forest classifier with n_estimators=[10]
model = RandomForestClassifier(n_estimators=10)

# Train the classifier with your data
model.fit(X_train, y_train)  # Assuming X_train and y_train are your training data
model_predictions=model.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, model_predictions)
print("Accuracy:", accuracy)
pickle.dump(model,open('model.pkl','wb'))#save trained model.