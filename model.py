import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from dateutil import parser
import pickle

data=pd.read_csv(r"credit.new.csv")
print("loaded data")

#Removing unwanted columns
data=data.drop(['ID','Customer_ID','SSN','Name'],axis=1)

#Feature engineering
loan_type_data = list(data['Type_of_Loan'])

#Create a dictionary to store the counts of all the various loan types

loan_type_dict = dict()
for value in loan_type_data:
    values = value.split(',')
    for each_value in values:
        loan_type = each_value.strip(' ')
        if 'and' in loan_type:
            loan_type = loan_type[4 : ]
        if loan_type in loan_type_dict:
            loan_type_dict[loan_type] += 1
        else:
            loan_type_dict[loan_type] = 1

#Creating 8 different lists for each loan type
auto_loan = [0] * (len(data))
credit_builder_loan = [0] * (len(data))
personal_loan = [0] * (len(data))
home_equity_loan = [0] * (len(data))
mortgage_loan = [0] * (len(data))
student_loan = [0] * (len(data))
debt_consolidation_loan = [0] * (len(data))
payday_loan = [0] * (len(data))

#Using 0's and 1's if a customer has a particular loan
for index in range(len(loan_type_data)):
    #For Auto Loan
    if 'Auto' in loan_type_data[index]:
        auto_loan[index] = 1
    
    #For Credit Builder Loan
    if 'Credit-Builder' in loan_type_data[index]:
        credit_builder_loan[index] = 1
        
    #For Personal Loan
    if 'Personal' in loan_type_data[index]:
        personal_loan[index] = 1
    
    #For Home Equity Loan
    if 'Home' in loan_type_data[index]:
        home_equity_loan[index] = 1
    
    #For Mortgage Loan
    if 'Mortgage' in loan_type_data[index]:
        mortgage_loan[index] = 1
    
    #For Student Loan
    if 'Student' in loan_type_data[index]:
        student_loan[index] = 1
        
    #For Debt Consolidation loan
    if 'Debt' in loan_type_data[index]:
        debt_consolidation_loan[index] = 1
    
    #For Payday loan
    if 'Payday' in loan_type_data[index]:
        payday_loan[index] = 1

#Adding the new columns to the dataset
data['Auto_Loan'] = auto_loan
data['Credit_Builder_Loan'] = credit_builder_loan
data['Personal_Loan'] = personal_loan
data['Home_Enquity_Loan'] = home_equity_loan
data['Mortgage_Loan'] = mortgage_loan
data['Student_Loan'] = student_loan
data['Debt_Consolidation_Loan'] = debt_consolidation_loan
data['Payday_Loan'] = payday_loan

#Removing the column - Type_of_loan
data.drop(['Type_of_Loan'], axis = 1, inplace = True)

data[['Spending_behaviour', 'temp']] = data['Payment_Behaviour'].str.split('_', expand=True, n=1)
data[['temp','Payment_behaviour']] = data['temp'].str.split('_', expand=True, n=1)
data[['Payment_behaviour','temp']] = data['Payment_behaviour'].str.split('_', expand=True, n=1)

#Removing the column - Payment_Behaviour and temp
data.drop(['Payment_Behaviour','temp'], axis = 1, inplace = True)

#Dropping one from each pair of highly correlated features
data=data.drop(['Amount_invested_monthly', 'Monthly_Inhand_Salary'],axis=1)

#Label encoding categorical columns
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
to_encode=['Occupation','Credit_Mix','Payment_of_Min_Amount','Spending_behaviour','Payment_behaviour']
for col in to_encode:
    data[col]=le.fit_transform(data[col])

x=data.drop(['Credit_Score'],axis=1)
y=data['Credit_Score']

from collections import Counter
print('dataset shape :', Counter(y))

from imblearn.over_sampling import SMOTE

smote = SMOTE()

# fit predictor and target variable
x_smote, y_smote = smote.fit_resample(x, y)

print('Original dataset shape', Counter(y))
print('Resample dataset shape', Counter(y_smote))

#Dividing the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(x_smote, y_smote, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


from sklearn.ensemble import ExtraTreesClassifier
xt_cls=ExtraTreesClassifier()

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 5],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
}

# Perform GridSearchCV
grid_search = GridSearchCV(xt_cls, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Print best hyperparameters
print("Best hyperparameters:", grid_search.best_params_)
print("Best accuracy:", grid_search.best_score_)

best_xt= grid_search.best_estimator_
best_xt_pred = best_xt.predict(X_test)

# Evaluate the model performance
print('Accuracy score on the validation set',accuracy_score(y_test,best_xt_pred))


#Creating pickle file for extra tree model
pickle_file="xt_model.pickle"
with open(pickle_file,'wb') as file:
    pickle.dump(best_xt,file)

print(f"Best model has been pickled and saved to {pickle_file}")