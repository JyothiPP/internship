import pandas as pd
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('xt_model.pickle', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    Month= int(request.form['Month'])
    Age= int(request.form['Age'])
    Occupation= int(request.form['Occupation'])
    Annual_Income= float(request.form['Annual_Income'])
    Num_Bank_Accounts= int(request.form['Num_Bank_Accounts'])
    Num_Credit_Card= int(request.form['Num_Credit_Card'])
    Interest_Rate= float(request.form['Interest_Rate'])
    Num_of_Loan= int(request.form['Num_of_Loan'])
    Delay_from_due_date= int(request.form['Delay_from_due_date'])
    Num_of_Delayed_Payment= int(request.form['Num_of_Delayed_Payment'])
    Changed_Credit_Limit= float(request.form['Changed_Credit_Limit'])
    Num_Credit_Inquiries= int(request.form['Num_Credit_Inquiries'])
    Credit_Mix= int(request.form['Credit_Mix'])
    Outstanding_Debt= float(request.form['Outstanding_Debt'])
    Credit_Utilization_Ratio= float(request.form['Credit_Utilization_Ratio'])
    Credit_History_Age= int(request.form['Credit_History_Age'])
    Payment_of_Min_Amount= int(request.form['Payment_of_Min_Amount'])
    Total_EMI_per_month= float(request.form['Total_EMI_per_month'])
    Monthly_Balance= float(request.form['Monthly_Balance'])
    Auto_Loan= int(request.form['Auto_Loan'])
    Credit_Builder_Loan= int(request.form['Credit_Builder_Loan'])
    Personal_Loan= int(request.form['Personal_Loan'])
    Home_Enquity_Loan= int(request.form['Home_Enquity_Loan'])
    Mortgage_Loan= int(request.form['Mortgage_Loan'])
    Student_Loan= int(request.form['Student_Loan'])
    Debt_Consolidation_Loan= int(request.form['Debt_Consolidation_Loan'])
    Payday_Loan= int(request.form['Payday_Loan'])
    Spending_behaviour= int(request.form['Spending_behaviour'])
    Payment_behaviour= int(request.form['Payment_behaviour'])

    
    # Create a DataFrame with the user's input
    input_data = pd.DataFrame({

        'Month':[Month],
        'Age':[Age],
        'Occupation':[Occupation],
        'Annual_Income':[Annual_Income],
        'Num_Bank_Accounts':[Num_Bank_Accounts],
        'Num_Credit_Card':[Num_Credit_Card],
        'Interest_Rate':[Interest_Rate],
        'Num_of_Loan':[Num_of_Loan],
        'Delay_from_due_date':[Delay_from_due_date],
        'Num_of_Delayed_Payment':[Num_of_Delayed_Payment],
        'Changed_Credit_Limit':[Changed_Credit_Limit],
        'Num_Credit_Inquiries':[Num_Credit_Inquiries],
        'Credit_Mix':[Credit_Mix],
        'Outstanding_Debt':[Outstanding_Debt],
        'Credit_Utilization_Ratio':[Credit_Utilization_Ratio],
        'Credit_History_Age':[Credit_History_Age],
        'Payment_of_Min_Amount':[Payment_of_Min_Amount],
        'Total_EMI_per_month':[Total_EMI_per_month], 
        'Monthly_Balance':[Monthly_Balance], 
        'Auto_Loan':[Auto_Loan], 
        'Credit_Builder_Loan':[Credit_Builder_Loan], 
        'Personal_Loan':[Personal_Loan], 
        'Home_Enquity_Loan':[Home_Enquity_Loan], 
        'Mortgage_Loan':[Mortgage_Loan],
        'Student_Loan':[Student_Loan],
        'Debt_Consolidation_Loan':[Debt_Consolidation_Loan],
        'Payday_Loan':[Payday_Loan],
        'Spending_behaviour':[Spending_behaviour],
        'Payment_behaviour':[Payment_behaviour]
    })
        
    # Make the prediction
    prediction = model.predict(input_data)[0]

    return render_template('result.html',prediction=prediction)

    
if __name__ == '__main__':
    app.run(debug=True)