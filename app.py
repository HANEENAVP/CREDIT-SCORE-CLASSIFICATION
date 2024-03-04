from flask import Flask, render_template, request
import pickle
import numpy as np
# Load the trained model from the file
model=pickle.load(open('model.pkl','rb'))
app = Flask(__name__)

# Route for home page
@app.route('/')
def home():
    return render_template('home.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    
  
    
    # Get input data
    age = float(request.form['age'])
    annual_income = float(request.form['annual_income'])
    monthly_inhand_salary = float(request.form['monthly_inhand_salary'])
    num_bank_accounts = float(request.form['num_bank_accounts'])
    num_credit_card = float(request.form['num_credit_card'])
    num_of_loan = float(request.form['num_of_loan'])
    delay_from_due_date =float(request.form['delay_from_due_date'])
    num_of_delayed_payment = float(request.form['num_of_delayed_payment'])
    num_credit_inquiries = float(request.form['num_credit_inquiries'])
    outstanding_debt = float(request.form['outstanding_debt'])
    total_emi_per_month = float(request.form['total_emi_per_month'])
    amount_invested_monthly = float(request.form['amount_invested_monthly'])
    monthly_balance = float(request.form['monthly_balance'])
    occupation= request.form['occupation']
    credit_mix = request.form['credit_mix']
    payment_of_min_amount= request.form['payment_of_min_amount']
    payment_behaviour= request.form['payment_behaviour']
    print(occupation)
    # Encode categorical variables
    from sklearn.preprocessing import LabelEncoder
    le=LabelEncoder()

    

    Occupation_encoded=le.fit_transform([occupation])
    Credit_Mix_encoded=le.fit_transform([credit_mix])
    Payment_of_Min_Amount_encoded=le.fit_transform([payment_of_min_amount])
    Payment_Behaviour_encoded=le.fit_transform([payment_behaviour])
     # Make a prediction
    input_features = [age,Payment_of_Min_Amount_encoded,Credit_Mix_encoded,Occupation_encoded,Payment_Behaviour_encoded,annual_income,monthly_inhand_salary,num_bank_accounts,num_credit_card,num_of_loan,num_of_delayed_payment,delay_from_due_date,outstanding_debt,num_credit_inquiries,total_emi_per_month,amount_invested_monthly,monthly_balance]
    
    #creating numpy array
    clean_data = [float(i) for i in input_features]
   # Reshape the Data as a Sample not Individual Features
    ex1 = np.array(clean_data).reshape(1,-1)
    output=model.predict(ex1)
   
    if output[0] == 1:#poor
     prediction_string = f"You are not eligible for a loan and your credit score is poor"
    elif output[0] == 0:#standard
     prediction_string = f"You are eligible for a loan and your credit score is standard"
    elif output[0] == 2:#good
     prediction_string = f"You are eligible for a loan and your credit score is good"

     
    return render_template('res.html',prediction_text=prediction_string)

    

if __name__ == '__main__':
    app.run(debug=True)
