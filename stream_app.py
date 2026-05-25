import pickle
import streamlit as st
import numpy as np
from sklearn.preprocessing import MinMaxScaler

st.title("Bank Customer Churn Model")

# Loading the trained model
with open('model.pkl', 'rb') as file:
    classifier = pickle.load(file)

# Try to load the scaler, if not available create a note
try:
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
except FileNotFoundError:
    st.warning("Scaler not found. Please run the notebook to save scaler.pkl")
    scaler = None

@st.cache_data
def prediction(credit_score, gender, age, tenure, balance, products_number, credit_card, active_member, estimated_salary, country):   
    """Make prediction with proper preprocessing"""
    
    # Convert string inputs to numeric
    tenure = int(tenure)
    products_number = int(products_number)
    
    # Encode categorical variables
    if gender == "Male":
        gender = 0
    else:
        gender = 1
 
    if credit_card == "No":
        credit_card = 0
    else:
        credit_card = 1
 
    if active_member == "No":
        active_member = 0
    else:
        active_member = 1  

    if country == "Spain":
        country = 0
    elif country == "Germany":
        country = 1
    elif country == "France":
        country = 2
    
    # Create feature array
    features = np.array([[credit_score, gender, age, tenure, balance, products_number, 
                         credit_card, active_member, estimated_salary, country]])
    
    # Scale features if scaler is available
    if scaler is not None:
        features = scaler.transform(features)
    
    # Make prediction
    prediction_result = classifier.predict(features)[0]
     
    if prediction_result == 1:
        pred = 'Churn (Leave!)'
    else:
        pred = 'Retain (Stay!)'
    return pred

def main():  
    """Main Streamlit app interface"""
      
    st.subheader("Enter Customer Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        CreditScore = st.number_input("Customer's Credit Score", min_value=0, max_value=1000, value=650)
        Gender = st.selectbox("Customer's Gender", ("Male", "Female"))
        Age = st.number_input("Customer's Age", min_value=18, max_value=100, value=35)
        Tenure = st.selectbox("Years as Client", ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"))
        Balance = st.number_input("Customer's Account Balance", min_value=0.0, value=50000.0)
    
    with col2:
        NumOfProducts = st.selectbox("Number of Bank Products Used", ("1", "2", "3", "4"))
        HasCrCard = st.selectbox("Has Credit Card?", ("Yes", "No"))
        IsActiveMember = st.selectbox("Is Active Member?", ("Yes", "No"))
        EstimatedSalary = st.number_input("Estimated Salary", min_value=0.0, value=100000.0)
        country = st.selectbox("Country", ("Spain", "Germany", "France"))
    
    # Make prediction when button is clicked
    if st.button("Predict Churn", key="predict_btn"):
        if scaler is None:
            st.error("Cannot make prediction: Scaler not found. Run the notebook first to save scaler.pkl")
        else:
            result = prediction(CreditScore, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, country)
            st.success(f'Prediction: Customer will **{result}**')
     
if __name__ == '__main__': 
    main()
