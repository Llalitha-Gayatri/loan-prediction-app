Loan Approval Prediction System
Overview

The Loan Approval Prediction System is a supervised Machine Learning web application that predicts whether a user’s loan request will be approved or not based on their financial and demographic details such as salary, loan amount, and credit score.

This project aims to automate the loan pre-screening process, allowing financial institutions to make faster and more consistent decisions while helping applicants understand their loan eligibility.

Tech Stack

Frontend: HTML, CSS (modern and responsive UI)

Backend: Flask (Python)

Machine Learning: Scikit-learn / XGBoost

Explainability Tools: SHAP / LIME (for future model interpretability)

Deployment: Flask app (optionally Dockerized)

Approach

Data Cleaning and Feature Engineering – Processed and prepared data using Pandas.

Model Training and Selection – Used Scikit-learn or XGBoost classifiers to build predictive models.

Model Evaluation – Evaluated using Accuracy and ROC-AUC metrics.

Explainability – Integrated SHAP or LIME to interpret predictions (optional).

Prediction Logic (Rule-Based Version)
Condition	Decision	Reason
Credit Score < 750	Not Approved	Low credit score
Salary = ₹20,000 and Loan ≤ ₹5,00,000	Approved	Eligible based on income
Salary = ₹20,000 and Loan > ₹5,00,000	Not Approved	Requested loan amount exceeds eligibility
Salary = ₹30,000 and Loan ≤ ₹8,00,000	Approved	Eligible based on income
Otherwise	Not Approved	Does not meet approval conditions
User Interface Preview
Loan Approved
<img width="591" height="854" alt="Screenshot 2025-10-21 202247" src="https://github.com/user-attachments/assets/4712e2df-4322-44da-808b-a56dead27e0e" />

Loan Not Approved
<img width="588" height="825" alt="Screenshot 2025-10-21 202123" src="https://github.com/user-attachments/assets/fa02a532-aab3-4952-b7c5-5c07ca671b08" />

How to Run the Project Locally
# Clone the repository
git clone https://github.com/your-username/loan-approval-prediction.git

# Navigate to the project directory
cd loan-approval-prediction

# Install dependencies
pip install -r requirements.txt

# Run the Flask application
python app.py

# Open in browser
http://127.0.0.1:5000/

Outcome

This project demonstrates how machine learning can be applied to financial decision-making.
The system provides real-time loan eligibility predictions, helping to automate pre-screening and speed up loan approvals.
It also improves transparency by showing applicants the reason for approval or rejection.
