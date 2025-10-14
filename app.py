from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Load the trained machine learning model
# Note: The 'model.pkl' file must be in the same directory as this script.
try:
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    print("Error: 'model.pkl' not found. Please run train_model.py to create it.")
    model = None

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Handles both GET and POST requests for the main page.
    GET: Displays the form.
    POST: Processes form data, applies prediction logic, and displays the result.
    """
    prediction_result = None

    if request.method == 'POST':
        # --- Get User Input ---
        try:
            name = request.form['name']
            monthly_salary = float(request.form['monthly_salary'])
            loan_amount = float(request.form['loan_amount'])
            credit_score = int(request.form['credit_score'])
        except (ValueError, KeyError) as e:
            # Handle cases where form fields are missing or have incorrect types
            return render_template('index.html', error=f"Invalid input: Please fill all fields correctly.")

        # --- Prediction Logic (as per requirements) ---
        status = ''
        reason = ''
        # --- Prediction Logic using the Machine Learning Model ---
        
        # 1. Create a feature array from the user's input
        # The model expects the data in the same order it was trained on.
        features = np.array([[monthly_salary, loan_amount, credit_score]])

        # 2. Use the loaded model to make a prediction
        prediction = model.predict(features) # This will return [0] or [1]

        # 3. Interpret the prediction
        if prediction[0] == 1:
            status = 'Approved'
            reason = 'The application meets the criteria based on our predictive model.'
        else:
            status = 'Not Approved'
            reason = 'The application does not meet the criteria based on our predictive model.'

        # Optional: You can still keep a hard-coded rule for very low credit scores
        if credit_score < 600: # Example of an overriding rule
             status = 'Not Approved'
             reason = 'Credit score is too low for consideration.'

        prediction_result = {
            'name': name,
            'status': status,
            'reason': reason
        }

    # Render the HTML page, passing the prediction result if it exists
    return render_template('index.html', prediction_result=prediction_result)

if __name__ == '__main__':
    # Run the app in debug mode, which provides helpful error messages.
    app.run(debug=True)
