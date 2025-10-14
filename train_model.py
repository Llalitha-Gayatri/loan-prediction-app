import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

print("--- Starting Model Training ---")

# --- 1. Load the Dataset ---
try:
    df = pd.read_csv('loan_data.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'loan_data.csv' not found. Please place it in the project directory.")
    exit()

# --- 2. Data Preprocessing ---
# **FIXED**: Using column names that match your CSV file.
features_to_use = ['income', 'loan_amount', 'credit_score']
target = 'loan_approved'

# Drop rows where our key columns have missing values
df = df.dropna(subset=features_to_use + [target]) 

# **FIXED**: Convert the 'loan_approved' column (True/False) into numbers (1/0)
df[target] = df[target].astype(int)

print("Data preprocessing complete.")
print(f"Dataset shape after cleaning: {df.shape}")


# --- 3. Define Features (X) and Target (y) ---
# **FIXED**: Using the correct feature names for X
X = df[features_to_use]
y = df[target]


# --- 4. Model Training ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# --- 5. Model Evaluation ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy on Test Data: {accuracy * 100:.2f}%")


# --- 6. Save the Final Model ---
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nModel trained on real data and saved as model.pkl")


