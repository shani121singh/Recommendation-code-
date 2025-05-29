from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load model and preprocessing objects
with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

categorical_cols = ['gender', 'marital_status', 'employment_status']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract form data
    data = {key: request.form[key] for key in request.form}

    # Convert values to correct types
    input_data = {
        'age': int(data['age']),
        'gender': data['gender'],
        'income': int(data['income']),
        'marital_status': data['marital_status'],
        'children': int(data['children']),
        'employment_status': data['employment_status'],
        'health_score': int(data['health_score']),
        'owns_home': int(data['owns_home']),
        'credit_score': int(data['credit_score']),
        'past_claims': int(data['past_claims']),
        'vehicles_owned': int(data['vehicles_owned']),
        'years_with_company': int(data['years_with_company'])
    }

    df = pd.DataFrame([input_data])

    # Encode categorical variables
    for col in categorical_cols:
        le = label_encoders[col]
        df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

    # Reorder columns as expected
    expected_cols = ['age', 'gender', 'income', 'marital_status', 'children', 'employment_status',
                     'health_score', 'owns_home', 'credit_score', 'past_claims', 'vehicles_owned', 'years_with_company']
    df = df[expected_cols]

    # Scale and predict
    scaled = scaler.transform(df)
    pred_encoded = model.predict(scaled)[0]
    prediction = label_encoders['recommended_product'].inverse_transform([pred_encoded])[0]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
