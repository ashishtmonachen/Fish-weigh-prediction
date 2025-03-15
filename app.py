from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model, scaler, and encoder
model = joblib.load("fish_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder_columns = joblib.load("encoder.pkl")

@app.route('/')
def home():
    return render_template("index.html")  

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])

        # One-Hot Encode 'Species'
        df = pd.get_dummies(df)
        df = df.reindex(columns=encoder_columns, fill_value=0)

        # Scale numeric features
        df_scaled = scaler.transform(df)

        # Predict fish weight
        prediction = model.predict(df_scaled)

        return jsonify({"prediction": float(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
