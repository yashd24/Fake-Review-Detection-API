from flask import Flask, request, jsonify
import joblib

# Load your trained model
model = joblib.load('LogReg.pkl')  # Adjust path if needed

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    review_text = data.get('review', '')

    if not review_text.strip():
        return jsonify({'error': 'Review text is required.'}), 400

    try:
        # Prediction & probability
        prediction = model.predict([review_text])[0]
        proba = model.predict_proba([review_text])[0]

        fake_prob = proba[1] * 100  # Assuming index 1 is "Fake"
        genuine_prob = proba[0] * 100
        label = "Fake" if prediction == 1 else "Genuine"

        return jsonify({
            'review': review_text,
            'prediction': label,
            'probabilities': {
                'fake': round(fake_prob, 2),
                'genuine': round(genuine_prob, 2)
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
