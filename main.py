from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from sentence_transformers import SentenceTransformer

# Load models once at startup
model = joblib.load('LogReg.pkl')  # Adjust path if needed
sentence_transformer_model = SentenceTransformer(
    'sentence-transformers/paraphrase-xlm-r-multilingual-v1', device='cpu'
)

app = Flask(__name__)
CORS(app)

# Helper function to convert structured data to natural text
def concatenate_categorical_values(rating, verified, category, text):
    verified_string = "It is a verified purchase" if verified.upper() == "Y" else "It is not a verified purchase"
    return f"The rating is {rating}. {verified_string} with product category {category}. {text}"

# Preprocess a single review entry
def preprocess_review(rating, verified, category, title, review_text):
    full_text = f"{title}. {review_text}".strip()
    return concatenate_categorical_values(rating, verified, category, full_text)

# Embed text using SentenceTransformer
def generate_embeddings(texts):
    return sentence_transformer_model.encode(texts, convert_to_numpy=True)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Validate input fields
    required_fields = ['rating', 'verified', 'category', 'title', 'review_text']
    if not all(field in data for field in required_fields):
        return jsonify({'error': f'Missing one of required fields: {required_fields}'}), 400

    try:
        # Preprocess and embed
        processed_text = preprocess_review(
            data['rating'], data['verified'], data['category'], data['title'], data['review_text']
        )
        embedding = generate_embeddings([processed_text])

        # Predict
        prediction = model.predict(embedding)[0]
        proba = model.predict_proba(embedding)[0]

        fake_prob = proba[1] * 100
        genuine_prob = proba[0] * 100
        label = "Fake" if prediction == 1 else "Genuine"

        return jsonify({
            'review_text': processed_text,
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
