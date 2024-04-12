.from flask import Flask, request, jsonify
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import nltk
import joblib
from flask_cors import CORS


app = Flask(_name_)
CORS(app)

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')



# NLP
def lowercase_text(text):
    return text.lower()

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    text_without_punct = text.translate(translator).strip()
    return text_without_punct

def tokenize_text(text):
    tokens = word_tokenize(text)
    return tokens

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return filtered_tokens

def lemmatize_text(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

# Preprocessing Container function
def preprocess_text(text):
    text = lowercase_text(text)
    text = remove_punctuation(text)
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_text(tokens)
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# Load the TF-IDF vectorizer
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Load the Decision Tree Classifier
model = joblib.load('best_svm_classifier.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json.get('text')

    # Clean and lemmatize the text
    text = preprocess_text(text)

    text_vectorized = tfidf_vectorizer.transform([text]).toarray()

    predicted_label = model.predict(text_vectorized)[0]

    return jsonify({'predicted': predicted_label})

if _name_ == '_main_':
    app.run(host='0.0.0.0', port=8080, debug=True)
