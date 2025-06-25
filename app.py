from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import string

# Download NLTK stopwords
nltk.download("punkt")
nltk.download("stopwords")

app = Flask(__name__)

# Load the trained brain tumor detection model
tumor_model = load_model("F:/brain tumor/brain_tumor_detection_model.h5")

# Load intents from JSON
with open("F:/brain tumor/brain_tumor_intents.json", "r") as file:
    intents = json.load(file)

# Preprocess text: tokenize, remove stopwords, and punctuation
def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stopwords.words("english") and word not in string.punctuation]
    return " ".join(tokens)

# Prepare patterns and responses for intent matching
patterns = []
responses = []
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(preprocess_text(pattern))
        responses.append(intent["responses"])

# Convert patterns to TF-IDF vectors
vectorizer = TfidfVectorizer()
pattern_vectors = vectorizer.fit_transform(patterns)

# Function to get the closest intent
def get_closest_intent(user_input):
    user_input = preprocess_text(user_input)
    user_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vector, pattern_vectors)
    closest_index = similarities.argmax()
    return responses[closest_index]

# Function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))  # Resize to match model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize pixel values
    return img_array

# Function to make predictions
def predict_tumor(img_path):
    img_array = preprocess_image(img_path)
    prediction = tumor_model.predict(img_array)
    return "Tumor Detected! Please consult a doctor immediately!!" if prediction[0][0] > 0.5 else "No Tumor Detected!!"

# Route for the home page
@app.route("/")
def home():
    return render_template("index.html")

# Route to handle chatbot messages
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]
    response = random.choice(get_closest_intent(user_input))
    return jsonify({"response": response})

# Route to handle image upload and prediction
@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"})
    
    # Save the uploaded file
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)
    
    # Make prediction
    result = predict_tumor(file_path)
    return jsonify({"result": result})

if __name__ == "__main__":
    # Create uploads directory if it doesn't exist
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    
    app.run(debug=True)