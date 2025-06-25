import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageTk
import json
import random

# Load the trained brain tumor detection model
tumor_model = load_model("brain_tumor_detection_model.h5")

# Load intents from JSON
with open("brain_tumor_intents.json", "r") as file:
    intents = json.load(file)

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
    return "Tumor Detected! Please consult a doctor immediately!!" if prediction[0][0] > 0.5 else "No Tumor Detected!"

# Function to handle image upload
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        # Display the image
        img = Image.open(file_path)
        img = img.resize((150, 150))
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img  # Keep a reference to avoid garbage collection

        # Make prediction
        result = predict_tumor(file_path)
        chat_log.insert(tk.END, f"System: {result}\n\n")

# Function to get a response from the chatbot
def get_response(user_input):
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            if pattern.lower() in user_input.lower():
                return random.choice(intent["responses"])
    return "I'm sorry, I don't understand that. Can you please rephrase?"

# Function to handle user input
def send_message():
    user_input = user_entry.get()
    if user_input.strip() == "":
        return

    # Display user message in the chat log
    chat_log.insert(tk.END, f"You: {user_input}\n\n")

    # Respond to user input
    if "upload image" in user_input.lower():
        chat_log.insert(tk.END, "System: Please click the 'Upload Image' button to upload a brain image for tumor detection.\n\n")
    else:
        response = get_response(user_input)
        chat_log.insert(tk.END, f"System: {response}\n\n")

    # Clear the input field
    user_entry.delete(0, tk.END)

# Create the main window
root = tk.Tk()
root.title("Brain Tumor Detection Chatbot")
root.geometry("600x500")

# Create a chat log area
chat_log = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=70, height=20)
chat_log.pack(padx=10, pady=10)

# Create an entry field for user input
user_entry = tk.Entry(root, width=50)
user_entry.pack(pady=10)

# Create a button to send user input
send_button = tk.Button(root, text="Send", command=send_message)
send_button.pack()

# Create a button to upload an image
upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack(pady=10)

# Create a label to display the uploaded image
image_label = tk.Label(root)
image_label.pack()

# Run the Tkinter event loop
root.mainloop()