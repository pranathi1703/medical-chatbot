import tkinter as tk
from tkinter import filedialog, messagebox
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageTk

# Load the trained model
model = load_model("brain_tumor_detection_model.h5")

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
    prediction = model.predict(img_array)
    return "Tumor Detected! Please consult a doctor immediately!!" if prediction[0][0] > 0.5 else "No Tumor Detected!!"

# Function to handle image upload
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        # Display the image
        img = Image.open(file_path)
        img = img.resize((200, 200))
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img  # Keep a reference to avoid garbage collection

        # Make prediction
        result = predict_tumor(file_path)
        result_label.config(text=result)

# Create the main window
root = tk.Tk()
root.title("Brain Tumor Detection")
root.geometry("400x400")

# Create a button to upload an image
upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack(pady=20)

# Create a label to display the image
panel = tk.Label(root)
panel.pack()

# Create a label to display the prediction result
result_label = tk.Label(root, text="", font=("Arial", 16))
result_label.pack(pady=20)

# Run the Tkinter event loop
root.mainloop()