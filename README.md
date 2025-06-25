# medical-chatbot
✅ Step 1: Environment Setup
Clone the repository.
Install dependencies:
pip install numpy opencv-python tensorflow flask nltk matplotlib

🧠 Step 2: Train the CNN Model
Navigate to the training script directory.
Train the CNN model using brain MRI image dataset:
python train.py
The model will classify images as Tumor or No Tumor.

💬 Step 3: Develop the Chatbot using NLP
Define intents (symptoms, causes, treatment, etc.) in a JSON file.
Use nltk for tokenization and text preprocessing.
Train a chatbot model using the intents and save it for predictions.

🌐 Step 4: Build Web Interface using Flask
Design frontend using HTML/CSS for user interaction.
Create Flask routes for:
Chatbot interaction (/get)
Image upload & tumor prediction (/predict)

🔗 Step 5: Integrate Chatbot with CNN Model
Combine NLP responses and image prediction in the Flask backend.
Users can:
Ask medical queries.
Upload MRI scans to detect tumor.

