Language Detection System
Overview

Language Detection System is an NLP and Machine Learning project that identifies the language of a given text.

The system is trained using a dataset containing text samples from different languages. It uses TF-IDF for converting text into numerical features and Logistic Regression for predicting the language.

The project also includes a simple and interactive Streamlit web application where users can enter text and get the predicted language.

Features
Detects the language of input text
Supports multiple languages
Text processing using NLP techniques
TF-IDF based feature extraction
Logistic Regression classification
Interactive Streamlit interface
Simple and easy-to-use design
Displays the predicted language instantly
Technologies Used
Python
Pandas
Scikit-learn
NLP
TF-IDF
Logistic Regression
Streamlit
Pickle
Project Structure
Language_Prediction/
│
├── app.py
├── train_model.py
├── language_detection_dataset.csv
├── language_model.pkl
├── vectorizer.pkl
├── requirements.txt
└── README.md
File Description
File	Description
app.py	Streamlit web application
train_model.py	Trains and saves the ML model
language_detection_dataset.csv	Dataset used for training
language_model.pkl	Saved Logistic Regression model
vectorizer.pkl	Saved TF-IDF vectorizer
requirements.txt	Required Python libraries
README.md	Project documentation
How It Works

The project follows these basic steps:

User Input
    ↓
Text Processing
    ↓
TF-IDF Vectorization
    ↓
Machine Learning Model
    ↓
Language Prediction
    ↓
Display Result
1. Dataset

The dataset contains text examples along with their corresponding language labels.

Example:

Text	Language
Hello, how are you?	English
नमस्ते, आप कैसे हैं?	Hindi
मला मराठी शिकायला आवडते.	Marathi
2. TF-IDF

TF-IDF (Term Frequency-Inverse Document Frequency) converts text into numerical features that can be understood by a Machine Learning model.

For this project, character-level TF-IDF can be used to capture language-specific character patterns.

3. Logistic Regression

The TF-IDF features are provided to a Logistic Regression classifier.

The model learns patterns from the training data and predicts the language of new text.

4. Streamlit

Streamlit provides the user interface.

Users can:

Enter text.
Click the Detect Language button.
View the predicted language.
Installation
Step 1: Clone the Repository
git clone YOUR_GITHUB_REPOSITORY_URL
Step 2: Open the Project Folder
cd Language_Prediction
Step 3: Create a Virtual Environment
python -m venv venv
Step 4: Activate the Environment

Windows:

venv\Scripts\activate
Step 5: Install Required Libraries
pip install -r requirements.txt
Train the Model

Run:

python train_model.py

This will train the Machine Learning model and create:

language_model.pkl
vectorizer.pkl
Run the Application

After training the model, run:

streamlit run app.py

The application will open in your browser.

Example

Input:

Hello, how are you?

Output:

Detected Language: English

Another example:

मला Python शिकायला आवडते.

Output:

Detected Language: Marathi
Machine Learning Workflow
Dataset
   ↓
Load Data
   ↓
Train-Test Split
   ↓
TF-IDF Vectorization
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Save Model
   ↓
Streamlit Application
   ↓
Language Prediction
Future Improvements
Add more languages
Improve accuracy with a larger dataset
Support mixed-language text
Add confidence/probability scores
Improve text preprocessing
Deploy the application online
Add support for more Indian languages
Learning Outcomes

Through this project, I learned how to:

Work with NLP datasets
Perform text preprocessing
Use Pandas for dataset handling
Convert text into numerical features using TF-IDF
Train a Machine Learning classification model
Evaluate model performance
Save and load trained models using Pickle
Build an interactive NLP application using Streamlit
Author

Vikrant
