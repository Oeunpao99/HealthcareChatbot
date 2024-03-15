import random
import streamlit as st
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(0)

# Generate synthetic data
num_samples = 1000

# Generate features
age = np.random.randint(18, 80, num_samples)
bmi = np.random.normal(25, 5, num_samples)
blood_pressure = np.random.normal(120, 10, num_samples)

# Generate target variable (binary: 0 or 1)
# Let's assume the target variable is based on age and BMI
probability = 1 / (1 + np.exp(-(age/10 + bmi/20 - 8)))  # sigmoid function
target = np.random.binomial(1, probability)

# Create DataFrame
data = pd.DataFrame(
    {'Age': age, 'BMI': bmi, 'BloodPressure': blood_pressure, 'Condition': target})

# Save data to CSV
data.to_csv('healthcare_data.csv', index=False)

# Load data from CSV
data = pd.read_csv('healthcare_data.csv')

# Split data into features and target variable
X = data.drop("Condition", axis=1)
y = data["Condition"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Load data from CSV
data = pd.read_csv('healthcare_data.csv')

# Split data into features and target variable
X = data.drop("Condition", axis=1)
y = data["Condition"]

# Initialize and train the model
model = RandomForestClassifier()
model.fit(X, y)

# Define the UI components
st.title("Healthcare Chatbot")
user_input = st.text_input("Enter your symptoms:")
submit_button = st.button("Submit")

# Backend logic (to be implemented)

# Backend logic (to be implemented)
if submit_button:
    # Dummy function for demonstration
    # Replace this with your actual logic for getting recommendations
    def get_recommendations(user_input):
        # Dictionary mapping symptoms to recommendations
        recommendations_dict = {
            "headache": "Take some pain relievers and get some rest.",
            "cough": "You may have a respiratory infection. Drink plenty of fluids and rest.",
            "fever": "Fever could be a sign of infection. Consult with a healthcare professional.",
            "fatigue": "Make sure you're getting enough sleep and consider reducing stress levels.",
            "nausea": "Avoid heavy meals and stay hydrated. Ginger tea might help alleviate nausea.",
            "sore throat": "Gargle with warm salt water and rest your voice.",
            "muscle pain": "Apply a warm compress and consider gentle stretching exercises.",
            "shortness of breath": "If severe, seek medical attention immediately. Practice deep breathing exercises.",
            "diarrhea": "Stay hydrated and avoid foods that can aggravate your stomach.",
            "loss of taste or smell": "This could be a symptom of COVID-19. Consider getting tested.",
            "chest pain": "If severe or persistent, seek emergency medical attention immediately.",
            "dizziness": "Sit or lie down and drink water. Avoid sudden movements.",
            "joint pain": "Apply ice packs and consider over-the-counter pain relievers.",
            "abdominal pain": "If severe or persistent, seek medical attention immediately.",
            "rash": "Avoid scratching the rash. Apply calamine lotion or hydrocortisone cream.",
            "chills": "Stay warm and rest. Drink warm beverages.",
            "congestion": "Use saline nasal spray and consider using a humidifier.",
            "runny nose": "Blow your nose gently and use saline nasal drops.",
            "vomiting": "Stay hydrated and avoid solid foods until vomiting subsides.",
            "back pain": "Apply heat or ice packs and consider gentle stretching exercises."
        }

        recommendations = []
        # Split user input into individual symptoms
        user_symptoms = user_input.lower().split(',')
        # Get recommendation for each symptom
        for symptom in user_symptoms:
            if symptom.strip() in recommendations_dict:
                recommendations.append(recommendations_dict[symptom.strip()])
            else:
                recommendations.append(
                    "No specific recommendation for '{}'".format(symptom.strip()))

        return recommendations

    # Call the function to process user input and get recommendations
    recommendations = get_recommendations(user_input)
    st.write("Recommendations:")
    st.write(recommendations)
