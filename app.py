import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer

# Load the label encoder and preprocessed data
label_encoder = LabelEncoder()

label_encoder.classes_ = np.load('data/label_encoder_classes.npy', allow_pickle=True)


# Load models
model_paths = {
    "CNN_GRU": "models/CNN_GRU.pkl",
    "CNN_BiGRU": "models/CNN_BiGRU.pkl",
    "CNN_LSTM": "models/CNN_LSTM.pkl",
    "CNN_BiLSTM": "models/CNN_BiLSTM.pkl"
}

models = {}
for model_name, model_path in model_paths.items():
    with open(model_path, "rb") as f:
        models[model_name] = pickle.load(f)

# Load preprocessed embeddings
X_test_embeddings = np.load('data/X_test_embeddings.npy')
y_test = np.load('data/y_test.npy')

# Function to display metrics
def show_metrics(model, X_test, y_test):
    y_pred_probs = model.predict(X_test)
    y_pred = y_pred_probs.argmax(axis=1)
    
    # Classification Report
    st.subheader('Classification Report')
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    st.text(report)

    # Confusion Matrix
    st.subheader('Confusion Matrix')
    conf_matrix = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    st.pyplot(fig)

    # Additional Metrics: Precision, Recall, F1-Score
    precision = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
    recall = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    f1_score = 2 * (precision * recall) / (precision + recall)
    metrics_df = pd.DataFrame({
        "Class": label_encoder.classes_,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1_score
    })
    st.subheader('Per-Class Metrics')
    st.write(metrics_df)

    # Plot Precision, Recall, F1-Score
    st.subheader('Precision, Recall, and F1-Score per Class')
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    metrics_df.plot(x='Class', kind='bar', ax=ax2)
    ax2.set_title('Precision, Recall, and F1-Score per Class')
    ax2.set_ylabel('Score')
    ax2.set_xlabel('Class')
    st.pyplot(fig2)

# Load Sentence-BERT model for inference
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to predict emotion
def predict_emotion(text, model):
    embedding = sbert_model.encode([text])
    prediction = model.predict(embedding)
    predicted_class = prediction.argmax(axis=1)[0]
    predicted_emotion = label_encoder.inverse_transform([predicted_class])[0]
    return predicted_emotion

# Streamlit interface
st.title("MindReader")
st.write("Select a model, enter a sentence, and see the predicted emotion along with the metrics.")

# Choose model
model_choice = st.selectbox("Choose a model", options=["CNN_GRU", "CNN_BiGRU", "CNN_LSTM", "CNN_BiLSTM"])

# Text input
input_text = st.text_area("Enter a sentence to detect emotion", "")

if input_text:
    # Predict emotion
    st.subheader("Predicted Emotion:")
    predicted_emotion = predict_emotion(input_text, models[model_choice])
    st.write(predicted_emotion)

    # Show model metrics
    if st.button("Show Metrics"):
        show_metrics(models[model_choice], X_test_embeddings, y_test)

