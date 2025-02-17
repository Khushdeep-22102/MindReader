import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load dataset and embeddings
<<<<<<< HEAD
DATASET_PATH = "data/preprocessed_emotion_data.csv"
=======
DATASET_PATH = "data\preprocessed_emotion_data.csv"
>>>>>>> 631208bf4a71d5c4b81c524a65c62f94d54d3390
EMBEDDINGS_TEST_PATH = "data/X_test_embeddings.npy"
y_test_path = "data/y_test.npy"

data = pd.read_csv(DATASET_PATH)
X_test_embeddings = np.load(EMBEDDINGS_TEST_PATH)
y_test = np.load(y_test_path)

# Load label encoder (assuming label encoder was saved during training)
label_encoder = LabelEncoder()
label_encoder.fit(data['emotion'])

# Function to evaluate model
def evaluate_model(model, X_test, y_test):
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Predict probabilities and classes
    y_pred_probs = model.predict(X_test, verbose=1)
    y_pred = y_pred_probs.argmax(axis=1)

    # Convert y_test and y_pred to original labels
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    # Classification Report
    print("\nClassification Report:")
    # Ensure target_names is a list of string labels
    target_names = list(label_encoder.classes_.astype(str))
    print(classification_report(y_test_labels, y_pred_labels, target_names=target_names))

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

    # Additional Metrics: Precision, Recall, and F1-score
    precision = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
    recall = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    f1_score = 2 * (precision * recall) / (precision + recall)
    metrics_df = pd.DataFrame({
        "Class": target_names,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1_score
    })
    print("\nPer-Class Metrics:")
    print(metrics_df)

    # Plot Precision, Recall, F1-Score
    plt.figure(figsize=(12, 6))
    metrics_df.plot(x="Class", kind="bar", figsize=(16, 8))
    plt.title("Precision, Recall, and F1-Score per Class")
    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.legend(loc="lower right")
    plt.grid(axis="y")
    plt.show()

# Load models and evaluate them
MODELS_PATH = "models"  # Path to models directory
models = {
    "CNN_GRU": os.path.join(MODELS_PATH, "CNN_GRU.pkl"),
    "CNN_BiGRU": os.path.join(MODELS_PATH, "CNN_BiGRU.pkl"),
    "CNN_LSTM": os.path.join(MODELS_PATH, "CNN_LSTM.pkl"),
    "CNN_BiLSTM": os.path.join(MODELS_PATH, "CNN_BiLSTM.pkl")
}

for model_name, model_path in models.items():
    print(f"Evaluating {model_name} Model...")
    
    # Load the model
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    # Evaluate the model
    evaluate_model(model, X_test_embeddings, y_test)
