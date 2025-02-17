import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
# Import necessary libraries
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import tensorflow as tf

# Initialize Stopwords and Lemmatizer
stop_words = set(nltk.corpus.stopwords.words('english'))
lemmatizer = nltk.stem.WordNetLemmatizer()

# Preprocessing function
def preprocess_text(text):
    if not isinstance(text, str) or not text.strip():
        return "unknown"
    # Remove URLs and convert text to lowercase
    text = re.sub(r'https?://\S+|www\.\S+', '', text.lower())
    # Remove HTML tags, special characters, numbers, hashtags, and emojis
    text = re.sub(r'<.*?>|[^\w\s]|\d+|#\w+|[\U00010000-\U0010ffff]', '', text)
    # Tokenize words
    words = nltk.word_tokenize(text)
    # Lemmatize and remove stopwords
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Load dataset
<<<<<<< HEAD
file_path = r'data/emotion-emotion_69k.csv'  # Update with your dataset's path
=======
file_path = r'data\emotion-emotion_69k.csv'  # Update with your dataset's path
>>>>>>> 631208bf4a71d5c4b81c524a65c62f94d54d3390
df = pd.read_csv(file_path)

# Apply preprocessing to the 'Situation' column
df['Situation'] = df['Situation'].apply(preprocess_text)

# Fill missing emotions with 'unknown'
df['emotion'] = df['emotion'].fillna('unknown')

# Filter valid emotions
valid_emotions = [
    'sentimental', 'afraid', 'proud', 'faithful', 'terrified', 'joyful',
    'angry', 'sad', 'jealous', 'grateful', 'prepared', 'embarrassed',
    'excited', 'annoyed', 'lonely', 'ashamed', 'guilty', 'surprised',
    'nostalgic', 'confident', 'furious', 'disappointed', 'caring',
    'trusting', 'disgusted', 'anticipating', 'anxious', 'hopeful',
    'content', 'impressed', 'apprehensive', 'devastated'
]
df = df[df['emotion'].isin(valid_emotions)]

# Encode labels
label_encoder = LabelEncoder()
df['emotion_encoded'] = label_encoder.fit_transform(df['emotion'])

# Save label encoder classes to a .npy file
np.save('data/label_encoder_classes.npy', label_encoder.classes_)

# Optional: Print the classes to verify
print("Label classes:", label_encoder.classes_)

# Save the preprocessed data to a new CSV file
preprocessed_csv_path = r'data\preprocessed_emotion_data.csv'
df.to_csv(preprocessed_csv_path, index=False)
print(f"Preprocessed data saved to {preprocessed_csv_path}.")

# Split dataset into train, validation, and test sets
X_train_texts, X_temp_texts, y_train, y_temp = train_test_split(
    df['Situation'].values, df['emotion_encoded'].values, test_size=0.2, random_state=42
)
X_val_texts, X_test_texts, y_val, y_test = train_test_split(
    X_temp_texts, y_temp, test_size=0.5, random_state=42
)

# Load Sentence-BERT model
print("Loading Sentence-BERT...")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')  # You can replace with other models as needed

# Generate SBERT embeddings
def get_sbert_embeddings(texts):
    return np.array(sbert_model.encode(texts, batch_size=32, show_progress_bar=True))

print("Generating SBERT embeddings...")
X_train_embeddings = get_sbert_embeddings(X_train_texts)
X_val_embeddings = get_sbert_embeddings(X_val_texts)
X_test_embeddings = get_sbert_embeddings(X_test_texts)

# Save embeddings and labels to files
<<<<<<< HEAD
np.save(r'data/X_train_embeddings.npy', X_train_embeddings)
np.save(r'data/X_val_embeddings.npy', X_val_embeddings)
np.save(r'data/X_test_embeddings.npy', X_test_embeddings)
np.save(r'data/y_train.npy', y_train)
np.save(r'data/y_val.npy', y_val)
np.save(r'data/y_test.npy', y_test)
=======
np.save(r'data\X_train_embeddings.npy', X_train_embeddings)
np.save(r'data\X_val_embeddings.npy', X_val_embeddings)
np.save(r'data\X_test_embeddings.npy', X_test_embeddings)
np.save(r'data\y_train.npy', y_train)
np.save(r'data\y_val.npy', y_val)
np.save(r'data\y_test.npy', y_test)
>>>>>>> 631208bf4a71d5c4b81c524a65c62f94d54d3390
print("Embeddings and labels saved.")
