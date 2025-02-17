import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, GRU, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import pickle
import os

# Global Constants
EMBEDDING_DIM = 384  # SBERT embedding size (for all-MiniLM-L6-v2)
FILTERS = 128
KERNEL_SIZE = 3
POOL_SIZE = 2
HIDDEN_UNITS = 100
DROPOUT_RATE = 0.5
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 32

# Paths (provide your file paths here)
EMBEDDINGS_TRAIN_PATH = "data\X_train_embeddings.npy"
EMBEDDINGS_VAL_PATH = "data\X_val_embeddings.npy"
EMBEDDINGS_TEST_PATH = "data\X_test_embeddings.npy"
DATASET_PATH = "data\preprocessed_emotion_data.csv"
MODELS_PATH = "models"  # Directory where models will be saved

# Create models directory if it doesn't exist
os.makedirs(MODELS_PATH, exist_ok=True)
# Load Dataset
data = pd.read_csv(DATASET_PATH)
classes = data['emotion'].nunique()  # Extract the number of unique classes
print(f"Number of classes: {classes}")

# Load Embeddings
X_train_embeddings = np.load(EMBEDDINGS_TRAIN_PATH)
X_val_embeddings = np.load(EMBEDDINGS_VAL_PATH)
X_test_embeddings = np.load(EMBEDDINGS_TEST_PATH)
# Load Labels
y_train = np.load("data\y_train.npy")
y_val = np.load("data\y_val.npy")
y_test = np.load("data\y_test.npy")


# Model Definitions
def build_cnn_gru_model():
    input_layer = Input(shape=(EMBEDDING_DIM,), name="Input")
    reshaped_layer = tf.keras.layers.Reshape((1, EMBEDDING_DIM))(input_layer)
    conv_layer = Conv1D(FILTERS, KERNEL_SIZE, activation='relu', padding='same')(reshaped_layer)
    pooling_layer = MaxPooling1D(POOL_SIZE, padding='same')(conv_layer)
    gru_layer_1 = GRU(HIDDEN_UNITS, return_sequences=True)(pooling_layer)
    gru_layer_2 = GRU(HIDDEN_UNITS)(gru_layer_1)
    dropout_layer = Dropout(DROPOUT_RATE)(gru_layer_2)
    dense_layer = Dense(128, activation='relu')(dropout_layer)
    output_layer = Dense(classes, activation='softmax')(dense_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_cnn_bi_gru_model():
    input_layer = Input(shape=(EMBEDDING_DIM,), name="Input")
    reshaped_layer = tf.keras.layers.Reshape((1, EMBEDDING_DIM))(input_layer)
    conv_layer = Conv1D(FILTERS, KERNEL_SIZE, activation='relu', padding='same')(reshaped_layer)
    pooling_layer = MaxPooling1D(POOL_SIZE, padding='same')(conv_layer)
    bi_gru_layer_1 = Bidirectional(GRU(HIDDEN_UNITS, return_sequences=True))(pooling_layer)
    bi_gru_layer_2 = Bidirectional(GRU(HIDDEN_UNITS))(bi_gru_layer_1)
    dropout_layer = Dropout(DROPOUT_RATE)(bi_gru_layer_2)
    dense_layer = Dense(128, activation='relu')(dropout_layer)
    output_layer = Dense(classes, activation='softmax')(dense_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
def build_cnn_lstm_model():
    input_layer = Input(shape=(EMBEDDING_DIM,), name="Input")
    reshaped_layer = tf.keras.layers.Reshape((1, EMBEDDING_DIM))(input_layer)
    conv_layer = Conv1D(FILTERS, KERNEL_SIZE, activation='relu', padding='same', name="Conv1D")(reshaped_layer)
    pooling_layer = MaxPooling1D(POOL_SIZE, padding='same', name="MaxPooling1D")(conv_layer)
    lstm_layer_1 = LSTM(HIDDEN_UNITS, return_sequences=True, name="LSTM_1")(pooling_layer)
    lstm_layer_2 = LSTM(HIDDEN_UNITS, name="LSTM_2")(lstm_layer_1)
    dropout_layer = Dropout(DROPOUT_RATE, name="Dropout")(lstm_layer_2)
    dense_layer = Dense(128, activation='relu', name="Dense_128")(dropout_layer)
    output_layer = Dense(classes, activation='softmax', name="Output")(dense_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_cnn_bilstm_model():
    input_layer = Input(shape=(EMBEDDING_DIM,), name="Input")
    reshaped_layer = tf.keras.layers.Reshape((1, EMBEDDING_DIM))(input_layer)
    conv_layer = Conv1D(FILTERS, KERNEL_SIZE, activation='relu', padding='same', name="Conv1D")(reshaped_layer)
    pooling_layer = MaxPooling1D(POOL_SIZE, padding='same', name="MaxPooling1D")(conv_layer)
    bi_lstm_layer_1 = Bidirectional(LSTM(HIDDEN_UNITS, return_sequences=True), name="BiLSTM_1")(pooling_layer)
    bi_lstm_layer_2 = Bidirectional(LSTM(HIDDEN_UNITS), name="BiLSTM_2")(bi_lstm_layer_1)
    dropout_layer = Dropout(DROPOUT_RATE, name="Dropout")(bi_lstm_layer_2)
    dense_layer = Dense(128, activation='relu', name="Dense_128")(dropout_layer)
    output_layer = Dense(classes, activation='softmax', name="Output")(dense_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
# Training and Saving Models
models = {
    "CNN_GRU": build_cnn_gru_model(),
    "CNN_BiGRU": build_cnn_bi_gru_model(),
    "CNN_LSTM": build_cnn_lstm_model(),
    "CNN_BiLSTM":build_cnn_bilstm_model()
}

for model_name, model in models.items():
    print(f"Training {model_name}...")
    history = model.fit(
        X_train_embeddings, y_train,
        validation_data=(X_val_embeddings, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )
    print(f"Saving {model_name}...")
    model_save_path = os.path.join(MODELS_PATH, f"{model_name}.pkl")
    with open(model_save_path, "wb") as f:
        pickle.dump(model, f)

print("All models trained and saved successfully!")
