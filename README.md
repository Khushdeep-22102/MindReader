
# MindReader

This repository contains the code and data for an emotion detection project using deep learning models and Sentence-BERT embeddings.

## Project Description

<<<<<<< HEAD
This project leverages the Empathetic Dialogues dataset, a publicly available dataset on [Kaggle](https://www.kaggle.com/datasets/atharvjairath/empathetic-dialogues-facebook-ai) consisting of 25,000 conversations grounded in emotional situations. Each data point includes:
=======
This project leverages the Empathetic Dialogues dataset, a publicly available dataset on [Kaggle](https://www.kaggle.com/datasets/parulpandey/emotion-dataset) consisting of 25,000 conversations grounded in emotional situations. Each data point includes:
>>>>>>> 631208bf4a71d5c4b81c524a65c62f94d54d3390
  

- **Situation**: A brief context describing the emotional scenario.  
- **Emotion**: The dominant emotion expressed in the dialogue (e.g., sentimental, excited).  
- **Empathetic Dialogues**: Exchanges between a customer and an agent, emphasizing empathy in responses.  
- **Labels**: Target responses designed to align with the expressed emotion.  

The raw data (stored in `emotion-emotion_69k.csv`) is preprocessed to clean text, encode emotions into numerical labels, and extract semantic features using **Sentence-BERT (SBERT)** embeddings. These embeddings are fed into four neural network models—CNN-GRU, CNN-BiGRU, CNN-LSTM, and CNN-BiLSTM—that blend convolutional layers for feature extraction with sequential layers for contextual understanding, enabling precise emotion classification.

## Getting Started

Follow these steps to set up the project and run the code.

### Prerequisites

- Python 3.8 or later
- Required Python libraries (see [requirements.txt](requirements.txt))
- NLTK resources: stopwords, punkt, and wordnet

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Khushdeep-22102/MindReader.git
   cd MindReader
   ```

2. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download necessary NLTK resources:

   ```bash
   python -m nltk.downloader stopwords punkt wordnet
   ```

### Preprocessing

1. Place your raw dataset in the `data` directory.
2. Update the file path in `preprocess.py` to point to your dataset.
3. Run the preprocessing script:

   ```bash
   python scripts/preprocess.py
   ```

This will preprocess the dataset, generate Sentence-BERT embeddings, and save the processed data and labels in the `data` directory.

### Training

1. Ensure the preprocessed data is available in the `data` directory.
2. Run the training script:

   ```bash
   python scripts/train.py
   ```

This will train four models (CNN-GRU, CNN-BiGRU, CNN-LSTM, CNN-BiLSTM) and save them in the `models` directory.

### Evaluation

1. Run the evaluation script to assess model performance:

   ```bash
   python scripts/evaluate.py
   ```

This will display performance metrics, including precision, recall, F1-score, and a confusion matrix.

### Streamlit App

1. Launch the Streamlit app:

   ```bash
   streamlit run app.py
   ```

2. Use the app to input text, select a model, and detect the emotion.

## Results

- The trained models are evaluated based on accuracy, precision, recall, and F1-score.
- Confusion matrices and per-class metrics are visualized for better insights.



## Acknowledgments

- [NLTK](https://www.nltk.org/)
- [Sentence-BERT](https://www.sbert.net/)
- [TensorFlow](https://www.tensorflow.org/)
- [Streamlit](https://streamlit.io/)

