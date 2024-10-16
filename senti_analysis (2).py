import re
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.utils import to_categorical
import pickle

    
def preprocess_text(text):
    # Remove numbers
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'\d+', '', text)
    return text

# Load your dataset
data = pd.read_csv(r"C:\Users\91961\OneDrive\Desktop\gen_ai_DR\datasets\train_sentiment.csv", encoding='iso-8859-1')  # Update with your file path
print(data.head())

# Preprocess the text data to remove numbers
data['text'] = data['text'].apply(preprocess_text)

# Encode labels
le = LabelEncoder()
data['sentiment'] = le.fit_transform(data['sentiment'])  # Assuming 'sentiment' is the sentiment column

# Convert labels to one-hot encoding
y = to_categorical(data['sentiment'], num_classes=3)

# Split data into features and labels
X = data['text'].values  # Assuming 'text' is the text column

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

max_words = 100
X_train = pad_sequences(X_train, maxlen=max_words)
X_test = pad_sequences(X_test, maxlen=max_words)

# Build the LSTM model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=max_words))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3, activation = 'softmax'))  # Update to 3 units for 3 classes

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

score = model.evaluate(X_test,y_test,verbose=1)
# Print model summary
print(model.summary())

# Train the model
batch_size = 128
epochs = 12
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Function to predict sentiment of new feedback
def predict_sentiment(feedback):
    # Preprocess the feedback
    feedback = preprocess_text(feedback)  # Apply preprocessing to remove numbers
    sequence = tokenizer.texts_to_sequences([feedback])
    padded_sequence = pad_sequences(sequence, maxlen=max_words)

    # Predict sentiment
    prediction = model.predict(padded_sequence)
    sentiment_labels = ['negative', 'neutral', 'positive']
    sentiment_labels = ['negative 😢','neutral 😐','positive😁']
    sentiment = sentiment_labels[prediction.argmax()]
    return sentiment


sample_feedback = input()
sentiment = predict_sentiment(sample_feedback)
print(f"Sentiment: {sentiment}")

model.save('sentiment.keras')
