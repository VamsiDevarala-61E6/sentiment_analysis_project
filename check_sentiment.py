
#import modules required
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from tensorflow.keras.models import load_model

#preprocess the input , remove numbers and Nan
def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'\d+', '', text)
    return text

# Load the tokenizer
with open('tkn.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


#load the model
model = load_model('sentiment.keras')

# Define the function to predict sentiment of new feedback
def predict_sentiment(feedback):
    feedback = preprocess_text(feedback)
    sequence = tokenizer.texts_to_sequences([feedback])
    padded_sequence = pad_sequences(sequence, maxlen=100)

    prediction = model.predict(padded_sequence)
    sentiment_labels = ['negative 😢', 'neutral 😐', 'positive 😁']
    sentiment = sentiment_labels[prediction.argmax()]
    return sentiment

# Define the check function to verify model import and prediction

# Get feedback from the user for sentiment analysis
#runs in infinite loop, 'ex' to exit the loop
while True:
    st = input()
    if st=='ex':
        break
    else:
        sentiment = predict_sentiment(st)
        print(f"Sentiment: {sentiment}")
