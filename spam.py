import streamlit as st
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model and tokenizer
model = load_model('spam_classifier_lstm.h5')

with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Settings
maxlen = 100  # Use the same as during training

# Streamlit UI
st.title("Spam Detection with LSTM")
input_text = st.text_area("Enter your message:")

if st.button("Classify"):
    if input_text:
        # Preprocess input
        sequence = tokenizer.texts_to_sequences([input_text])
        padded = pad_sequences(sequence, maxlen=maxlen)

        # Predict
        prediction = model.predict(padded)[0][0]
        label = "Spam" if prediction > 0.5 else "Not Spam"

        st.write(f"### Prediction: {label} ({prediction:.2f})")
    else:
        st.warning("Please enter a message to classify.")
