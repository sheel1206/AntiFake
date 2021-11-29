import streamlit as st
import joblib
from tensorflow import keras
from tensorflow.keras.preprocessing import sequence

news_title = ""
model = keras.models.load_model('BTC_model_500k.h5')

st.title("ANTI FAKE")

st.header("CHECK YOUR CRYPTO NEWS HERE!")
news_title = st.text_area('Enter your news below')

X = list()
X.append(news_title)
tokenizer = open('BTC_tokenizer_500k.pkl', 'rb')
tokenized = joblib.load(tokenizer)
max_len = 50
tokenized_pred = tokenized.texts_to_sequences(X)
X = sequence.pad_sequences(tokenized_pred, maxlen=max_len)

prediction = model.predict(X) * 100
print(prediction[0])
if st.button("Detect"):
        if prediction[0] <= 50:
          st.success("Your news is FAKE with score of " + str((100 - prediction[0])))
        else:
            st.success("Your news is REAL with score of " + str(int(prediction[0])))
