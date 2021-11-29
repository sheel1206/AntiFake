# -*- coding: utf-8 -*
import dash
from dash import dcc, html
import dash.dependencies as dd
import pandas as pd

from io import BytesIO
from wordcloud import WordCloud
import base64

import tensorflow as tf
#import dill

import joblib
import os
assets_path = os.getcwd() +'/static'
data1 = pd.read_csv(r"AntiFake New -- Aproach-1/Tweetdata.csv")
val = "BTC"

external_stylesheets = [{"href": "https://fonts.googleapis.com/css2?""family=Lato:wght@400;700&display=swap","rel": "stylesheet",},]

app = dash.Dash(__name__,assets_folder=assets_path,external_stylesheets=external_stylesheets)

server = app.server

app.title = 'AntiFake'

app.layout = html.Div(
    children=[
        html.Div(
            children = [
        html.H1(children="IXXO AntiFake News",className="header-title"),
        html.P(
            children="Predicting if the entered News Article\n"
            " is Fake or not and giving it a Fake news Score",
            className="header-description",
            )],
            className="header"),
        html.Div(
            
            children = [
                dcc.Textarea(
                    id='textarea_val',
                    placeholder='Enter the news you want to test if it is fake or not',
                    className='box'),
                dcc.Dropdown(
                    id='dropdown_input',
                    options=[
                        {'label': 'Covid News', 'value': 'CVD'},
                        {'label': 'Bitcoin/Crypto News', 'value': 'BTC'},
                        ],
                    searchable=False,
                    placeholder="Select News Subject",
                    className='dropbox',
                    value = val
                        )
                ]),
        html.Button('Submit', 
                    id='textarea-state-example-button', 
                    n_clicks=0,
                    className='button'),
        html.Div(id='textarea-state-example-output', className="pred"),
        html.Img(id="image_wc",className='img'),
    ],
)




@app.callback(
    dd.Output('textarea-state-example-output', 'children'),
    dd.Input('textarea-state-example-button', 'n_clicks'),
    dd.Input('dropdown_input', 'value'),
    dd.State('textarea_val', 'value')
)
def run_model(n_clicks, dropdown_input, textarea_val):
    value = textarea_val
    print( value, dropdown_input)
    if n_clicks > 0:
        #if dropdown_input == 'CVD':
            #print("Using CVD Prediction model")
            #model = tf.keras.models.load_model('CVD_model.h5')
            #tokenizer = open('CVD_tokenizer.pkl', 'rb')
            #max_len = 70
        if dropdown_input == 'BTC':
            print("Using BTC Prediction model")
            model = tf.keras.models.load_model('AntiFake New -- Aproach-1/BTC_model_500k.h5')
            tokenizer = open('AntiFake New -- Aproach-1/BTC_tokenizer_500k.pkl', 'rb')
            max_len = 50
        tokenized = joblib.load(tokenizer)
        
        X = list()
        X.append(value)
        if value == None:
            return "Enter a News Article"
        tokenized_pred = tokenized.texts_to_sequences(X)
        X = tf.keras.preprocessing.sequence.pad_sequences(tokenized_pred, maxlen=max_len)
        prediction = int(model.predict(X) * 100)
        print(prediction)
        if prediction < 50:
            return "The Entered news is Fake with a score of " + str(100 - prediction) + "%"
        else:
            return "The Entered news is Real with a score of " + str(prediction) + "%"

for i in range(len(data1)):
    data1['processed'][i] = str(data1['processed'][i])

wc_data = data1[data1.Label == 0].processed

def plot_wordcloud(data):

    d = {x: x for x in data}
    wordcloud = WordCloud(width = 500 , height = 200 , max_words = 3000).generate(" ".join(data))
    return wordcloud.to_image()

@app.callback(dd.Output('image_wc', 'src'), [dd.Input('image_wc', 'id')])
def make_image(b):
    img = BytesIO()
    plot_wordcloud(data=wc_data).save(img, format='PNG')
    return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())

if __name__ == '__main__':
    app.run_server(debug=True, port=80)