# Importing packages
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import matplotlib.pyplot as plt

import joblib

#Loading pipeline
lr_pipeline = joblib.load(open("Model/emotion_classifier_logistic_regression_pipeline_04_05_23.pkl","rb"))

#Prediction Function
def predict_emotions(text):
    results = lr_pipeline.predict([text])
    return results[0]


def get_emotion_probability(text):
    results = lr_pipeline.predict_proba([text])
    return results


def main():
    st.set_page_config(page_title="EmoExpert", page_icon=":sunglasses:")

    hcol1, hcol2 = st.columns([1, 2])

    with hcol1:
        st.title("EmoExpert")

    with hcol2:
        st.image("assets/logo.png", width=90, clamp=True)
    menu = ["Home", "About"]
    options = st.sidebar.selectbox("Menu", menu)


    if options == "Home":
        st.subheader("Empowering emotional intelligence with AI")

        with st.form(key='emotion_text_form'):
            raw_text = st.text_area('Enter Text Here')
            submit_button = st.form_submit_button(label="Show Emotion")

        if submit_button:
            col1, col2 = st.columns(2)

            #calling functions defined above
            predictions = predict_emotions(raw_text)
            probability = get_emotion_probability(raw_text)

            with col1:
                st.info("Prediction")
                st.write(predictions)

                # Show corresponding GIF based on prediction
                if predictions == "joy":
                    st.image("https://i.gifer.com/7F5y.gif")
                elif predictions == "sadness":
                    st.image("https://i.gifer.com/Fv3X.gif")
                elif predictions == "fear":
                    st.image("https://i.gifer.com/93c5.gif")
                elif predictions == "disgust":
                    st.image("https://i.gifer.com/Fv44.gif")
                elif predictions == "anger":
                    st.image("https://i.gifer.com/AQRR.gif")
                elif predictions == "neutral":
                    st.image("https://i.gifer.com/CmND.gif")
                elif predictions == "shame":
                    st.image("https://i.gifer.com/Fv3X.gif")

                
                st.write("Confidence level: {}%".format(round(np.max(probability)*100, 2)))

            with col2:
               st.info("Prediction probability")
               st.write('Probability graph')
                # Create pandas DataFrame and plot emotion countplot
               data = {'emotion': lr_pipeline.classes_, 'probability': probability[0]}
               df = pd.DataFrame(data)
               df['emotion'].replace({'joy': 'Joy', 'sadness': 'Sadness', 'fear': 'Fear', 'disgust': 'Disgust', 'anger': 'Anger', 'neutral': 'Neutral'}, inplace=True)

               fig = px.bar(df, x='emotion', y='probability', color='emotion', height=500)
               fig.update_layout(title='Predicted Emotion Probability', xaxis_title='Emotion', yaxis_title='Probability')
               st.plotly_chart(fig, use_container_width=True)
                            

    else:
        st.subheader("About")
        st.write(''' EmoExpert is an emotion recognition application that uses natural
          language processing and machine learning techniques to analyze text input and predict 
          the user's emotional state. It allows users to submit text and receive a prediction of
            the dominant emotion expressed in the text, along with a probability score for each of
            the possible emotions. EmoExpert can be used for various purposes, such as improving communication skills, providing emotional support, or monitoring emotional states in mental health settings. Its simple and user-friendly interface makes it easy for anyone to use, and its accuracy and speed make it a valuable tool for individuals, businesses, and healthcare professionals alike.
        ''')


if __name__ == '__main__':
    main()
