import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('Logistic Regression_model.joblib')

# Key for scale of measurement
key = {0:'select what appply',
    1: 'Doesn\'t look like me at all.',
    2: 'Looks somewhat unlike me.',
    3: 'Looks a little unlike me.',
    4: 'Looks somewhat like me.',
    5: 'Looks quite a bit like me.',
    6: 'Looks just like me.'
}
def predict_adhd(features):
    # Convert features to numpy array
    features_np = np.array(features).reshape(1, -1)
    # Make prediction
    prediction = model.predict(features_np)
    return prediction[0]

# Streamlit UI
def main():
    st.title('ADHD Prediction App')
    st.sidebar.header('User Input Features')

    # Collect user input features
    age = st.sidebar.slider('Age', 1, 100, 25, 1)

    tr1 = st.sidebar.selectbox('I don\'t pay attention to details', list(key.values()), 0)
    tr2 = st.sidebar.selectbox('Accused of inaccuracies in work', list(key.values()), 0)
    tr8 = st.sidebar.selectbox('Mind often elsewhere', list(key.values()), 0)
    tr12 = st.sidebar.selectbox('Difficulty organizing time', list(key.values()), 0)
    tr14 = st.sidebar.selectbox('Difficulty organizing tasks with several steps', list(key.values()), 0)
    tr15 = st.sidebar.selectbox('Avoid tasks requiring sustained mental effort', list(key.values()), 0)
    tr19 = st.sidebar.selectbox('Lose things needed for work', list(key.values()), 0)
    tr21 = st.sidebar.selectbox('Easily distracted by environment', list(key.values()), 0)
    tr23 = st.sidebar.selectbox('Easily distracted by surroundings', list(key.values()), 0)
    tr24 = st.sidebar.selectbox('Subject to forgetfulness in daily life', list(key.values()), 0)
    tr26 = st.sidebar.selectbox('Wiggle hands or feet on seat', list(key.values()), 0)
    tr27 = st.sidebar.selectbox('Trouble sitting still', list(key.values()), 0)
    tr28 = st.sidebar.selectbox('Leave seat unnecessarily during a meeting', list(key.values()), 0)
    tr29 = st.sidebar.selectbox('Leave seat unnecessarily at work', list(key.values()), 0)
    tr33 = st.sidebar.selectbox('Hard to stay in place when demanded', list(key.values()), 0)
    tr34 = st.sidebar.selectbox('Entourage finds you difficult to follow', list(key.values()), 0)
    tr37 = st.sidebar.selectbox('Tend to monopolize conversations', list(key.values()), 0)
    tr38 = st.sidebar.selectbox('Finish other people\'s sentences', list(key.values()), 0)
    tr40 = st.sidebar.selectbox('Difficult to wait turn in a conversation', list(key.values()), 0)

    # Map selected values back to the corresponding keys
    tr1 = next((k for k, v in key.items() if v == tr1), None)
    tr2 = next((k for k, v in key.items() if v == tr2), None)
    tr8 = next((k for k, v in key.items() if v == tr8), None)
    tr12 = next((k for k, v in key.items() if v == tr12), None)
    tr14 = next((k for k, v in key.items() if v == tr14), None)
    tr15 = next((k for k, v in key.items() if v == tr15), None)
    tr19 = next((k for k, v in key.items() if v == tr19), None)
    tr21 = next((k for k, v in key.items() if v == tr21), None)
    tr23 = next((k for k, v in key.items() if v == tr23), None)
    tr24 = next((k for k, v in key.items() if v == tr24), None)
    tr26 = next((k for k, v in key.items() if v == tr26), None)
    tr27 = next((k for k, v in key.items() if v == tr27), None)
    tr28 = next((k for k, v in key.items() if v == tr28), None)
    tr29 = next((k for k, v in key.items() if v == tr29), None)
    tr33 = next((k for k, v in key.items() if v == tr33), None)
    tr34 = next((k for k, v in key.items() if v == tr34), None)
    tr37 = next((k for k, v in key.items() if v == tr37), None)
    tr38 = next((k for k, v in key.items() if v == tr38), None)
    tr40 = next((k for k, v in key.items() if v == tr40), None)

    input_features = [age, tr1, tr2, tr8, tr12, tr14, tr15, tr19, tr21, tr23, tr24, tr26, tr27, tr28, tr29, tr33, tr34, tr37, tr38, tr40]

    # Add an explanation section for influential features
    if st.button('Predict'):
        result = predict_adhd(input_features)
        confidence = model.predict_proba(np.array(input_features).reshape(1, -1)).max()

        if result == 0:
            st.success(f"The model predicts that the individual does not have ADHD with {confidence * 100:.2f}% confidence.")
            st.image('https://media.giphy.com/media/mEGSYkHW33NyJ6g9pI/giphy.gif?cid=82a1493b8kwiec84j8p2t6m00h5s0cx87397jg2qjxdgeget&ep=v1_gifs_trending&rid=giphy.gif&ct=g',  use_column_width=True)
            st.write("Congratulations! You are on the right track.")
        else:
            st.warning(f"The model predicts that the individual has ADHD with {confidence * 100:.2f}% confidence.")
            st.image('https://media.giphy.com/media/R6gvnAxj2ISzJdbA63/giphy.gif?cid=790b7611lv67ochx0wljgv5eksgt9fhoi94tf3lwgo15sgvr&ep=v1_gifs_trending&rid=giphy.gif&ct=g', use_column_width=True)
            st.write("It's okay! Many successful people have ADHD. Embrace your uniqueness!")

        # Display input feature visualization
        st.subheader('Input Feature Visualization')
        input_features_names = ['Age', 'tr1', 'tr2', 'tr8', 'tr12', 'tr14', 'tr15', 'tr19', 'tr21', 'tr23', 'tr24',
                                 'tr26', 'tr27', 'tr28', 'tr29', 'tr33', 'tr34', 'tr37', 'tr38', 'tr40']
        st.bar_chart(dict(zip(input_features_names, input_features)))

        # Display an explanation for influential features using coefficients
        st.subheader('Influential Features (Logistic Regression Coefficients)')
        coefficients = model.coef_[0]
        st.bar_chart(dict(zip(input_features_names, coefficients)))


if __name__ == '__main__':
    main()
