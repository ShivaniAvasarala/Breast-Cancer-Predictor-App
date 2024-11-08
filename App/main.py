import streamlit as st
import pickle as pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

def get_clean_data():
    data = pd.read_csv("Data/data.csv")
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis']= data['diagnosis'].map({'M': 1, 'B': 0})


    return data

def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")

    data = get_clean_data()

    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    input_dict = {}

    for label, key in slider_labels:
        input_dict[key]= st.sidebar.slider(
            label = label,
            min_value =float(0),
            max_value = float(data[key].max()),
            value = float(data[key].mean()),
        )
    return input_dict

def get_scaled_values(input_dict):
    data = get_clean_data()

    X = data.drop(['diagnosis'], axis=1)

    scaler = MinMaxScaler()
    scaler.fit(X)

    input_df = pd.DataFrame([input_dict])

    scaled_values = scaler.transform(input_df)
    scaled_dict = dict(zip(input_df.columns, scaled_values[0]))

    return scaled_dict

def get_radar_chart(input_data):

    input_data = get_scaled_values(input_data)
    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                'Smoothness', 'Compactness', 
                'Concavity', 'Concave Points',
                'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
            input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
            input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean',
        line_color= 'mediumpurple'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
          input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
          input_data['concave points_se'], input_data['symmetry_se'],input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error',
        line_color= 'hotpink'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
          input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
          input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
          input_data['fractal_dimension_worst']
        ],
            theta=categories,
            fill='toself',
            name='Worst Value',
            line_color= 'lightblue'
        ))

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1]
        )),
    showlegend=True
    )

    return fig

def add_predictions(input_data):
    model = pickle.load(open("Model/model.pkl", "rb"))
    scaler = pickle.load(open("Model/scaler.pkl", "rb"))

    input_array = np.array(list(input_data.values())).reshape(1,-1)

    input_array_scaled = scaler.transform(input_array)

    prediction = model.predict(input_array_scaled)

    st.subheader('Cell cluster prediction')
    st.write('The cell is:')

    if prediction[0]==0:
        st.write("<span class= 'diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class= 'diagnosis malicious'>Malicious</span>", unsafe_allow_html=True)

    st.write('The probability of being benign is: ', model.predict_proba(input_array_scaled)[0][0])
    st.write('The probability of being malicious is: ', model.predict_proba(input_array_scaled)[0][1])

    st.write("<span class = 'warning'>This app is for educational purposes only and should not be used for real medical decisions. Please consult a healthcare professional for diagnosis or treatment.</span>", unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title = 'Breast Cancer Predictor',
        page_icon = ':female-doctor:',
        layout = 'wide',
        initial_sidebar_state = 'expanded'
    )

    with open('Assets/style.css') as f:
        st.markdown('<style>{}<style>'.format(f.read()), unsafe_allow_html=True)

    input_data = add_sidebar()
    

    with st.container():
        st.title('Breast Cancer Predictor')
        st.write('This app is built using the Breast Cancer Wisconsin (Diagnostic) Data Set and leverages a logistic regression model to predict whether a tumor is benign or malignant based on various cell nuclei measurements. Users can input measurements through the sidebar, and the app will provide a prediction along with the probabilities of being benign or malignant.')

    col1, col2 = st.columns([4,1])

    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)

    with col2:
        add_predictions(input_data)



if __name__ == '__main__':
    main()