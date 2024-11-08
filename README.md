# Breast Cancer Predictor
This Streamlit app utilizes the Breast Cancer Wisconsin (Diagnostic) Data Set to predict the likelihood of a breast tumor being Benign or Malignant. It uses a logistic regression model trained on cell nuclei measurements to provide predictions and probability estimates based on user inputs.

### Features
- User Input: Users can input various cell nuclei features (e.g., radius, texture, perimeter, area) through an interactive sidebar.
- Prediction Model: The app employs a logistic regression model to predict whether a tumor is benign or malignant and provides probability scores for each class.
- Radar Chart Visualization: The app includes a radar chart to visualize mean, standard error, and worst values of cell nuclei features, allowing users to understand their input data distribution.
- Educational Notice: This app is intended solely for educational purposes and should not be used as a substitute for medical diagnosis. Consult a healthcare professional for real medical decisions.

### Technology Stack
- Backend: Python, with a logistic regression model trained on scikit-learn.
- Frontend: Streamlit for interactive UI and Plotly for radar chart visualization.
- Data Handling: pandas for data manipulation, with MinMaxScaler and StandardScaler for data scaling.

### Disclaimer
This app is for educational purposes only and is not intended for use in real medical decisions. Please consult a qualified healthcare professional for any medical concerns.
