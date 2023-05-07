from sklearn.datasets import load_iris
import streamlit as st
import pickle
iris = load_iris()
# Load the saved model
with open('Logistic Regression.pkl', 'rb') as f:
    clf = pickle.load(f)

# Create the user interface
st.title("Iris Classification App")
sepal_length = st.slider("Sepal length", 0.0, 10.0, 5.0)
sepal_width = st.slider("Sepal width", 0.0, 10.0, 5.0)
petal_length = st.slider("Petal length", 0.0, 10.0, 5.0)
petal_width = st.slider("Petal width", 0.0, 10.0, 5.0)

# Make a prediction
prediction = clf.predict([[sepal_length, sepal_width, petal_length, petal_width]])

# Display the prediction to the user
species = iris.target_names[prediction[0]]
st.header(f"Prediction: {species}")
