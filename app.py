import streamlit as st
import pickle
from sklearn.datasets import load_iris

def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def main():
    st.title('Iris Flower Classification Art Center')
    st.sidebar.header('Input Parameters')

    # Load the model
    model_path = 'iris_classifier.pkl'
    model = load_model(model_path)

    # Load the iris dataset
    iris = load_iris()

    # Add input widgets
    sepal_length = st.sidebar.slider('Sepal Length', 4.0, 8.0, 5.0)
    sepal_width = st.sidebar.slider('Sepal Width', 2.0, 4.5, 3.0)
    petal_length = st.sidebar.slider('Petal Length', 1.0, 7.0, 4.0)
    petal_width = st.sidebar.slider('Petal Width', 0.1, 2.5, 1.0)

    # Function to predict iris species
    def predict_species(sepal_length, sepal_width, petal_length, petal_width):
        # Make prediction
        prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
        return prediction[0]

    # Button to trigger prediction
    if st.sidebar.button('Predict'):
        species_prediction = predict_species(sepal_length, sepal_width, petal_length, petal_width)
        species_name = iris.target_names[species_prediction]

        # Display artwork based on predicted species
        if species_name == 'setosa':
            st.image('setosa.jpg', caption='Iris Setosa', use_column_width=True)
        elif species_name == 'versicolor':
            st.image('versicolor.jpg', caption='Iris Versicolor', use_column_width=True)
        elif species_name == 'virginica':
            st.image('virginica.jpg', caption='Iris Virginica', use_column_width=True)

if __name__ == '__main__':
    main()