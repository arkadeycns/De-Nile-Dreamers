import streamlit as st
import main_  # Importing processing module

def main_app():
    st.title("Sentence Semantic Similarity Checker")

    # Dropdown for selecting the model
    model_options = [
        "lightgbm", 
        "xgboost", 
        "random_forest", 
        "logistic_regression", 
        "catboost", 
        "naive_bayes", 
        "svm", 
        "knn", 
        "sbert"
    ]
    selected_model = st.selectbox("Select a model for processing:", model_options)

    # User Inputs
    sentence1 = st.text_area("Enter first sentence:")
    sentence2 = st.text_area("Enter second sentence:")

    if st.button("Check Similarity"):
        if sentence1 and sentence2:
            result = main_.main(sentence1, sentence2, selected_model)  # Sending to main.py
            st.write("### Result:")
            st.success(result)
        else:
            st.warning("Please enter both sentences.")

if __name__ == "__main__":
    main_app()
