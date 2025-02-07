import streamlit as st
import pandas as pd
import main_  # Importing processing module

def main_app():
    st.title("Sentence Semantic Similarity Checker")

    # Initialize session state for history if not already created
    if "history" not in st.session_state:
        st.session_state.history = []

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
            
            # Save to history with an incremental index
            st.session_state.history.append(
                {"Index": len(st.session_state.history) + 1,  # Assign index
                 "Sentence1": sentence1, "Sentence2": sentence2, 
                 "Model": selected_model, "Result": result}
            )
            
            st.write("### Result:")
            st.success(result)
        else:
            st.warning("Please enter both sentences.")
    
    # Display history
    if st.session_state.history:
        st.write("### History:")
        
        # Convert history to DataFrame and sort by Index (latest on top)
        df = pd.DataFrame(st.session_state.history)
        df = df.sort_values(by="Index", ascending=False).reset_index(drop=True)
        
        # Display dataframe with Index as a column
        st.dataframe(df.set_index("Index"), use_container_width=True)

if __name__ == "__main__":
    main_app()
