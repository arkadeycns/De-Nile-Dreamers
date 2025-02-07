import joblib
from FeatureExtractor import FeatureExtractor
from DataLoader import DataLoader
from TextPreprocessor import TextPreprocessor

def main(sentence1, sentence2, model_name):
    input_text_1 = sentence1
    input_text_2 = sentence2
    input_text_pair = [(input_text_1, input_text_2)]
        
    pre_processor = TextPreprocessor()
    input_processed_pair = pre_processor.preprocess_pairs(input_text_pair)
    input_feature_extractor = FeatureExtractor()
    input_features = input_feature_extractor.extract_features(input_text_pair)
    print(input_features)
    loaded_model = joblib.load(f"saved_models/{model_name}.pkl")
    print("Model loaded successfully!")


    predictions = loaded_model.predict(input_features)
    if predictions[0] == 0: 
        return f"The sentences are not paraphrases."
    if predictions[0] == 1: 
        return f"The sentences are paraphrases."

# if __name__ == "__main__":
#     main()