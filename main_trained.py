import joblib
from FeatureExtractor import FeatureExtractor
from DataLoader import DataLoader
from TextPreprocessor import TextPreprocessor
import sbert_test

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
        result = f"The sentences are not paraphrases."
    if predictions[0] == 1:
        result = f"The sentences are paraphrases."
    
    similarity_score = sbert_test.get_similarity(sentence1, sentence2)[0]
    similarity_score = f"{similarity_score * 100:.2f}%"
    print("Returning:", result, similarity_score)
    return result, similarity_score

# if __name__ == "__main__":
#     main()