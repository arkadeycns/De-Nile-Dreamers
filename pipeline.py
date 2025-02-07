import joblib
from FeatureExtractor import FeatureExtractor
from DataLoader import DataLoader
from TextPreprocessor import TextPreprocessor

input_text_1 = input()
input_text_2 = input()
input_text_pair = [(input_text_1, input_text_2)]
    
pre_processor = TextPreprocessor()
input_processed_pair = pre_processor.preprocess_pairs(input_text_pair)
input_feature_extractor = FeatureExtractor()
input_features = input_feature_extractor.extract_features(input_text_pair)

loaded_model = joblib.load('saved_models\svm.pkl')
print("Model loaded successfully!")


predictions = loaded_model.predict(input_features)
print("Predictions:", predictions)
