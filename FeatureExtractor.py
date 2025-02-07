
from TextPreprocessor import TextPreprocessor
from sentence_transformers import SentenceTransformer
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine, cityblock

class FeatureExtractor:
    def __init__(self, model_name="paraphrase-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def get_embedding(self, text):
        return self.model.encode(text, convert_to_numpy=True)

    def get_features(self, text1, text2):
        vec1 = self.get_embedding(text1)
        vec2 = self.get_embedding(text2)
        
        return {
            "cosine_similarity": 1 - cosine(vec1, vec2),
            "manhattan_distance": cityblock(vec1, vec2),
        }

    def extract_and_save_features(self, text_pairs, output_path):
        """Extract features and save them to a CSV file."""
        if os.path.exists(output_path):
            print(f"Loading existing features from {output_path}")
            return pd.read_csv(output_path)

        print(f"Extracting new features and saving to {output_path}")
        features = []
        for text1, text2 in text_pairs:
            feature_dict = self.get_features(text1, text2)
            features.append(feature_dict)
        
        features_df = pd.DataFrame(features)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        features_df.to_csv(output_path, index=False)
        
        return features_df

    def prepare_features(self, text_pairs, output_path):
        """Prepare and save features if they don't exist."""
        preprocessor = TextPreprocessor()
        
        
        processed_pairs = preprocessor.preprocess_pairs(text_pairs)
        return self.extract_and_save_features(processed_pairs, output_path)
    
    def extract_features(self, text_pairs):
        features = []
        for text1, text2 in text_pairs:
            feature_dict = self.get_features(text1, text2)
            features.append(feature_dict)
        
        features_df = pd.DataFrame(features)
        return features_df
