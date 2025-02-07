import numpy as np
import pandas as pd
import spacy
import os
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
import joblib

from FeatureExtractor import FeatureExtractor
from DataLoader import DataLoader
from TextPreprocessor import TextPreprocessor

class TextSimilarityPipeline:
    def __init__(self, model_type='lightgbm', model_params=None):
        self.model_type = model_type
        self.model_params = model_params or {}
        self.model = self._create_model()

    def _create_model(self):
        """Create model based on type and parameters."""
        models = {
            'lightgbm': lgb.LGBMClassifier,
            'xgboost': xgb.XGBClassifier,
            'random_forest': RandomForestClassifier,
            'svm': SVC,
            'logistic_regression': LogisticRegression,
            'catboost': cb.CatBoostClassifier
        }
        
        if self.model_type not in models:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
        model_class = models[self.model_type]
        return model_class(**self.model_params)

    def train(self, features, labels):
        """Train the model using pre-extracted features."""
        self.model.fit(features, labels)

    def predict(self, features):
        """Make predictions using pre-extracted features."""
        return self.model.predict(features)

    def evaluate(self, features, labels):
        """Evaluate model using pre-extracted features."""
        predictions = self.predict(features)
        return accuracy_score(labels, predictions)


def main():
    # Create output directory for features
    features_dir = 'features'
    os.makedirs(features_dir, exist_ok=True)

    # Load data
    data_loader = DataLoader()
    train_pairs, train_labels = data_loader.load_txt_data('data/msr_paraphrase_train.txt')
    test_pairs, test_labels = data_loader.load_txt_data('data/msr_paraphrase_test.txt')

    if train_pairs is None or test_pairs is None:
        print("Error loading data. Please check file paths and format.")
        return
    feature_extractor = FeatureExtractor()
    # Extract or load features
    train_features = feature_extractor.prepare_features(train_pairs, os.path.join(features_dir, 'train_features.csv'))
    test_features = feature_extractor.prepare_features(test_pairs, os.path.join(features_dir, 'test_features.csv'))

    # Model configurations
    model_configs = [
        {
            'model_type': 'lightgbm',
            
        },
        {
            'model_type': 'xgboost',
            
        },
        {
            'model_type': 'svm',
            
        }
    ]

    # Train and evaluate different models
    for config in model_configs:
        print(f"\nTraining {config['model_type']} model...")
        pipeline = TextSimilarityPipeline(
            model_type=config['model_type'],
            
        )
        pipeline.train(train_features, train_labels)
        accuracy = pipeline.evaluate(test_features, test_labels)
        print(f"{config['model_type']} Test Accuracy: {accuracy:.4f}")
        joblib.dump(pipeline.model, f"saved_models\{config['model_type']}.pkl")

if __name__ == "__main__":
    main()