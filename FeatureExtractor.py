
from TextPreprocessor import TextPreprocessor
from sentence_transformers import SentenceTransformer
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine, cityblock
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from scipy.spatial.distance import cityblock
import spacy
from collections import Counter

class FeatureExtractor:
    def __init__(self, model_name="paraphrase-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.nlp = spacy.load("en_core_web_sm")  # Loading spaCy model for NER

    def get_embedding(self, text):
        return self.model.encode(text, convert_to_numpy=True)

    def get_features(self, text1, text2):
        vec1 = self.get_embedding(text1)
        vec2 = self.get_embedding(text2)

        # Cosine Similarity and Manhattan Distance
        cosine_sim = 1 - cosine(vec1, vec2)
        manhattan_dist = cityblock(vec1, vec2)
        
        # Word Count Similarity
        word_count_sim = self.word_count_similarity(text1, text2)
        
        # NER-based similarity
        ner_sim = self.ner_similarity(text1, text2)
        
        # synset_sim = self.synset_similarity(text1, text2)

        return {
            "cosine_similarity": cosine_sim,
            "manhattan_distance": manhattan_dist,
            "word_count_similarity": word_count_sim,
            "ner_similarity": ner_sim
            # "synset_sim":synset_sim

        }
    

    # def synset_similarity(self, text1, text2):
    #     words1 = text1.split()
    #     words2 = text2.split()
        
    #     total_score = 0
    #     count = 0
        
    #     for word1 in words1:
    #         synsets1 = wordnet.synsets(word1)
    #         if not synsets1:
    #             continue
            
    #         for word2 in words2:
    #             synsets2 = wordnet.synsets(word2)
    #             if not synsets2:
    #                 continue
                
    #             # Compute max Wu-Palmer similarity for the best synset pair
    #             max_sim = max(s1.wup_similarity(s2) or 0 for s1 in synsets1 for s2 in synsets2)
    #             total_score += max_sim
    #             count += 1
        
    #     return total_score / count if count > 0 else 0  # Avoid division by zero


    def word_count_similarity(self, text1, text2):
        words1 = set(text1.split())
        words2 = set(text2.split())
        common_words = words1.intersection(words2)
        if (len(words1) + len(words2) - len(common_words)) == 0:
            return 0 
        return len(common_words) / (len(words1) + len(words2) - len(common_words))  # Jaccard similarity

    def ner_similarity(self, text1, text2):
        doc1 = self.nlp(text1)
        doc2 = self.nlp(text2)

        # Extract entities
        ents1 = set([ent.text for ent in doc1.ents])
        ents2 = set([ent.text for ent in doc2.ents])

        # Separate out numerical entities (like numbers, dates, etc.)
        numerical_ents1 = {ent.text for ent in doc1.ents if ent.label_ in ['CARDINAL', 'MONEY', 'QUANTITY', 'PERCENT', 'DATE']}
        numerical_ents2 = {ent.text for ent in doc2.ents if ent.label_ in ['CARDINAL', 'MONEY', 'QUANTITY', 'PERCENT', 'DATE']}
        
        # Common entities
        common_ents = ents1.intersection(ents2)
        
        # Common numerical entities
        common_numerical_ents = numerical_ents1.intersection(numerical_ents2)

        # Increase weight for common numerical entities
        numerical_weight = 4  
        ner_similarity_score = len(common_ents) + (len(common_numerical_ents) * numerical_weight)

        return ner_similarity_score



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
