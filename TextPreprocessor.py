import spacy
import numpy as np
import pandas as pd

class TextPreprocessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def preprocess(self, text):
        """Preprocess single text."""
        text = text.lower()
        doc = self.nlp(text)
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        return ' '.join(tokens)

    def preprocess_pairs(self, text_pairs):
        """Preprocess list of text pairs."""
        processed_pairs = []
        for text1, text2 in text_pairs:
            processed_pairs.append((
                self.preprocess(text1),
                self.preprocess(text2)
            ))
        return processed_pairs