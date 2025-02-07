# Semantic Textual Similarity Benchmark Model

## Overview
This repository aims to develop a benchmark model for **Semantic Textual Similarity (STS)** problems. The goal is to measure the degree of similarity between two text inputs, which is an important task in Natural Language Processing (NLP). To address this problem, we employ two different solution methods, each designed to optimize the performance and accuracy of the model:

We are using the Microsoft Paraphrase Corpus as our dataset.

1. **Binary Classification Techniques**: Utilizing handcrafted features derived from vector embeddings.
2. **Pretrained Sentence Transformer Models**: Leveraging state-of-the-art models to achieve maximum accuracy in STS tasks.

## Problem Description
Semantic Textual Similarity is a crucial task for a variety of NLP applications, including:

- Information retrieval - Finding similar entries, blocking a particular type of comment under a YouTube video.
- Text summarization - In generating summaries, STS helps ensure that the summary retains the core meaning of the original text, even if the wording changes.
- Paraphrase detection - Quora may use it to better categorize the questions.
- Question answering - Various questions can be categorized to a single common to give an appropriate answer.

The goal of STS is to predict how similar two pieces of text are, based on their meanings rather than their surface-level similarity. The challenge is to design models that capture deep semantic relationships, even when the surface forms of the texts differ significantly.

## Solution Methods

### 1. Gradient Boosting Techniques
In the first approach, we explore the use of **Gradient Boosting algorithms** to predict semantic similarity. These models rely on features extracted from vector embeddings of the input texts. The features are designed to capture various semantic aspects of the texts, and the Gradient Boosting model leverages these features to make accurate similarity predictions.

#### Key Steps:
- **Feature Engineering**: We extract a variety of features from the vector embeddings, such as cosine similarity, word overlap, and sentence structure metrics.
- **Model Training**: A gradient boosting algorithm (e.g., XGBoost, LightGBM) is trained on these features to predict the similarity score between sentence pairs.
- **Model Evaluation**: We evaluate the model using standard STS evaluation metrics.
  
### 2. Pretrained Sentence Transformer Models
For the second approach, we leverage **Pretrained Sentence Transformers** to directly generate embeddings for sentence pairs. These models, like BERT and its variants, have been pretrained on vast corpora and fine-tuned on STS tasks to capture semantic relationships between text pairs more effectively.

#### Key Steps:
- **Sentence Embedding Generation**: We use a pretrained transformer model (`sentence-transformers/paraphrase-MiniLM-L6-v2`) to generate embeddings for the input sentences.
- **Similarity Scoring**: The cosine similarity between the embeddings of two sentences is calculated to predict the degree of similarity.

### Key Observations
- Just cosine similarity as just this factor was not sufficient, so we tried to get more features to feed to ML algorithms
  - Refering to msr dataset, row 7 and 26 are completely contradictory... 'She is beautiful' vs 'She is beautiful and intelligent', 0 or 1?
  - We implemented NER specifically to bring special focus to numbers... If our entire sentence is same other than just a numerical value, what should our program return, 0 or 1? So based, on the training data, our model with adjust this variation.
- Levenshtein distance is useless given that we need to find 'semantic' similarity.
- We could use bert, siamesse, etc. for similarity calculation but we are using sbert (didn't use tf-idf or word2vec).
- Tf-idf is inferior to word2vec or any above embedding.
- Neural network and siamese etc poor due to small corpus, so is sbert (but aboe, we are using pre-trained model)
- NLTK removed bcz spacy is concise, nltk(a toolkit, but more flexibiolity)  need to download corpus, spacy has inbuilt tokeniser , lemmatizer(pre trained)
- LSTM -> overfitting, because small corpus, so overfitting even on early stopping, moreover time consuming


### Comparison of Methods
We compare the performance of both approaches using standard STS benchmarks to evaluate which method achieves better accuracy. The pretrained transformer-based model typically performs better on challenging semantic tasks, while the gradient boosting approach can offer competitive results when feature engineering is optimized.

## Setup Instructions

### Requirements
To run this project, the required dependencies have been listed in the requirements.txt file.

You can create a virtual env and install the required libraries using the following command:

```bash
pip install -r requirements.txt
