# Airbnb Vienna — NLP & Price Prediction

Analyzing 612,000+ guest reviews from Vienna Airbnb listings using 
NLP techniques — and building ML models to predict listing prices 
based on property features, location, and guest sentiment.

## Overview

Vienna hosts thousands of Airbnb listings with reviews in dozens of 
languages. This project combines natural language processing and 
supervised machine learning to understand what guests actually care 
about — and what drives listing prices.

**Data:** Inside Airbnb (insideairbnb.com) — listings + reviews, Vienna, Austria (Sep 2025)  
**Context:** Group project — SCH-MGMT 661, Isenberg School of Management

## What's inside

**Data & preprocessing**
- EDA on 14,123 listings and 612,430 reviews across 79 variables
- Language detection — filtered to English-language reviews
- Text normalization: tokenization, stopword removal, lemmatization
- Multicollinearity analysis and feature reduction (23 → final clean set)

**NLP pipeline**
- Bag-of-Words (BoW) and TF-IDF vectorization
- Bigram and trigram extraction (top terms: "public transport", "walking distance")
- Topic modeling: LDA (BoW), NMF (TF-IDF), LSA (TF-IDF) — 4 topics identified
- Sentiment analysis: DistilBERT (avg. sentiment score per listing)

**Price prediction models**
- Ridge Regression (sklearn)
- Neural Network: 3 hidden layers, ReLU, dropout 0.2, early stopping (Keras)
- Log transformation of target variable to handle price skew

## Results

**Topic modeling — NMF produced most distinct topics:**

| Topic | Label |
|---|---|
| Topic 1 | Location & Comfort |
| Topic 2 | Host Quality & Service |
| Topic 3 | Customer Satisfaction |
| Topic 4 | Transport Accessibility |

Location & Comfort dominated at ~40% of overall topic distribution.

**Price prediction:**

| Model | RMSE (Log) | R² | RMSE ($) |
|---|---|---|---|
| Ridge Regression | 0.398 | 0.513 | $231.92 |
| Neural Network | 0.363 | 0.594 | $216.19 |

Neural Network outperformed Ridge on both metrics. Top price predictors: 
bedrooms, estimated occupancy, accommodates, and Innere Stadt location.

## Key findings

- Listing capacity (bedrooms, accommodates) is the single strongest 
  driver of price
- Innere Stadt commands a significant price premium over all other districts
- Sentiment scores correlated 0.51 with review ratings — but had limited 
  direct impact on price
- Both models underpredict high-priced listings (>$300) — unexplained 
  variance likely driven by interior quality, host reputation, and 
  seasonal demand

## Tech stack

Python · scikit-learn · Keras · pandas · NLTK · DistilBERT (HuggingFace) 
· matplotlib · seaborn · TF-IDF · CountVectorizer

## My contribution

- Cleaned and preprocessed large-scale review dataset (612K+ rows)
- Built full NLP pipeline: tokenization, vectorization (BoW, TF-IDF, n-grams)
- Conducted topic modeling using LDA, NMF, and LSA
- Performed DistilBERT sentiment analysis and assigned scores per listing
- Applied permutation importance to neural network to identify top features

## Authors

Olga Nureeva · Avanthikaa Kumaraguru · Vivek Vikram Singh · Alexander McDonough  
MS Business Analytics, Isenberg School of Management, UMass Amherst  
Course: SCH-MGMT 661

[LinkedIn](https://www.linkedin.com/in/olga-nureeva) · [GitHub Portfolio](https://github.com/onureeva)
