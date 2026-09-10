# Airbnb Vienna — NLP & Price Prediction

Using structured Airbnb listing data and guest review text to understand what drives price — and how review themes differ across market segments in Vienna.

## Interactive Tableau Story

Explore the findings and use the interactive Listing Strategy Calculator:

[View the interactive Tableau Story](https://public.tableau.com/app/profile/olga.nureeva/viz/ViennaAirbnbFromReviewstoStrategy/Story1)

## Overview

This project combines natural language processing and supervised machine learning to answer two related questions:

1. How much predictive value does guest-review text add beyond traditional listing characteristics?
2. How does the content of guest reviews differ across Airbnb price segments?

The analysis uses Airbnb listings and reviews from Vienna, Austria. Review text is transformed into sentiment and topic features, then combined with structural listing characteristics in a Ridge Regression model.

**Data:** Inside Airbnb — Vienna, Austria, snapshot of September 14, 2025

**Origin:** Originally developed as a group project for SCH-MGMT 661 at Isenberg School of Management; substantially reworked, corrected, and extended independently for this portfolio.

## Data

The project uses two Inside Airbnb datasets:

* `listings.csv.gz` — property characteristics, location, capacity, amenities, and price
* `reviews.csv.gz` — guest review text

After language filtering and preprocessing, the review dataset contains approximately **319,000 English-language reviews**.

The final modeling dataset contains **8,138 unique listings**, with each listing paired with the concatenated text of its 10 most recent reviews.

## NLP Pipeline

The text analysis includes:

* English-language review filtering
* Text cleaning and normalization
* Bag-of-Words and TF-IDF vectorization
* Bigram and trigram analysis
* Topic modeling using:

  * LDA
  * NMF
  * LSA
* DistilBERT sentiment scoring
* Listing-level aggregation of sentiment and review-topic features

NMF produced the most interpretable topic structure.

### Final NMF Topics

1. **Walkability & Local Amenities**
2. **Property Condition**
3. **Transport & Accessibility**
4. **Host & Hospitality**

## Price Prediction

The final predictive analysis compares:

* **Median baseline**
* **Ridge Regression using structural listing features**
* **Ridge Regression using structural + NLP features**

The target variable is log-transformed price.

| Model                     | R² (log price) |        MAE | Median AE |
| ------------------------- | -------------: | ---------: | --------: |
| Median baseline           |              — |     €80.06 |    €30.00 |
| Structural features only  |          0.381 |     €66.83 |    €21.33 |
| Structural + NLP features |      **0.410** | **€65.84** |    €21.48 |

Adding NLP features increased R² by approximately **0.03** and reduced mean absolute error by approximately **€0.99**.

The improvement is real but modest: structural characteristics remain the main source of predictive power, while review text contributes additional information about market positioning and guest experience.

## What Drives Price?

The strongest Ridge coefficients include:

* Entire rental unit property type
* Number of guests accommodated
* Number of bedrooms
* Serviced apartment property type
* Innere Stadt location

Among the NLP features, **Transport & Accessibility** has one of the largest coefficients in magnitude.

Because the model is observational, these relationships should be interpreted as associations rather than causal effects.

## Price Segments & Review Themes

Listings were divided into five business-oriented price segments:

* **Budget:** up to €75
* **Standard:** €76–125
* **Premium:** €126–200
* **High-end:** €201–500
* **Luxury:** above €500

Review-topic emphasis changes noticeably across these segments.

Budget listings show above-average emphasis on **Transport & Accessibility**, while Premium and High-end listings show stronger emphasis on **Host & Hospitality**.

For example:

* Budget transport-topic index: **123.9**
* High-end transport-topic index: **57.3**
* High-end hospitality-topic index: **141.3**

An index of 100 represents the overall dataset average.

Dominant-topic analysis shows a similar pattern: Transport & Accessibility is the most frequent dominant topic among Budget listings, while hospitality becomes more prominent in higher-priced segments.

## Statistical Testing

I used the **Kruskal–Wallis test** to test whether review-topic distributions differed across price segments.

All four topics showed statistically significant differences after Holm correction for multiple testing:

| Topic                          | H statistic |
| ------------------------------ | ----------: |
| Walkability & Local Amenities  |      173.82 |
| Property Condition             |       55.45 |
| Transport & Accessibility      |      786.64 |
| Host & Hospitality             |      288.76 |

Holm-adjusted p-values were below 0.001 for all four topics.

Transport & Accessibility showed the strongest difference across price segments.

## Key Findings

* Structural listing characteristics explain substantially more price variation than review-derived NLP features.
* Guest-review text still adds incremental predictive information.
* Listing capacity and property type are among the strongest price predictors.
* Innere Stadt commands a significant price premium over every other district (€191 median vs. €118 for the next-highest district).
* Review content changes systematically across market segments.
* Transport-related content is much more prominent among lower-priced listings.
* Hospitality-related content becomes more prominent in Premium and High-end listings.
* Review text appears more useful for understanding positioning and guest experience than for dramatically improving price prediction.

## Methodological Note

An earlier version of the project merged listings with individual reviews before the train/test split. This duplicated listings across observations and allowed the same listing characteristics to appear in both training and test data, artificially inflating R² to approximately 0.62.

The pipeline was corrected to aggregate review text to **one row per listing before modeling**. The final out-of-sample R² is approximately **0.41**.

I kept this correction documented because identifying and fixing the leakage issue was an important part of the analytical process.

## Limitations

* The analysis covers one city and one snapshot in time.
* Airbnb prices may also be influenced by seasonality, events, host pricing strategy, and other market conditions not included here.
* Topic models simplify complex text into a small number of latent themes.
* The relationships identified are associative, not causal.
* TF-IDF and topic extraction were performed before the predictive train/test split. A stricter ML pipeline would fit these transformations on training data only.
* Review-based insights reflect only guests who left reviews and may not represent all stays.
* Some listing segments have small comparable samples, which can make topic indexes less stable.

## How to Run

* Python 3.10+, dependencies listed in `requirements.txt`.
* `listings.csv.gz` and `reviews.csv.gz` download directly from [Inside Airbnb](https://insideairbnb.com/get-the-data/) inside the notebooks — no manual download needed.
* Cached intermediate outputs are included in `data/`: `cleaned_reviews.csv.gz` (preprocessed review text) and `listing_sentiment.csv` (DistilBERT sentiment scores). Sentiment scoring is the only expensive step, so it's cached; the code that produced it is preserved and documented in `02_airbnb_analysis.ipynb`.
* Run `01_reviews_preprocessing.ipynb` first, then `02_airbnb_analysis.ipynb` top to bottom.

## Repository Structure

```text
airbnb-vienna-nlp-pricing/
│
├── data/
│   ├── cleaned_reviews.csv.gz
│   └── listing_sentiment.csv
│
├── notebooks/
│   ├── 01_reviews_preprocessing.ipynb
│   └── 02_airbnb_analysis.ipynb
│
├── requirements.txt
└── README.md
```

## Tech Stack

Python · scikit-learn · pandas · NLTK · DistilBERT (HuggingFace) · matplotlib · seaborn · TF-IDF · CountVectorizer

## Author

Olga Nureeva
MS Business Analytics, Isenberg School of Management, UMass Amherst

[LinkedIn](https://www.linkedin.com/in/olga-nureeva) · [GitHub Portfolio](https://github.com/onureeva)
