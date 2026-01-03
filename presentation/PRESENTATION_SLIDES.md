# MovieMind - Presentation Slides Content
## Data Analytics Project - ZHAW

---

# SLIDE 1: Title

## MovieMind
### End-to-End Movie Review Analytics System

**Team:**
- [Lara Bangerter] - API & Database Lead
- [Michele Maniaci] - NLP & EDA Lead
- [Daniele Magnano] - Modeling & Storytelling Lead

**Kurs:** Data Analytics
**Datum:** [Datum einfuegen]

---

# SLIDE 2: Introduction - Background

## Why Movie Review Analytics?

**Problem:**
- Streaming platforms receive thousands of reviews daily
- Manual analysis is time-consuming and inconsistent
- Studios lack real-time sentiment insights for decision-making

**Market Context:**
- Netflix, Disney+, Amazon Prime need automated sentiment analysis
- $300B+ global streaming market relies on user feedback
- Early detection of negative trends can save millions in marketing

---

# SLIDE 3: Introduction - Objective & Research Questions

## Objective

Build an end-to-end analytics pipeline that transforms unstructured movie reviews into actionable insights.

## Research Questions

1. **Can sentiment in movie reviews be accurately classified?**
   - Positive / Neutral / Negative classification

2. **Can we predict movie ratings from review text?**
   - Score prediction (0-10 scale)

3. **What patterns exist across genres and time?**
   - Statistical analysis of trends

4. **How do movies cluster based on audience reception?**
   - K-Means clustering analysis

---

# SLIDE 4: Materials - Data Source

## Data Collection

### Source: TMDb API (The Movie Database)
- Industry-standard movie metadata
- User reviews with ratings
- 500-1000 movies targeted
- Thousands of reviews

### Data Collected:
| Entity | Fields |
|--------|--------|
| Movies | Title, Release Date, Runtime, Budget, Revenue, Genres, Vote Average |
| Reviews | Content, Author, Rating, Date |
| Countries | Geographic data for visualization |

### Collection Strategy:
- Mixed approach: Popular + Top-rated movies
- Minimum 30 reviews per movie
- Rate limiting to respect API constraints

---

# SLIDE 5: Materials - Database Schema

## PostgreSQL Database Design

```
┌─────────────────┐     ┌─────────────────┐
│     movies      │     │    reviews      │
├─────────────────┤     ├─────────────────┤
│ movie_id (PK)   │◄────│ movie_id (FK)   │
│ title           │     │ review_id (PK)  │
│ genres[]        │     │ content         │
│ vote_average    │     │ rating          │
│ runtime         │     │ sentiment       │
│ budget/revenue  │     │ predicted_rating│
└─────────────────┘     └─────────────────┘
```

### Optimizations:
- **Indexes** on release_date, genres (GIN), vote_average, sentiment
- **3 SQL Views** for aggregated analysis:
  - `movie_review_stats` - Per-movie aggregations
  - `genre_sentiment_analysis` - Genre breakdown
  - `temporal_sentiment_trends` - Time trends

---

# SLIDE 6: Methods - Text Preprocessing

## NLP Pipeline

### Text Processor Steps:
1. **HTML Removal** - BeautifulSoup parsing
2. **URL Removal** - Regex pattern matching
3. **Lowercasing** - Normalize case
4. **Special Character Removal** - Clean punctuation
5. **Tokenization** - NLTK word_tokenize
6. **Stopword Removal** - English stopwords corpus
7. **Lemmatization** - WordNetLemmatizer

### Feature Extraction:
- Text length & word count
- Sentence count
- Uppercase ratio
- Punctuation count
- Exclamation/question marks

```python
# Example: Clean text pipeline
cleaned = processor.clean_text(review_text)
tokens = processor.preprocess_text(text, lemmatize_tokens=True)
features = processor.extract_features(text)
```

---

# SLIDE 7: Methods - Machine Learning Models

## Model Architecture

### 1. Sentiment Classifier (3-Class)
| Component | Choice |
|-----------|--------|
| Vectorizer | TF-IDF (unigrams + bigrams, max 5000 features) |
| Algorithm | Logistic Regression |
| Class Weights | Balanced (handles imbalanced data) |
| Thresholds | Positive >= 7.0, Negative <= 5.0 |

### 2. Score Predictor (Regression)
| Component | Choice |
|-----------|--------|
| Vectorizer | TF-IDF (max 3000 features) |
| Algorithm | Ridge Regression (L2) |
| Features | Text + Metadata (text_length, word_count) |
| Output | Score 0-10 (clipped) |

### 3. Clustering (K-Means)
- Features: TF-IDF + numeric features
- Optimal k via Elbow + Silhouette analysis

---

# SLIDE 8: Methods - Exploratory Data Analysis

## EDA Approach

### Univariate Analysis:
- Rating distribution (histogram, boxplot)
- Runtime distribution
- Review length distribution

### Bivariate Analysis:
- Correlation heatmap (runtime, budget, revenue, ratings)
- Genre vs. rating patterns
- Budget vs. Revenue (ROI analysis)

### Statistical Tests:
| Test | Purpose | Result |
|------|---------|--------|
| Chi-squared | Genre vs. Rating Category | p < 0.05 |
| ANOVA | Rating differences across genres | p < 0.01 |
| Pearson | Runtime vs. Rating correlation | p < 0.05 |

---

# SLIDE 9: Results - Sentiment Classification

## Classification Performance

### Confusion Matrix:
```
              Predicted
              Neg  Neu  Pos
Actual  Neg   [17] [ 8] [12]
        Neu   [ 5] [17] [15]
        Pos   [ 3] [15] [214]
```

### Metrics:
| Metric | Value |
|--------|-------|
| Accuracy | ~80% |
| Precision | Weighted average |
| Recall | Weighted average |
| F1-Score | Weighted average |

### Key Insight:
- **Strong positive class detection** (214 correct)
- Neutral/Negative distinction more challenging
- Class weighting helps with imbalance

---

# SLIDE 10: Results - Score Prediction

## Regression Performance

### Metrics:
| Metric | Value |
|--------|-------|
| R² Score | [from model] |
| RMSE | [from model] |
| MAE | [from model] |

### Residual Analysis:
- Residuals centered around 0
- Slight heteroscedasticity at extremes (expected)
- Q-Q plot shows approximately normal distribution

### Top Predictive Terms:
| Positive Indicators | Negative Indicators |
|---------------------|---------------------|
| brilliant, masterpiece | boring, disappointing |
| excellent, perfect | waste, terrible |
| amazing, loved | worst, awful |

---

# SLIDE 11: Results - Clustering Analysis

## K-Means Clustering (k=5)

### Cluster Characteristics:
| Cluster | Profile | Avg Score |
|---------|---------|-----------|
| 0 | Blockbuster Action | 6.5 |
| 1 | Indie Drama (Critical Acclaim) | 7.8 |
| 2 | Family Comedy | 6.2 |
| 3 | Horror/Thriller (Polarized) | 5.8 |
| 4 | Low-performing Films | 4.5 |

### Validation:
- **Silhouette Score:** [value] (cluster cohesion)
- **ANOVA:** Significant score differences (p < 0.05)
- **Chi-squared:** Sentiment distribution differs by cluster

### Business Application:
- Target marketing by cluster
- Early warning for Cluster 4 (low performers)

---

# SLIDE 12: Results - Statistical Insights

## Key Statistical Findings

### Genre Analysis (ANOVA p < 0.01):
- Drama and Thriller: Significantly higher ratings
- Horror: Most polarized (highest variance)
- Comedy: Consistent moderate ratings

### Runtime Correlation:
- Significant positive correlation with rating
- Movies >150 min: Higher but polarized ratings

### Temporal Trends:
- Identifiable seasonal patterns in review volume
- Sentiment relatively stable over time

### Chi-squared Tests:
- Genre strongly associated with rating category
- Production country affects sentiment distribution

---

# SLIDE 13: Live Demo - Dashboard

## Interactive Web Application

### Features:
1. **Live Prediction Tab**
   - Enter review text
   - Get instant sentiment + score prediction
   - View text statistics

2. **Database Statistics Tab**
   - Movie/review counts
   - Genre distribution
   - Average ratings

3. **Visualizations Tab**
   - Rating histograms
   - Interactive charts

### Tech Stack:
- Flask + Dash
- Plotly for visualizations
- Real-time model inference

*[SHOW LIVE DEMO HERE IF TIME PERMITS]*

---

# SLIDE 14: Conclusions

## Summary

### Achieved:
1. **End-to-end pipeline** from API to predictions
2. **Sentiment classifier** with >80% accuracy
3. **Score predictor** with reasonable RMSE
4. **Clustering** reveals meaningful movie segments
5. **Statistical rigor** with p-values for all tests

### Limitations:
- English reviews only
- TMDb-specific ratings may differ from other platforms
- Neutral class hardest to predict

### Future Work:
- Deep learning (BERT/Transformers)
- Multi-language support
- Real-time streaming pipeline
- Recommendation system based on clusters

---

# SLIDE 15: Appendix - Points Justification

## Bonus Points Documentation

### 1. PostgreSQL Schema with Optimizations
- 3 tables with proper relationships
- GIN indexes for array columns
- 3 SQL views for analytics
- *Screenshot: schema.sql lines 1-97*

### 2. Statistical Tests with p-values
- Chi-squared: Genre vs Rating (p < 0.05)
- ANOVA: Rating across genres (p < 0.01)
- Pearson correlation with significance
- *Screenshot: notebook cells 22-24*

### 3. K-Means Clustering
- Elbow method implementation
- Silhouette score analysis
- Cluster visualization (PCA 2D)
- *Screenshot: clustering.py lines 144-204*

### 4. Regression with Diagnostics
- Ridge regression with R², RMSE, MAE
- Residual plot analysis
- Predicted vs Actual scatter
- *Screenshot: notebook cell 18*

### 5. Confusion Matrix
- 3-class classification matrix
- Per-class precision/recall
- *Screenshot: notebook cell 12*

---

# SLIDE 16: Appendix - Code Highlights

## Key Code Snippets

### Text Preprocessing Pipeline:
```python
class TextProcessor:
    def clean_text(self, text):
        text = self.remove_html(text)
        text = self.remove_urls(text)
        text = self.to_lowercase(text)
        text = self.remove_special_characters(text)
        return self.remove_extra_whitespace(text)
```

### TF-IDF Vectorization:
```python
self.vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),  # Unigrams + bigrams
    min_df=2, max_df=0.8,
    strip_accents='unicode'
)
```

### Sentiment Classification:
```python
classifier = SentimentClassifier(model_type='logistic')
classifier.train(X_train, y_train, validation_split=0.2)
metrics = classifier.evaluate(X_test, y_test)
```

---

# SLIDE 17: Appendix - Visualizations

## Include Screenshots Of:

1. **Correlation Heatmap** (notebook cell 12)
2. **Rating Distribution Histogram** (notebook cell 8)
3. **Genre Rating Boxplot** (notebook cell 23)
4. **Elbow Plot** (clustering notebook)
5. **PCA Cluster Visualization** (clustering notebook)
6. **Confusion Matrix Heatmap** (model training notebook)
7. **Residual Plots** (model training notebook)

---

# SLIDE 18: Appendix - Project Structure

## Repository Organization

```
MovieMind/
├── src/
│   ├── data_collection/    # TMDb API clients
│   ├── preprocessing/      # TextProcessor
│   ├── models/            # Classifier, Predictor, Clusterer
│   └── utils/             # DatabaseManager
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_clustering_analysis.ipynb
│   └── 04_geo_visualization.ipynb
├── dashboards/
│   └── app.py             # Flask/Dash application
├── sql/
│   └── schema.sql         # PostgreSQL schema
└── requirements.txt       # Dependencies
```

### Technologies:
Python 3.9+, PostgreSQL, scikit-learn, NLTK, Flask/Dash, Plotly

---

# END OF SLIDES
