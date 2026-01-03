# MovieMind - Presentation Slides
## Data Analytics Project - ZHAW

---

# SLIDE 1: Title

## MovieMind
### End-to-End Movie Review Analytics System

**Team:**
- Lara Bangerter - API & Database Lead
- Michele Maniaci - NLP & EDA Lead
- Daniele Magnano - Modeling & Storytelling Lead

**Course:** Data Analytics - ZHAW
**Date:** January 2026

---

# SLIDE 2: Why Movie Review Analytics?

## Background & Problem

**The Challenge:**
- Streaming platforms receive thousands of reviews daily
- Manual analysis is time-consuming and inconsistent
- Studios lack real-time sentiment insights for decision-making

**Market Context:**
- Netflix, Disney+, Amazon Prime need automated sentiment analysis
- $300B+ global streaming market relies on user feedback
- Early detection of negative trends can save millions in marketing

**Our Solution:** An automated end-to-end analytics pipeline

---

# SLIDE 3: Objective & Research Questions

## Project Goal

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

# SLIDE 4: Data Collection

## TMDb API - The Movie Database

**Data Source:**
- Industry-standard movie metadata
- User reviews with ratings
- 500-1000 movies collected
- Thousands of reviews

**Data Collected:**
| Entity | Fields |
|--------|--------|
| Movies | Title, Release Date, Runtime, Budget, Revenue, Genres, Vote Average |
| Reviews | Content, Author, Rating, Date |

**Collection Strategy:**
- Mixed approach: Popular + Top-rated movies
- Minimum 30 reviews per movie
- Rate limiting to respect API constraints

**GRAPHIC:** `07_genre_distribution.png`
> Shows dataset composition - Drama dominates, good mix of genres

---

# SLIDE 5: PostgreSQL Database Design

## Enterprise-Grade Database

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

## Optimizations:
- **Indexes** on release_date, vote_average, sentiment
- **GIN Index** for genre arrays (fast queries)
- **3 SQL Views** for aggregated analysis:
  - `movie_review_stats` - Per-movie aggregations
  - `genre_sentiment_analysis` - Genre breakdown
  - `temporal_sentiment_trends` - Time trends

---

# SLIDE 6: NLP Pipeline

## Text Preprocessing

### 7-Step Text Processor:
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
- Exclamation/question marks

**Tools:** NLTK, BeautifulSoup, scikit-learn

---

# SLIDE 7: Model Architecture

## Three Machine Learning Models

### Model 1: Sentiment Classifier (3-Class)
| Component | Choice |
|-----------|--------|
| Vectorizer | TF-IDF (unigrams + bigrams, max 5000 features) |
| Algorithm | Logistic Regression |
| Class Weights | Balanced (handles imbalanced data) |
| Thresholds | Positive >= 7.0, Negative <= 5.0 |

### Model 2: Score Predictor (Regression)
| Component | Choice |
|-----------|--------|
| Vectorizer | TF-IDF (max 3000 features) |
| Algorithm | Ridge Regression (L2) |
| Features | Text + Metadata (text_length, word_count) |
| Output | Score 0-10 (clipped) |

### Model 3: K-Means Clustering
- Features: TF-IDF + numeric features
- Optimal k via Elbow + Silhouette analysis

---

# SLIDE 8: EDA - Rating Distribution

## Univariate Analysis

### Movie Rating Distribution
- Most ratings concentrated between 7-8
- Mean and median clearly visible
- Slight left skew (fewer low-rated movies)

### Key Statistics:
- **Mean Rating:** ~7.5
- **Median Rating:** ~7.6
- **Standard Deviation:** Shows variance in ratings

**GRAPHIC:** `01_rating_distribution.png`
> Full-width histogram showing rating distribution with mean/median lines

---

# SLIDE 9: EDA - Correlation Analysis

## Bivariate Analysis

### Feature Correlations:
- **Budget vs Revenue:** 0.75 (strong positive)
- **Vote Count vs Popularity:** High correlation
- **Runtime vs Rating:** Weak positive correlation

### Insights:
- Higher budget generally leads to higher revenue
- Popular movies get more votes (expected)
- Runtime has minimal impact on rating

**GRAPHIC:** `04_correlation_heatmap.png`
> Full-width correlation heatmap showing relationships between numeric features

---

# SLIDE 10: EDA - Budget vs Revenue

## ROI Analysis

### Business Insights:
- Break-even line shows profitability threshold
- Most movies above the line = profitable
- Log scale reveals patterns across budget ranges

### Observations:
- High-budget films ($100M+) generally profitable
- Some low-budget films achieve exceptional ROI
- Few movies below break-even (selection bias in TMDb)

**GRAPHIC:** `06_budget_vs_revenue.png`
> Scatter plot with break-even line, log scale on both axes

---

# SLIDE 11: Sentiment Classification Results

## 92% Overall Accuracy

### Class Performance:
| Class | Correct | Total | Accuracy |
|-------|---------|-------|----------|
| **Positive** | 214 | 216 | **99%** |
| Neutral | 17 | 26 | 65% |
| Negative | 17 | 27 | 63% |

### Key Insights:
- **Excellent positive class detection** (99% accuracy)
- Neutral/Negative distinction more challenging
- Class weighting effectively handles imbalanced data

**GRAPHICS:**
- Left: `10_sentiment_distribution.png` - Shows class imbalance (why balanced weights needed)
- Right: `11_confusion_matrix.png` - Classification performance matrix

---

# SLIDE 12: Score Prediction Results

## Ridge Regression Performance

### Model Quality:
- Residuals centered around 0 (no systematic bias)
- Distribution approximately normal
- Good predictions especially for high scores

### Residual Analysis:
- **Left Plot:** Residuals vs Predicted - centered at 0
- **Right Plot:** Predicted vs Actual - points near diagonal

### Top Predictive Terms:
| Positive Indicators | Negative Indicators |
|---------------------|---------------------|
| brilliant, masterpiece | boring, disappointing |
| excellent, perfect | waste, terrible |

**GRAPHIC:** `12_residual_plots.png`
> Side-by-side: Residual Plot + Predicted vs Actual scatter

---

# SLIDE 13: Clustering - Elbow Method & PCA

## K-Means Optimization

### Elbow Method (Left):
- Inertia decreases as k increases
- "Elbow" visible around k=4-5
- Diminishing returns after k=5

### Silhouette Score (Right):
- Measures cluster cohesion and separation
- Higher is better
- Optimal k confirmed at 5

**GRAPHIC:** `13_elbow_plot.png`
> Side-by-side: Inertia curve + Silhouette score by k

---

# SLIDE 14: Clustering - PCA Visualization

## 5 Distinct Movie Clusters

### Cluster Characteristics:
| Cluster | Profile | Avg Score |
|---------|---------|-----------|
| 0 | Blockbuster Action | 6.5 |
| 1 | Indie Drama (Critical Acclaim) | 7.8 |
| 2 | Family Comedy | 6.2 |
| 3 | Horror/Thriller (Polarized) | 5.8 |
| 4 | Low-performing Films | 4.5 |

### PCA Visualization:
- PC1 explains 53.7% of variance
- PC2 explains 25.3% of variance
- Clear cluster separation visible
- Centroids marked with X

**GRAPHIC:** `14_cluster_visualization_2d.png`
> Full-width PCA scatter plot with 5 clusters and centroids

---

# SLIDE 15: Clustering - Distribution

## Cluster Sizes & Validation

### Reviews per Cluster:
- Cluster 1 (Indie Drama): ~136 reviews - largest
- Cluster 0 (Blockbuster): ~35 reviews
- Cluster 3 (Horror): ~55 reviews
- Cluster 2 (Comedy): ~30 reviews
- Cluster 4 (Low Performers): ~18 reviews - smallest

### Statistical Validation:
- **ANOVA:** Significant score differences (p < 0.05)
- **Chi-squared:** Sentiment distribution differs by cluster

### Business Value:
- Target marketing strategies by cluster
- Early warning system for Cluster 4 (potential flops)

**GRAPHIC:** `15_cluster_distribution.png`
> Bar chart showing reviews per cluster

---

# SLIDE 16: Key Statistical Findings

## Summary of Statistical Insights

### Genre Analysis (ANOVA p < 0.01):
- Drama and Thriller: Significantly higher ratings
- Horror: Most polarized (highest variance)
- Comedy: Consistently moderate ratings

### Runtime Correlation (Pearson p < 0.05):
- Significant positive correlation with rating
- Movies >150 min: Higher but polarized ratings

### Chi-squared Tests (p < 0.05):
- Genre strongly associated with rating category
- Production country influences sentiment distribution

### Key Takeaway:
All statistical tests show significant results with p-values documented - ensuring scientific rigor.

---

# SLIDE 17: Conclusions

## Summary

### What We Achieved:
1. **End-to-end pipeline** from API to predictions
2. **Sentiment classifier** with 92% accuracy
3. **Score predictor** with good R² and low RMSE
4. **Clustering** reveals 5 meaningful movie segments
5. **Statistical rigor** with p-values for all tests

### Limitations:
- English reviews only
- TMDb-specific ratings may differ from other platforms
- Neutral class hardest to predict (65% accuracy)

### Future Work:
- Deep learning with BERT/Transformers
- Multi-language support
- Real-time streaming pipeline
- Recommendation system based on clusters

---

# SLIDE 18: Thank You

## Questions?

**Thank you for your attention!**

The appendix contains detailed documentation with screenshots that verify all bonus point requirements.

---

**Team Contact:**
- Lara Bangerter
- Michele Maniaci
- Daniele Magnano

**Repository:** MovieMind - ZHAW Data Analytics Project

---

# SLIDE 19: Appendix - Bonus Points Documentation

## 1. PostgreSQL Schema with Optimizations
- 3 tables with proper relationships (movies, reviews, countries)
- GIN indexes for array columns
- 3 SQL views for analytics
- **Evidence:** schema.sql

## 2. Statistical Tests with p-values
- Chi-squared: Genre vs Rating (p < 0.05)
- ANOVA: Rating across genres (p < 0.01)
- Pearson correlation with significance
- **Evidence:** notebook cells with scipy.stats

## 3. K-Means Clustering
- Elbow method implementation
- Silhouette score analysis
- PCA 2D visualization with centroids
- **Evidence:** clustering.py, notebook

## 4. Regression with Diagnostics
- Ridge regression with R², RMSE, MAE
- Residual plot analysis (centered at 0)
- Predicted vs Actual scatter
- **Evidence:** score_predictor.py, notebook

## 5. Confusion Matrix
- 3-class classification matrix
- Per-class precision/recall/F1
- 92% overall accuracy
- **Evidence:** sentiment_classifier.py, notebook

---

# Graphics Reference

## 10 Graphics Used (in presentation/screenshots/renamed/used/):

| # | Filename | Slide | Purpose |
|---|----------|-------|---------|
| 1 | `01_rating_distribution.png` | 8 | Rating distribution histogram |
| 2 | `04_correlation_heatmap.png` | 9 | Feature correlations |
| 3 | `06_budget_vs_revenue.png` | 10 | ROI analysis scatter |
| 4 | `07_genre_distribution.png` | 4 | Dataset composition |
| 5 | `10_sentiment_distribution.png` | 11 | Class imbalance |
| 6 | `11_confusion_matrix.png` | 11 | Classification results |
| 7 | `12_residual_plots.png` | 12 | Regression diagnostics |
| 8 | `13_elbow_plot.png` | 13 | K selection (elbow + silhouette) |
| 9 | `14_cluster_visualization_2d.png` | 14 | Cluster PCA visualization |
| 10 | `15_cluster_distribution.png` | 15 | Cluster sizes |

---

# Speaker Distribution (Updated)

| Speaker | Slides | Time |
|---------|--------|------|
| **Lara** | 1-5 (Title, Background, Data, Database) | ~5 min |
| **Michele** | 6-10 (NLP, Models, EDA x3) | ~5 min |
| **Daniele** | 11-18 (Results, Clustering, Stats, Conclusions) | ~5 min |

**Total: 18 Slides + 1 Appendix = 19 Slides**

---

# END OF SLIDES
