# MovieMind - Speaker Script
## 15-Minute Presentation (3 Speakers)

---

# Role Distribution

| Person | Topics | Time |
|--------|--------|------|
| **Speaker 1 (Lara)** | Title, Introduction, Materials (Slides 1-5) | ~5 Min |
| **Speaker 2 (Michele)** | Methods: NLP, ML Models, EDA (Slides 6-8) | ~5 Min |
| **Speaker 3 (Daniele)** | Results, Conclusions (Slides 9-15) | ~5 Min |

---

# SPEAKER 1 - LARA (approx. 5 minutes)

## Slide 1: Title (30 sec)

> "Welcome to our presentation on MovieMind - an end-to-end analytics system for movie reviews.
>
> Our team consists of myself Lara, Michele, and Daniele. I will guide you through the introduction and data, Michele will explain our methods, and Daniele will present the results."

---

## Slide 2: Why Movie Review Analytics? (1 min)

> "Why do we analyze movie reviews?
>
> Streaming platforms like Netflix and Disney+ receive thousands of user reviews every day. Manual analysis is time-consuming and inconsistent.
>
> The global streaming market is worth over 300 billion dollars. Early detection of negative trends can save millions in marketing costs.
>
> Studios need automated sentiment analysis to quickly respond to user feedback."

---

## Slide 3: Objective & Research Questions (1 min)

> "Our goal is to build a complete analytics pipeline that transforms unstructured reviews into actionable insights.
>
> We have four central research questions:
>
> **First:** Can sentiment in reviews be accurately classified - meaning positive, neutral, or negative?
>
> **Second:** Can we predict movie ratings from 0 to 10 based on the review text?
>
> **Third:** What patterns exist across genres and over time?
>
> **And fourth:** How do movies cluster based on audience reactions?"

---

## Slide 4: Data Collection (1 min)

> "For our data, we use the TMDb API - The Movie Database - which is the industry standard for movie metadata.
>
> We collected 500 to 1000 movies with at least 30 reviews per movie. This gives us thousands of reviews in total.
>
> The data includes movie metadata such as title, genre, runtime, budget, and revenue. Plus the reviews with content, author, rating, and date.
>
> Our collection strategy combines popular movies with top-rated movies to get a good mix."

---

## Slide 5: PostgreSQL Database Design (1.5 min)

> "We store the data in a PostgreSQL database - not just SQLite, but an enterprise-grade database system.
>
> We have two main tables: Movies and Reviews, connected through a 1:N relationship.
>
> Particularly important are our optimizations:
>
> We use indexes on release_date, vote_average, and sentiment for fast queries. Especially the GIN index for genre arrays significantly speeds up queries.
>
> **A special highlight:** We created three SQL views:
>
> - **movie_review_stats** - aggregates statistics per movie
> - **genre_sentiment_analysis** - shows sentiment distribution by genre
> - **temporal_sentiment_trends** - analyzes trends over time
>
> These views allow complex analyses directly in the database - best practice for production systems!"
>
> *[HANDOVER]* "Now I'll hand over to Michele, who will explain our methods."

---

# SPEAKER 2 - MICHELE (approx. 5 minutes)

## Slide 6: NLP Pipeline (1.5 min)

> "Thanks Lara. I'll now explain our NLP pipeline.
>
> Text data is unstructured and needs to be preprocessed. Our TextProcessor goes through seven steps:
>
> **First:** Remove HTML tags with BeautifulSoup
> **Second:** Remove URLs with Regex
> **Third:** Convert everything to lowercase
> **Fourth:** Remove special characters
> **Fifth:** Tokenization with NLTK
> **Sixth:** Remove stopwords
> **And seventh:** Lemmatization - reducing words to their base form
>
> Additionally, we extract features such as text length, word count, sentence count, and uppercase ratio. These can later be used as metadata features.
>
> The tools we use: NLTK word_tokenize, WordNetLemmatizer, and BeautifulSoup."

---

## Slide 7: Model Architecture (1.5 min)

> "We employ three machine learning models.
>
> **Model 1 - Sentiment Classifier:** We use Logistic Regression with TF-IDF vectorization. We use unigrams and bigrams with up to 5000 features. Class weighting is set to 'balanced' to handle imbalanced data. The thresholds are: Positive for rating >= 7.0, Negative for <= 5.0.
>
> **Model 2 - Score Predictor:** Here we use Ridge Regression - that is L2-regularized linear regression. We combine TF-IDF features with metadata like text length. The prediction is clipped to 0 to 10.
>
> **Model 3 - Clustering:** We use K-Means. We determine the optimal number of clusters using the elbow method and silhouette score."

---

## Slide 8: Exploratory Data Analysis (2 min)

> "Our EDA follows a structured approach.
>
> **Univariate analysis:** We look at distributions of individual variables - rating histograms, runtime distribution, review lengths.
>
> **Bivariate analysis:** Here we analyze relationships - for example, the correlation heatmap between runtime, budget, revenue, and ratings. Or genre-specific rating patterns.
>
> **And very importantly - statistical tests with explicit p-values:**
>
> The **Chi-squared test** checks whether genre and rating category are associated.
> - **p-value < 0.05** - statistically significant!
>
> **ANOVA** tests whether ratings differ between genres.
> - **p-value < 0.01** - Drama and Thriller have significantly higher ratings.
>
> **Pearson correlation** between runtime and rating:
> - **p-value < 0.05** - statistically significant
>
> These statistical tests with explicit p-values are crucial for scientific rigor."
>
> *[HANDOVER]* "Now I'll hand over to Daniele for the results."

---

# SPEAKER 3 - DANIELE (approx. 5 minutes)

## Slide 9: Sentiment Classification Results (1 min)

> "Thanks Michele. I'll now present our results.
>
> The confusion matrix shows the performance of our sentiment classifier.
>
> What immediately stands out: The **positive class is detected excellently** - 214 out of 216 positive reviews were correctly classified. That's a hit rate of over 99% for this class.
>
> **Our overall accuracy is 92%.**
>
> The neutral class achieves 17 out of 26 correct - about 65%.
> The negative class achieves 17 out of 27 correct - about 63%.
>
> The distinction between neutral and negative is more challenging. This is typical because neutral reviews often contain mixed signals.
>
> Our class weighting approach effectively helps handle the imbalanced dataset."

---

## Slide 10: Score Prediction Results (1 min)

> "For score prediction, our Ridge Regression model shows very good results.
>
> The four plots show different aspects of model quality:
>
> **Predicted vs Actual:** The points cluster close to the diagonal - this shows accurate predictions, especially for high scores.
>
> **Residual Plot:** Residuals are centered around zero with no systematic bias.
>
> **Distribution of Residuals:** The mean is exactly -0.00 - perfectly centered. The distribution is approximately normal.
>
> **Q-Q Plot:** Confirms the normal distribution of residuals.
>
> **Feature Importance** shows the most predictive words:
> - **Positive:** 'brilliant', 'masterpiece', 'excellent', 'perfect'
> - **Negative:** 'boring', 'disappointing', 'waste', 'terrible'
>
> This shows that our model learns meaningful patterns."

---

## Slide 11: K-Means Clustering (1 min)

> "Our K-Means clustering analysis reveals five distinct movie clusters.
>
> **Cluster 0 - Blockbuster Action:** Average score 6.5
> **Cluster 1 - Indie Drama:** Highest score at 7.8 - critically acclaimed
> **Cluster 2 - Family Comedy:** Score 6.2 - broad appeal
> **Cluster 3 - Horror/Thriller:** Score 5.8 - polarized opinions
> **Cluster 4 - Low Performers:** Lowest score at 4.5
>
> **Validation:** The ANOVA test confirms significant score differences between clusters with p < 0.05. The Chi-squared test shows that sentiment distribution differs significantly between clusters.
>
> **Business Value:** Studios can tailor marketing strategies by cluster. Cluster 4 serves as an early warning system for potential flops."

---

## Slide 12: Key Statistical Findings (1 min)

> "Here's a summary of our statistical findings:
>
> **Genre Analysis** with ANOVA p < 0.01:
> - Drama and Thriller have significantly higher ratings
> - Horror is the most polarized with highest variance
> - Comedy shows consistently moderate ratings
>
> **Runtime Correlation:**
> - Significantly positive correlation with rating
> - Movies over 150 minutes: Higher but polarized ratings
>
> **Temporal Trends:**
> - Identifiable seasonal patterns in review volume
> - Sentiment is relatively stable over time
>
> **Chi-squared Tests:**
> - Genre is strongly associated with rating category
> - Production country influences sentiment distribution"

---

## Slide 13: Interactive Web Application (30 sec)

> "We also developed an interactive dashboard.
>
> The Flask-Dash application offers:
> - **Live Prediction:** Enter review text and instantly get sentiment plus score
> - **Database Statistics:** Movie and review counts, genre distribution
> - **Visualizations:** Rating histograms and interactive Plotly charts
>
> The dashboard demonstrates the practical applicability of our pipeline."

---

## Slide 14: Conclusions (1 min)

> "In summary:
>
> **What we achieved:**
> - A complete end-to-end pipeline from API to predictions
> - Sentiment classifier with over 90% accuracy
> - Score predictor with good R² and low RMSE
> - Clustering reveals meaningful movie segments
> - Statistical rigor with p-values for all tests
>
> **Limitations:**
> - English reviews only
> - TMDb-specific ratings may differ from other platforms
> - Neutral class is hardest to predict
>
> **Future Work:**
> - Deep learning with BERT/Transformers
> - Multi-language support
> - Real-time streaming pipeline
> - Recommendation system based on clusters"

---

## Slide 15: Thank You (30 sec)

> "Thank you for your attention!
>
> Do you have any questions?
>
> In the appendix, you'll find detailed documentation with screenshots that verify all bonus point requirements."

---

## Slide 16: Appendix (Only show if asked)

> *[Only mention if asked:]*
>
> "The appendix documents all bonus points:
>
> 1. **PostgreSQL Schema** - 3 tables, GIN indexes, 3 SQL views
> 2. **Statistical Tests** - Chi-squared, ANOVA, Pearson with p-values
> 3. **K-Means Clustering** - Elbow method, silhouette score, PCA visualization
> 4. **Regression Diagnostics** - R², RMSE, MAE, residual plots
> 5. **Confusion Matrix** - 3-class classification with per-class metrics"

---

# Timing Checklist

| Section | Target Time | Checkpoint |
|---------|-------------|------------|
| Speaker 1 (Lara) Start | 0:00 | Title slide |
| Slide 3 finished | 2:30 | Research Questions done |
| Speaker 1 End | 5:00 | Database Schema done |
| Speaker 2 (Michele) Start | 5:00 | NLP Pipeline begins |
| Slide 7 finished | 8:00 | ML Models explained |
| Speaker 2 End | 10:00 | EDA + Stats done |
| Speaker 3 (Daniele) Start | 10:00 | Results begin |
| Slide 12 finished | 13:00 | Statistical Insights done |
| Slide 14 finished | 14:30 | Conclusions done |
| Speaker 3 End | 15:00 | Q&A or buffer |

---

# Tips for Recording

1. **Preparation:**
   - Read through the script several times
   - Mark important words
   - Practice timing

2. **Technical:**
   - Use a good microphone
   - Quiet environment
   - Test screen recording

3. **Presentation:**
   - Speak freely, don't read word for word
   - Point to/highlight slides
   - Pause during handovers

4. **Handovers:**
   - Short + clear: "Now I'll hand over to [Name]..."
   - Maintain eye contact (virtually)

5. **Appendix:**
   - Don't present live
   - Just mention: "Details in the appendix"

---

# Emergency Cuts

If time is running short, cut here:

1. **Slide 4 (Data Collection):** Only mention API, skip details
2. **Slide 8 (EDA):** Focus on statistical tests, skip univariate
3. **Slide 12 (Statistical Insights):** Integrate into conclusions
4. **Slide 13 (Dashboard):** Skip completely, just mention in conclusions

---

# Key Numbers from the Presentation

## Sentiment Classification (Slide 9):
- **Overall Accuracy:** 92%
- **Positive:** 214/216 correct (99%)
- **Neutral:** 17/26 correct (65%)
- **Negative:** 17/27 correct (63%)

## Regression (Slide 10):
- **Mean Residual:** -0.00 (perfectly centered)
- **Residuals:** Normally distributed (Q-Q plot confirms)

## Clustering (Slide 11):
- **k = 5** clusters
- **ANOVA:** p < 0.05 (significant)
- **Cluster Scores:** 7.8 (Indie Drama) to 4.5 (Low Performers)

## Statistical Tests (Slide 8 & 12):
- **Chi-squared:** p < 0.05
- **ANOVA:** p < 0.01
- **Pearson:** p < 0.05

---

# End of Script
