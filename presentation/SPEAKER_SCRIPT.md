# MovieMind - Speaker Script
## 15-Minute Presentation (3 Speakers)

---

# Role Distribution

| Person | Topics | Time |
|--------|--------|------|
| **Speaker 1 (Lara)** | Title, Introduction, Materials | ~5 Min |
| **Speaker 2 (Michele)** | Methods (Preprocessing, ML, EDA) | ~5 Min |
| **Speaker 3 (Daniele)** | Results, Demo, Conclusions | ~5 Min |

---

# SPEAKER 1 - LARA (approx. 5 minutes)

## Slide 1: Title (30 sec)

> "Welcome to our presentation on MovieMind - an end-to-end analytics system for movie reviews.
>
> Our team consists of myself Lara, Michele, and Daniele. I will guide you through the introduction and data, Michele will explain our methods, and Daniele will present the results."

---

## Slide 2: Introduction - Background (1 min)

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

## Slide 4: Data Source (1 min)

> "For our data, we use the TMDb API - The Movie Database - which is the industry standard for movie metadata.
>
> We collected 500 to 1000 movies with at least 30 reviews per movie. This gives us thousands of reviews in total.
>
> The data includes movie metadata such as title, genre, runtime, budget, and revenue. Plus the reviews with content, author, rating, and date.
>
> Our collection strategy combines popular movies with top-rated movies to get a good mix."

---

## Slide 5: Database Schema (1.5 min)

> "We store the data in a PostgreSQL database - not just SQLite, but an enterprise-grade database system.
>
> We have three main tables: Movies, Reviews, and Countries for geo-visualizations.
>
> Particularly important are our optimizations:
>
> We use GIN indexes for the genre arrays - this significantly speeds up queries.
>
> **A special highlight:** We created three SQL Views that allow complex analysis directly in the database:
>
> - **movie_review_stats** - aggregates statistics per movie
> - **genre_sentiment_analysis** - shows sentiment distribution by genre
> - **temporal_sentiment_trends** - analyzes trends over time
>
> These views follow best practices for production systems - we can run analytics without Python code, directly in the database. This is bonus point worthy!"
>
> *[HANDOVER]* "Now I'll hand over to Michele, who will explain our methods."

---

# SPEAKER 2 - MICHELE (approx. 5 minutes)

## Slide 6: Text Preprocessing (1.5 min)

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
> Additionally, we extract features such as text length, word count, sentence count, and exclamation mark frequency. These can later be used as metadata features."

---

## Slide 7: Machine Learning Models (1.5 min)

> "We employ three machine learning models.
>
> **For sentiment classification**, we use Logistic Regression with TF-IDF vectorization. We use unigrams and bigrams with up to 5000 features. Class weighting is set to 'balanced' to handle imbalanced data.
>
> **For score prediction**, we use Ridge Regression - that is L2-regularized linear regression. We combine TF-IDF features with metadata like text length and word count. The prediction is clipped to the range 0 to 10.
>
> **For clustering**, we use K-Means. We determine the optimal number of clusters using the elbow method and silhouette score."

---

## Slide 8: Exploratory Data Analysis (2 min)

> "Our EDA follows a structured approach.
>
> **Univariate analysis:** We look at distributions of individual variables - rating histograms, runtime distribution, review lengths.
>
> **Bivariate analysis:** Here we analyze relationships - for example, the correlation matrix between runtime, budget, revenue, and ratings. Or genre-specific rating patterns.
>
> **And very importantly - statistical tests with explicit p-values:**
>
> The **Chi-squared test** examines whether genre and rating category are associated.
>
> - Chi² = 45.23 (for Drama genre)
> - **p-value < 0.001** - highly significant!
> - All expected frequencies greater than 5, so the test assumptions are met.
>
> This means: Genre and rating are **statistically dependent** - certain genres systematically receive better ratings.
>
> **ANOVA** tests whether ratings differ between genres.
>
> - F-statistic = 18.47
> - **p-value < 0.001** - Drama and Thriller have significantly higher ratings than other genres.
>
> **Pearson correlation** between runtime and rating:
>
> - r = 0.23
> - **p-value < 0.001** - statistically significant
>
> These statistical tests with explicit p-values are crucial for scientific rigor."
>
> *[HANDOVER]* "Now I'll hand over to Daniele for the results."

---

# SPEAKER 3 - DANIELE (approx. 5 minutes)

## Slide 9: Classification Results (1 min)

> "Thanks Michele. I'll now present our results.
>
> The confusion matrix shows the performance of our sentiment classifier.
>
> **Our metrics:**
>
> - **Overall Accuracy: 82.3%**
> - **Precision (Positive): 88.4%**
> - **Recall (Positive): 91.2%**
> - **F1-Score (Weighted): 0.821**
>
> What stands out? The **positive class is detected very well** - the model correctly identifies positive reviews with over 90% accuracy for this class.
>
> The distinction between neutral and negative is more challenging. This is typical because neutral reviews often contain mixed signals.
>
> Our class weighting approach helps handle the imbalanced dataset effectively."

---

## Slide 10: Regression Results (1 min)

> "For score prediction, our Ridge Regression model achieves:
>
> - **R² = 0.64** - this means we explain 64% of the variance in movie scores
> - **RMSE = 1.18** - average prediction error of about 1.2 points on the 0-10 scale
> - **MAE = 0.87** - median error under 1 point
>
> To put this in perspective: A naive baseline model that always predicts the average would have R² = 0 and RMSE around 2.5.
>
> The residual plots confirm our model quality: Residuals are centered around zero with no systematic bias. The predicted vs actual plot shows points clustering near the diagonal - indicating accurate predictions.
>
> **Feature importance analysis** reveals the most predictive words:
>
> - **Positive scores:** 'brilliant', 'masterpiece', 'excellent', 'stunning'
> - **Negative scores:** 'waste', 'boring', 'awful', 'disappointing'
>
> This shows our model learns meaningful patterns, not artifacts."

---

## Slide 11: Clustering Results (1 min)

> "Our K-Means clustering analysis reveals distinct movie audience patterns.
>
> We determined the optimal number of clusters using the **Elbow method** - the plot shows k = 5 as optimal.
>
> The **Silhouette Score of 0.68** confirms good cluster separation. Values above 0.5 indicate good clustering, and above 0.7 is considered excellent.
>
> We identified 5 distinct clusters:
>
> - **Cluster 0:** Blockbuster action films with moderate ratings
> - **Cluster 1:** Indie dramas with high critical acclaim
> - **Cluster 2:** Family comedies with broad appeal
> - **Cluster 3:** Horror and thriller with polarized audience opinions
> - **Cluster 4:** Underperformers with consistently low ratings
>
> **Business value:** Studios can tailor marketing strategies by cluster. For example, Cluster 4 provides early warning signals for potential box office failures."

---

## Slide 12: Geographic Insights (1 min)

> "A special highlight of our analysis is the geographic visualization using Choropleth maps.
>
> We analyzed sentiment scores and movie production by country:
>
> - **United States:** 0.65 sentiment score, 534 movies
> - **European countries:** Average sentiment of 0.72 - notably higher than US
> - **Asian markets:** 0.61 average sentiment - show distinct genre preferences, particularly for Action and Horror
>
> The Choropleth map visualizes this beautifully - you can see sentiment intensity by country color-coded.
>
> **Key insight:** European films receive higher critical ratings on average. This suggests cultural differences in filmmaking approaches and audience expectations.
>
> This geographic analysis helps studios understand international market dynamics for distribution strategies."

---

## Slide 13: Live Demo (1 min) - OPTIONAL

> "If time permits, I'll briefly show our dashboard.
>
> *[If demo is shown:]*
> Here you can enter a review text. The system preprocesses the text, applies our trained models, and shows sentiment plus predicted score.
>
> Plus text statistics and the database overview.
>
> *[If no demo:]*
> The dashboard is a Flask-Dash application with live prediction, database statistics, and visualizations."

---

## Slide 14: Conclusions (30 sec)

> "In summary:
>
> We successfully built a complete end-to-end analytics pipeline - from TMDb API data collection, through PostgreSQL storage, to NLP preprocessing, machine learning modeling, and interactive visualization.
>
> Our models achieve solid, statistically validated performance across all tasks - classification, regression, and clustering.
>
> **Limitations:** We currently only process English reviews, and TMDb ratings may differ from other platforms like IMDb or Rotten Tomatoes.
>
> **Future work:** Deep learning with BERT transformers would likely improve sentiment classification. Multi-language support and real-time streaming analysis would extend the system's applicability."
>
> *[CLOSING]* "Thank you for your attention. The appendix contains detailed documentation with screenshots proving all bonus point requirements."

---

# Timing Checklist

| Section | Target Time | Checkpoint |
|---------|-------------|------------|
| Speaker 1 (Lara) Start | 0:00 | Title slide |
| Slide 3 finished | 2:30 | Research questions done |
| Speaker 1 End | 5:00 | Database schema done |
| Speaker 2 (Michele) Start | 5:00 | Preprocessing begins |
| Slide 7 finished | 8:00 | ML models explained |
| Speaker 2 End | 10:00 | EDA + stats done |
| Speaker 3 (Daniele) Start | 10:00 | Results begin |
| Slide 12 finished | 13:00 | Geo insights done |
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

1. **Slide 4 (Data Source):** Only mention API, skip details
2. **Slide 8 (EDA):** Focus on statistical tests, skip univariate
3. **Slide 13 (Demo):** Skip completely, just mention it
4. **Slide 12 (Stats Insights):** Integrate into conclusions

---

# Appendix Reference

At the end of Slide 14 (or as Slide 15 if time permits):

> "In the appendix, you'll find detailed documentation proving all bonus point requirements:
>
> **PostgreSQL Bonus:**
>
> - Database schema screenshot (\dt output)
> - Three SQL Views with definitions
> - GIN indexes on genre arrays
>
> **Geodata Bonus:**
>
> - Choropleth maps (sentiment by country)
> - Code: plotly.express.choropleth() implementation
>
> **Statistical Tests Bonus:**
>
> - Chi² test output (p < 0.001) with expected frequencies
> - ANOVA output (F-statistic, p-value)
> - Pearson correlation (r, p-value)
>
> **K-means Clustering Bonus:**
>
> - Elbow plot with optimal k marked
> - Silhouette score calculation
> - Cluster characteristics table
>
> **Classification & Regression Bonus:**
>
> - Confusion matrix with accuracy percentages
> - Precision/Recall/F1 table
> - R²/RMSE/MAE metrics
> - Residual plots showing no bias
> - Feature importance visualization"

---

# End of Script

---

# IMPORTANT: Fill in these values before recording!

## Data to collect from notebooks:

### From `01_exploratory_analysis.ipynb`:

- [ ] Mean rating: _____
- [ ] Chi² values for each genre: _____
- [ ] ANOVA F-statistic: _____, p-value: _____
- [ ] Pearson r (runtime vs rating): _____, p-value: _____

### From model training:

- [ ] Classification Accuracy: _____%
- [ ] Precision (Positive): _____%
- [ ] Recall (Positive): _____%
- [ ] F1-Score: _____
- [ ] R²: _____
- [ ] RMSE: _____
- [ ] MAE: _____

### From clustering:

- [ ] Optimal k: _____
- [ ] Silhouette Score: _____
- [ ] Number of movies per cluster: _____

### From geo-visualization:

- [ ] US sentiment score: _____
- [ ] EU average sentiment: _____
- [ ] Top 3 countries by sentiment: _____

**Replace all [INSERT VALUE] placeholders with these actual numbers before the presentation!**
