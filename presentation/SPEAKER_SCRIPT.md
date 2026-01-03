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

> "We store the data in a PostgreSQL database.
>
> We have three main tables: Movies, Reviews, and Countries for geo-visualizations.
>
> Particularly important are our optimizations:
>
> We use GIN indexes for the genre arrays - this significantly speeds up queries.
>
> Additionally, we created three SQL views:
> - **movie_review_stats** aggregates statistics per movie
> - **genre_sentiment_analysis** shows sentiment distribution by genre
> - **temporal_sentiment_trends** analyzes trends over time
>
> This shows that we don't just store the data, but also analyze it directly in the database."
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
> **And very importantly - statistical tests with p-values:**
>
> The Chi-squared test shows whether genre and rating category are associated. Result: p below 0.05 - so it's significant.
>
> ANOVA tests whether ratings differ between genres. Result: p below 0.01 - Drama and Thriller have significantly higher ratings.
>
> Pearson correlation between runtime and rating is also significant.
>
> These statistical tests with explicit p-values are important for scientific rigor."
>
> *[HANDOVER]* "Now I'll hand over to Daniele for the results."

---

# SPEAKER 3 - DANIELE (approx. 5 minutes)

## Slide 9: Classification Results (1 min)

> "Thanks Michele. I'll now present our results.
>
> The confusion matrix shows the performance of our sentiment classifier.
>
> What stands out? The **positive class is detected very well** - 214 out of about 235 correctly classified. That's an accuracy of over 90% for this class.
>
> The distinction between neutral and negative is more difficult. This is typical because neutral reviews often have mixed signals.
>
> Overall, we achieve about 80% accuracy. Class weighting helps with imbalance, but there's still room for improvement."

---

## Slide 10: Regression Results (1 min)

> "For score prediction, we see the residual plots.
>
> The left plot shows residuals against predicted values. The residuals are centered around zero - that's good, it shows no systematic bias.
>
> The right plot shows predicted versus actual. The points lie close to the diagonal - the closer, the better the prediction.
>
> Our R-squared and RMSE show that the model explains a significant portion of the variance.
>
> The most important features for good scores are words like 'brilliant', 'masterpiece', 'excellent'. For bad scores: 'boring', 'disappointing', 'waste'."

---

## Slide 11: Clustering Results (1 min)

> "K-Means clustering with k equals 5 shows interesting patterns.
>
> We see five distinct clusters:
> - Cluster 1: Blockbuster action with moderate ratings
> - Cluster 2: Indie dramas with high critical acclaim
> - Cluster 3: Family comedies
> - Cluster 4: Horror and thriller with polarized opinions
> - Cluster 5: Underperformers with low ratings
>
> The silhouette score confirms good cluster separation.
>
> The business value: Studios can differentiate marketing strategies by cluster and detect early warning signals for Cluster 5."

---

## Slide 12: Statistical Insights (30 sec)

> "In summary, the key statistical findings:
>
> Genre is strongly associated with rating. Drama and Thriller perform significantly better.
>
> Longer movies tend to have higher but also more polarized ratings.
>
> All our hypothesis tests have significant p-values below 0.05."

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
> We built a complete end-to-end pipeline - from API to prediction.
>
> Our models achieve solid performance with statistically validated results.
>
> As limitations: We only work with English reviews, and TMDb ratings may differ from other platforms.
>
> For the future, deep learning with BERT would be interesting, as well as multi-language support."
>
> *[CLOSING]* "Thank you for your attention. The appendix shows our detailed documentation of bonus points with screenshots."

---

# Timing Checklist

| Section | Target Time | Checkpoint |
|---------|-------------|------------|
| Speaker 1 Start | 0:00 | |
| Slide 3 finished | 2:30 | |
| Speaker 1 End | 5:00 | |
| Speaker 2 Start | 5:00 | |
| Slide 7 finished | 8:00 | |
| Speaker 2 End | 10:00 | |
| Speaker 3 Start | 10:00 | |
| Slide 12 finished | 13:00 | |
| Speaker 3 End | 15:00 | |

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

At the end of Slide 14:

> "In the appendix, you'll find detailed documentation of our bonus points:
> - PostgreSQL schema with views and indexes
> - All statistical tests with p-values
> - K-Means clustering with elbow plot and silhouette
> - Regression diagnostics with residual plots
> - Confusion matrix and classification report
> - Code snippets from our implementations"

---

# End of Script
