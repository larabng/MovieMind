# MovieMind - Presentation Outline

**Project**: End-to-End Movie Review Analytics System
**Duration**: 15 minutes (3 team members × 5 minutes)
**Deadline**: January 11, 2026

---

## Slide Structure

### Slide 1: Title Slide (30 seconds)
**Title**: MovieMind - End-to-End Movie Review Analytics
**Subtitle**: NLP & ML-Driven Insights for the Film Industry
**Team Members**: [Names]
**Date**: January 2026

**Visual**: Movie-related imagery with data visualization overlay

---

### Slide 2: Introduction & Motivation (1 minute)
**Speaker**: Team Member 1

**Content**:
- **Problem**: Film industry needs data-driven insights from reviews
- **Opportunity**: Millions of reviews contain valuable patterns
- **Our Solution**: End-to-end pipeline from raw data to actionable insights

**Visuals**:
- Icon flow: Reviews → Processing → ML → Insights
- Key stakeholders: Streaming platforms, studios, audiences

**Key Points**:
- Early detection of audience sentiment
- Genre-specific patterns
- Score prediction for unreleased films

---

### Slide 3: Project Scope & Objectives (1 minute)
**Speaker**: Team Member 1

**Content**:
- **Data Source**: TMDb API (500-1000 movies, thousands of reviews)
- **Storage**: PostgreSQL database with optimized schema
- **Analysis**: NLP preprocessing + ML modeling
- **Outputs**:
  - Sentiment classification (positive/neutral/negative)
  - Score prediction (0-10 scale)
  - Clustering analysis
  - Geographic insights

**Visuals**:
- Architecture diagram showing data flow
- Sample database schema

---

### Slide 4: Methodology - Data Pipeline (2 minutes)
**Speaker**: Team Member 2 (API & DB Lead)

**Content**:
1. **Data Collection**
   - TMDb API integration
   - Rate limiting & error handling
   - Movies: metadata, genres, countries, ratings
   - Reviews: text, author, timestamps

2. **Data Storage**
   - PostgreSQL schema: `movies`, `reviews`, `countries`
   - Indexes for performance
   - SQL views for aggregations

3. **Data Preprocessing**
   - HTML removal, URL cleaning
   - Lowercasing, stopword removal
   - Lemmatization using NLTK
   - Feature extraction: text_length, word_count

**Visuals**:
- ETL pipeline diagram
- Database schema screenshot
- Before/after text preprocessing example

**Code Snippet** (optional):
```sql
-- Example SQL View
CREATE VIEW movie_review_stats AS
SELECT m.movie_id, m.title,
       COUNT(r.review_id) as review_count,
       AVG(r.sentiment_score) as avg_sentiment
FROM movies m
LEFT JOIN reviews r ON m.movie_id = r.movie_id
GROUP BY m.movie_id;
```

---

### Slide 5: Exploratory Data Analysis (2 minutes)
**Speaker**: Team Member 3 (NLP & EDA Lead)

**Content**:
- **Univariate Analysis**
  - Rating distribution (mean, median, std)
  - Review length distribution

- **Bivariate Analysis**
  - Runtime vs rating correlation
  - Genre vs average rating

- **Statistical Tests** (with p-values):
  - **Chi-squared**: Genre vs Rating Category (p < 0.05)
  - **ANOVA**: Rating differences across genres (p < 0.01)
  - **Pearson correlation**: Runtime vs Rating (r = 0.XX, p < 0.05)

**Visuals**:
- Rating distribution histogram
- Correlation heatmap
- Box plots: Rating by genre
- Statistical test results table

**Key Findings**:
- "Drama and Thriller have significantly higher ratings (ANOVA, p < 0.01)"
- "Movies >150 min show polarized ratings (variance analysis)"

---

### Slide 6: Machine Learning Models (3 minutes)
**Speaker**: Team Member 1 (Modeling Lead)

**Content**:

#### A. Sentiment Classification
- **Algorithm**: Logistic Regression with TF-IDF (unigrams + bigrams)
- **Classes**: Positive, Neutral, Negative
- **Features**: 5000 TF-IDF features
- **Results**:
  - **Accuracy**: XX% (>80% target)
  - **Precision**: XX%
  - **Recall**: XX%
  - **F1-Score**: XX%

**Visual**: Confusion matrix

#### B. Score Prediction
- **Algorithm**: Ridge Regression (L2 regularization)
- **Features**: TF-IDF + metadata (text_length, word_count)
- **Results**:
  - **R²**: 0.XX
  - **RMSE**: X.XX
  - **MAE**: X.XX

**Visual**:
- Predicted vs Actual scatter plot
- Residual plot

**Interpretation**:
- "Key predictive terms for positive sentiment: 'masterpiece', 'brilliant', 'compelling'"
- "Negative indicators: 'boring', 'confusing', 'disappointing'"

---

### Slide 7: Clustering Analysis (2 minutes)
**Speaker**: Team Member 2

**Content**:
- **Method**: K-means clustering
- **Features**: Ratings, sentiment, genres, runtime, budget
- **Optimal k**: Determined by Elbow method + Silhouette score
  - **Silhouette Score**: 0.XX

**Cluster Interpretation** (example for k=5):
- **Cluster 0**: Blockbuster Action (high budget, high revenue, avg rating 7.2)
- **Cluster 1**: Indie Drama (low budget, high critical acclaim, avg rating 7.8)
- **Cluster 2**: Family Comedy (moderate budget, broad appeal, avg rating 6.5)
- **Cluster 3**: Horror/Thriller (niche audience, polarized sentiment)
- **Cluster 4**: Low-performing films (low ratings, negative sentiment)

**Visuals**:
- Elbow plot
- Silhouette plot
- PCA 2D visualization of clusters
- Cluster characteristics table

---

### Slide 8: Geographic Insights (1.5 minutes)
**Speaker**: Team Member 3

**Content**:
- **Analysis by Production Country**
  - Top producing countries: US, UK, France, Germany
  - Average sentiment by country
  - Rating variations across regions

**Visuals**:
- **Choropleth map**: Sentiment score by country
- **Choropleth map**: Number of movies by country
- Bar chart: Top 15 countries by positive review percentage

**Insights**:
- "European films receive higher critical ratings (avg 7.5 vs 6.8 for US)"
- "Asian markets show distinct genre preferences (Action, Horror)"

---

### Slide 9: Demo - Interactive Dashboard (1.5 minutes)
**Speaker**: Team Member 1

**Content**:
- **Live Demo** of Flask/Dash application
  - Input: User enters review text
  - Output:
    - Predicted sentiment (pos/neu/neg)
    - Predicted score (0-10)
    - Top influential words (TF-IDF weights)
    - Confidence scores

**Visuals**:
- Screenshot of dashboard interface
- Live demo (if possible) or recorded GIF

**Example**:
- Input: "This movie was absolutely brilliant! The cinematography was stunning..."
- Output: Sentiment = Positive (95%), Predicted Score = 8.5

---

### Slide 10: Results & Impact (1 minute)
**Speaker**: Team Member 2

**Content**:
- **Quantitative Results**:
  - ✓ Sentiment classification: XX% accuracy
  - ✓ Score prediction: R² = 0.XX, RMSE = X.XX
  - ✓ Clustering: Silhouette score = 0.XX
  - ✓ Statistical significance: All tests p < 0.05

- **Business Impact**:
  - **Streaming platforms**: Early sentiment detection around release
  - **Studios**: Data-driven post-mortems, identify red flags
  - **Audiences**: Transparent aggregated opinions

**Visuals**:
- Metrics summary table
- Impact diagram (stakeholders → benefits)

---

### Slide 11: Challenges & Limitations (1 minute)
**Speaker**: Team Member 3

**Content**:
- **Challenges Faced**:
  - API rate limits → implemented caching & delays
  - Multilingual reviews → language detection & filtering
  - Imbalanced sentiment classes → stratified sampling

- **Limitations**:
  - **Bias**: Reviews from specific platforms (TMDb)
  - **Representativeness**: Not all audiences leave reviews
  - **Temporal drift**: Sentiment may change over time

- **Mitigation**:
  - Documented bias sources
  - Statistical validation (p-values)
  - Reproducible pipeline

**Visuals**:
- Challenges → Solutions table

---

### Slide 12: Conclusions & Future Work (1 minute)
**Speaker**: Team Member 1

**Content**:
- **Key Achievements**:
  - End-to-end pipeline: API → DB → NLP → ML → Insights
  - Statistical rigor: Chi², ANOVA, correlation tests
  - Actionable insights for multiple stakeholders

- **Future Enhancements**:
  - **Deep Learning**: BERT/Transformers for sentiment
  - **Real-time analysis**: Streaming pipeline (Kafka, Spark)
  - **Multi-language support**: Translate reviews
  - **Recommendation system**: Content-based filtering using clusters

**Visuals**:
- Future roadmap diagram

---

### Slide 13: Bonus Points Checklist (30 seconds)
**Speaker**: Team Member 2

**Content** (Appendix slide for grading):
- ✓ **PostgreSQL**: Schema, views, indexes
- ✓ **Geodata**: Choropleth maps by country
- ✓ **Statistical Tests**: Chi² (p < 0.05), ANOVA (p < 0.01), Correlation (p < 0.05)
- ✓ **K-means**: Elbow plot, Silhouette score
- ✓ **Regression**: R², RMSE, residual plots
- ✓ **Classification**: Accuracy, precision, recall, confusion matrix

**Visuals**:
- Checklist with checkmarks
- Screenshots of key outputs

---

### Slide 14: Q&A (Remaining time)
**Title**: Questions?

**Visual**: Contact information or thank you message

---

## Video Recording Guidelines

### Technical Setup
- **Duration**: 15 minutes (strict)
- **Format**: MP4, 1080p minimum
- **Audio**: Clear, no background noise
- **Slides**: Share screen with slides + occasional live demo

### Presentation Tips
1. **Rehearse**: Practice transitions between speakers
2. **Timing**: Use timer, allocate ~5 min per speaker
3. **Engagement**: Speak clearly, maintain energy
4. **Visuals**: Use animations sparingly, focus on clarity
5. **Demo**: Pre-record dashboard demo if live risky

### Delivery Strategy
- **Speaker 1** (5 min): Intro, Motivation, Scope, ML Models, Demo
- **Speaker 2** (5 min): Data Pipeline, Clustering, Results, Bonus Checklist
- **Speaker 3** (5 min): EDA, Statistical Tests, Geo-Insights, Challenges, Conclusions

---

## Files to Prepare

### For Moodle Upload
1. **presentation_group_XX.pdf** - Slide deck (PDF export)
2. **video_recording_group_XX.mp4** - 15-minute video
3. **materials_group_XX.zip** - All code, notebooks, data

### Appendix Materials
- `appendix_bonus_points.pdf` - Screenshots proving:
  - PostgreSQL schema (`\dt` output)
  - Elbow plot with code
  - Silhouette scores
  - Statistical test outputs (p-values highlighted)
  - Choropleth maps
  - Confusion matrix
  - Residual plots

---

## Key Messages

### Main Point
**"MovieMind transforms unstructured review data into actionable insights through a rigorous, reproducible end-to-end analytics pipeline."**

### Supporting Points
1. **Rigor**: Statistical tests with p-values, not just correlations
2. **Comprehensiveness**: API → DB → NLP → ML → Viz
3. **Practical**: Stakeholder-specific value (platforms, studios, audiences)
4. **Reproducible**: Documented code, README, requirements.txt

---

## Visual Design Guidelines
- **Color scheme**: Consistent (e.g., blue for data, green for ML, red for insights)
- **Fonts**: Sans-serif, minimum 18pt for body text
- **Charts**: Clean, labeled axes, legends
- **Code**: Syntax-highlighted, concise snippets only
- **Logos**: TMDb, PostgreSQL, Python/scikit-learn (if allowed)

---

## Backup Slides (Optional, not in main flow)
- Detailed API endpoints
- Full SQL schema
- Hyperparameter tuning results
- Additional statistical tests
- Error analysis (misclassified examples)
