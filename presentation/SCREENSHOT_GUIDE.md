# Screenshot Guide for Appendix
## MovieMind Presentation - Points Documentation

Diese Datei zeigt genau, welche Code-Stellen und Outputs du fuer Screenshots verwenden sollst.

---

## 1. PostgreSQL Schema (Bonus: Database Design)

### Screenshot 1.1: Tabellen-Definition
**Datei:** `sql/schema.sql`
**Zeilen:** 10-88

```sql
-- Zeige diese Teile:
CREATE TABLE movies (
    movie_id INTEGER PRIMARY KEY,
    title VARCHAR(500) NOT NULL,
    genres TEXT[],  -- Array type!
    vote_average DECIMAL(3,1),
    ...
);

CREATE TABLE reviews (
    review_id SERIAL PRIMARY KEY,
    movie_id INTEGER REFERENCES movies(movie_id),
    content TEXT NOT NULL,
    sentiment VARCHAR(20),
    predicted_rating DECIMAL(3,1),
    ...
);
```

### Screenshot 1.2: Indexes
**Datei:** `sql/schema.sql`
**Zeilen:** 91-96

```sql
CREATE INDEX idx_movies_genres ON movies USING GIN(genres);
CREATE INDEX idx_reviews_sentiment ON reviews(sentiment);
```

### Screenshot 1.3: SQL Views
**Datei:** `sql/schema.sql`
**Zeilen:** 99-148

```sql
CREATE VIEW movie_review_stats AS ...
CREATE VIEW genre_sentiment_analysis AS ...
CREATE VIEW temporal_sentiment_trends AS ...
```

---

## 2. Statistical Tests with P-Values

### Screenshot 2.1: Chi-Squared Test
**Datei:** `notebooks/01_exploratory_analysis.ipynb`
**Cell:** 22

```python
chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
print(f"Chi² = {chi2:.4f}")
print(f"p-value = {p_value:.4f}")
print(f"Significant: {'Yes' if p_value < 0.05 else 'No'}")
```

**Output zeigen:**
```
Drama:
  Chi² = 12.3456
  p-value = 0.0004
  Significant: Yes
```

### Screenshot 2.2: ANOVA Test
**Datei:** `notebooks/01_exploratory_analysis.ipynb`
**Cell:** 23

```python
f_stat, p_value = stats.f_oneway(*genre_groups)
print(f"F-statistic: {f_stat:.4f}")
print(f"p-value: {p_value:.4f}")
```

### Screenshot 2.3: Pearson Correlation
**Datei:** `notebooks/01_exploratory_analysis.ipynb`
**Cell:** 24

```python
corr, p_value = stats.pearsonr(valid_data['runtime'], valid_data['vote_average'])
print(f"Correlation: {corr:.4f}")
print(f"p-value: {p_value:.4f}")
```

---

## 3. K-Means Clustering

### Screenshot 3.1: Elbow Method
**Datei:** `src/models/clustering.py`
**Zeilen:** 144-177

```python
def elbow_analysis(self, X: np.ndarray, max_k: int = 10):
    for k in range(1, max_k + 1):
        kmeans_temp = KMeans(n_clusters=k, ...)
        metrics['silhouette'] = silhouette_score(X, kmeans_temp.labels_)
```

### Screenshot 3.2: Elbow Plot Output
**Datei:** `notebooks/02_model_training.ipynb`
**Cell:** 22

Zeige den generierten Plot mit:
- Inertia-Kurve (Elbow)
- Silhouette Score Kurve

### Screenshot 3.3: Silhouette Score
**Datei:** `notebooks/02_model_training.ipynb`
**Cell:** 23

```python
print(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
print(f"Davies-Bouldin Score: {metrics['davies_bouldin_score']:.4f}")
```

### Screenshot 3.4: PCA Visualization
**Datei:** `src/models/clustering.py`
**Zeilen:** 206-241

```python
def visualize_clusters_2d(self, X, labels):
    X_pca = self.pca.fit_transform(X)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
```

Zeige den 2D-Cluster-Plot mit Centroids.

---

## 4. Regression with Diagnostics

### Screenshot 4.1: Ridge Regression Setup
**Datei:** `src/models/score_predictor.py`
**Zeilen:** 29-68

```python
class ScorePredictor:
    def __init__(self, model_type: str = 'ridge', max_features: int = 3000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            ...
        )
        self.model = Ridge(alpha=1.0, random_state=42)
```

### Screenshot 4.2: Evaluation Metrics
**Datei:** `notebooks/02_model_training.ipynb`
**Cell:** 17

```python
# Output:
Test RMSE: 1.2345
Test MAE: 0.9876
Test R²: 0.6543
```

### Screenshot 4.3: Residual Plots
**Datei:** `notebooks/02_model_training.ipynb`
**Cell:** 18

Zeige beide Plots:
1. **Residual Plot:** Predicted vs. Residuals (sollte um 0 zentriert sein)
2. **Predicted vs. Actual:** Scatter mit Diagonale

---

## 5. Confusion Matrix

### Screenshot 5.1: Classification Code
**Datei:** `src/models/sentiment_classifier.py`
**Zeilen:** 26-68

```python
class SentimentClassifier:
    def __init__(self, model_type: str = 'logistic', max_features: int = 5000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            ...
        )
        self.model = LogisticRegression(
            class_weight='balanced',
            ...
        )
```

### Screenshot 5.2: Confusion Matrix Heatmap
**Datei:** `notebooks/02_model_training.ipynb`
**Cell:** 12

Zeige die Confusion Matrix Visualization:
```python
cm_display = ConfusionMatrixDisplay(
    confusion_matrix=test_metrics['confusion_matrix'],
    display_labels=classifier.model.classes_
)
cm_display.plot(cmap='Blues')
```

### Screenshot 5.3: Classification Report
**Datei:** `notebooks/02_model_training.ipynb`
**Cell:** 11

```
Classification Report:
              precision    recall  f1-score   support
    negative       0.XX      0.XX      0.XX        XX
     neutral       0.XX      0.XX      0.XX        XX
    positive       0.XX      0.XX      0.XX        XX
    accuracy                           0.XX       XXX
```

---

## 6. Additional Visualizations (Optional)

### Screenshot 6.1: Correlation Heatmap
**Datei:** `notebooks/01_exploratory_analysis.ipynb`
**Cell:** 12

### Screenshot 6.2: Rating Distribution
**Datei:** `notebooks/01_exploratory_analysis.ipynb`
**Cell:** 8

### Screenshot 6.3: Genre Boxplot
**Datei:** `notebooks/01_exploratory_analysis.ipynb`
**Cell:** 23

### Screenshot 6.4: Feature Importance
**Datei:** `notebooks/02_model_training.ipynb`
**Cell:** 13 oder 19

---

## 7. Dashboard (Live Demo)

### Screenshot 7.1: Main Interface
**Datei:** `dashboards/app.py`
Starte mit `python dashboards/app.py` und mache Screenshot von:
- Live Prediction Tab mit Eingabefeld
- Ergebnis nach Analyse

### Screenshot 7.2: Database Stats Tab
Zeige die Statistik-Karten (Movie Count, Review Count, Avg Rating)

---

## Tipps fuer Screenshots:

1. **Code-Screenshots:**
   - Verwende VS Code mit Dark Theme
   - Zeige Zeilennummern
   - Highlighte wichtige Teile

2. **Output-Screenshots:**
   - Jupyter Notebook Cells mit Output
   - Terminal-Output wenn sinnvoll

3. **Plot-Screenshots:**
   - Hohe Aufloesung
   - Achsenbeschriftungen sichtbar
   - Legende wenn vorhanden

4. **Format:**
   - Einheitliche Groesse (z.B. 1200x800)
   - Klare Beschriftung
   - Nicht zu viel auf einmal

---

## Checkliste fuer Appendix:

- [ ] PostgreSQL Schema (Tables, Indexes, Views)
- [ ] Chi-squared Test mit p-value
- [ ] ANOVA mit p-value
- [ ] Pearson Correlation mit p-value
- [ ] Elbow Plot
- [ ] Silhouette Score
- [ ] Cluster Visualization
- [ ] Regression Metrics (R², RMSE, MAE)
- [ ] Residual Plots
- [ ] Confusion Matrix
- [ ] Classification Report
- [ ] Dashboard Screenshot
