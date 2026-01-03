# 📸 Screenshot Checklist für MovieMind Präsentation

## Status: Notebooks werden gerade ausgeführt...
Die Notebooks werden mit Outputs gespeichert. Danach kannst du die Screenshots machen.

---

## PRIORITÄT 1: Code-Screenshots (JETZT möglich)

### 1️⃣ PostgreSQL Schema
📁 Datei: `sql/schema.sql`

#### Screenshot 1.1: Table Definitions
- **Zeilen:** 18-88
- **Was zeigen:** Movies & Reviews Tables mit:
  - `genres TEXT[]` (Array type)
  - Foreign Keys
  - Constraints

#### Screenshot 1.2: Indexes
- **Zeilen:** 91-96
- **Was zeigen:** GIN Index für genres, Index für sentiment

#### Screenshot 1.3: SQL Views
- **Zeilen:** 99-148
- **Was zeigen:** movie_review_stats, genre_sentiment_analysis, temporal_sentiment_trends

---

### 2️⃣ Machine Learning Code

#### Screenshot 2.1: K-Means Clustering
📁 Datei: `src/models/clustering.py`
- **Zeilen:** 144-177 (elbow_analysis Methode)
- **Highlight:** silhouette_score, KMeans setup

#### Screenshot 2.2: Ridge Regression
📁 Datei: `src/models/score_predictor.py`
- **Zeilen:** 29-68
- **Highlight:** TfidfVectorizer, Ridge model mit alpha=1.0

#### Screenshot 2.3: Sentiment Classification
📁 Datei: `src/models/sentiment_classifier.py`
- **Zeilen:** 26-68
- **Highlight:** LogisticRegression, class_weight='balanced'

---

## PRIORITÄT 2: Notebook Outputs (NACH Ausführung)

### 3️⃣ Statistical Tests
📁 Datei: `notebooks/01_exploratory_analysis.ipynb`

#### Screenshot 3.1: Chi-Squared Test
- **Cell:** 22
- **Output zeigen:** Chi² Wert, p-value, "Significant: Yes/No"

#### Screenshot 3.2: ANOVA Test
- **Cell:** 23
- **Output zeigen:** F-statistic, p-value

#### Screenshot 3.3: Pearson Correlation
- **Cell:** 24
- **Output zeigen:** Correlation coefficient, p-value

---

### 4️⃣ Clustering Visualizations
📁 Datei: `notebooks/02_model_training.ipynb` oder `03_clustering_analysis.ipynb`

#### Screenshot 4.1: Elbow Plot
- **Was zeigen:**
  - Inertia-Kurve (zeigt "Elbow")
  - Silhouette Score Kurve
  - Beide zusammen in einem Plot

#### Screenshot 4.2: Cluster Metrics
- **Output zeigen:**
  ```
  Silhouette Score: 0.XXXX
  Davies-Bouldin Score: 0.XXXX
  Optimal K: X
  ```

#### Screenshot 4.3: PCA Visualization
- **Was zeigen:** 2D Scatter Plot mit:
  - Verschiedene Farben für Cluster
  - Cluster Centroids markiert
  - Achsenbeschriftungen (PC1, PC2)

---

### 5️⃣ Regression Diagnostics
📁 Datei: `notebooks/02_model_training.ipynb`

#### Screenshot 5.1: Evaluation Metrics
- **Cell:** ~17
- **Output zeigen:**
  ```
  Test RMSE: X.XXXX
  Test MAE: X.XXXX
  Test R²: X.XXXX
  ```

#### Screenshot 5.2: Residual Plots
- **Cell:** ~18
- **Beide Plots zeigen:**
  1. Predicted vs Residuals (sollte um 0 zentriert sein)
  2. Predicted vs Actual (mit Diagonale)

---

### 6️⃣ Classification Results
📁 Datei: `notebooks/02_model_training.ipynb`

#### Screenshot 6.1: Confusion Matrix
- **Cell:** ~12
- **Heatmap zeigen:** 3x3 Matrix (negative, neutral, positive)
- Mit Farbskala (Blues)

#### Screenshot 6.2: Classification Report
- **Cell:** ~11
- **Tabelle zeigen:**
  ```
              precision    recall  f1-score   support
    negative       0.XX      0.XX      0.XX        XX
     neutral       0.XX      0.XX      0.XX        XX
    positive       0.XX      0.XX      0.XX        XX
    accuracy                           0.XX       XXX
  ```

---

### 7️⃣ Additional Visualizations (Optional aber gut)

📁 Datei: `notebooks/01_exploratory_analysis.ipynb`

#### Screenshot 7.1: Correlation Heatmap
- **Cell:** ~12
- Features: vote_average, runtime, budget, revenue

#### Screenshot 7.2: Rating Distribution
- **Cell:** ~8
- Histogram der vote_average

#### Screenshot 7.3: Genre Boxplot
- **Cell:** ~23
- Ratings pro Genre

---

## PRIORITÄT 3: Dashboard (Live Demo)

### 8️⃣ Dashboard Screenshots

#### Schritt 1: Dashboard starten
```bash
python dashboards/app.py
```

#### Screenshot 8.1: Main Interface
- **Tab:** "Live Prediction"
- **Zeigen:**
  - Textfeld mit Beispiel-Review
  - "Analyze Review" Button
  - Prediction Result (Sentiment + Rating)

#### Screenshot 8.2: Database Stats
- **Tab:** "Database Statistics"
- **Zeigen:**
  - Movie Count Card
  - Review Count Card
  - Average Rating Card
  - Top Genres

---

## 📋 SCREENSHOT TIPS

### Qualität:
- ✅ **Auflösung:** 1200x800 oder größer
- ✅ **Theme:** VS Code Dark Theme (konsistent)
- ✅ **Font:** Groß genug lesbar (Zoom in wenn nötig)
- ✅ **Zeilennummern:** Aktiviert in VS Code

### Was highlighten:
- 🔹 Wichtige Code-Zeilen (z.B. model definition)
- 🔹 P-values < 0.05 (statistisch signifikant)
- 🔹 High metrics (R² > 0.6, Accuracy > 0.75)

### Was vermeiden:
- ❌ Nicht zu viel Code auf einmal
- ❌ Unscharfe Bilder
- ❌ Unleserliche Achsenbeschriftungen
- ❌ Verschiedene Themes mischen

---

## 🎯 MINIMUM für Präsentation

**Must-have Screenshots (10-12):**
1. ✅ PostgreSQL Tables (1 Screenshot)
2. ✅ Chi-Squared Test Output (1 Screenshot)
3. ✅ ANOVA Output (1 Screenshot)
4. ✅ Elbow Plot (1 Screenshot)
5. ✅ Cluster Visualization (1 Screenshot)
6. ✅ Regression Metrics (1 Screenshot)
7. ✅ Residual Plot (1 Screenshot)
8. ✅ Confusion Matrix (1 Screenshot)
9. ✅ Classification Report (1 Screenshot)
10. ✅ Dashboard Main (1 Screenshot)

**Nice-to-have (2-4):**
- Correlation Heatmap
- Genre Boxplot
- Feature Importance
- Database Stats

---

## ✅ CHECKLISTE

Nach dem die Notebooks ausgeführt wurden:

- [ ] Alle Notebooks haben Outputs
- [ ] Code-Screenshots gemacht (SQL, Models)
- [ ] Statistical Tests Screenshots (Chi², ANOVA, Pearson)
- [ ] Clustering Screenshots (Elbow, Silhouette, PCA)
- [ ] Regression Screenshots (Metrics, Residuals)
- [ ] Classification Screenshots (Confusion Matrix, Report)
- [ ] Dashboard Screenshots
- [ ] Alle Screenshots in `presentation/screenshots/` gespeichert
- [ ] Screenshots in PowerPoint eingefügt

---

**Status:** ⏳ Warte auf Notebook-Ausführung...
**Geschätzte Zeit:** 5-10 Minuten
