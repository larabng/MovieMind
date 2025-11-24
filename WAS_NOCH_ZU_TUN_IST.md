# MovieMind - Was noch zu tun ist

**Stand**: 18. November 2025
**Abgabe**: 11. Januar 2026

---

## ✅ BEREITS FERTIG

### Code & Infrastruktur (100% komplett)
- ✅ **Projektstruktur**: Alle Ordner und Module erstellt
- ✅ **PostgreSQL Schema**: Tabellen (`movies`, `reviews`, `countries`), Views, Indexes
- ✅ **API Client**: TMDb-Integration mit Rate-Limiting
- ✅ **Text-Preprocessing**: HTML-Cleanup, Stopwörter, Lemmatisierung (NLTK)
- ✅ **Sentiment-Classifier**: TF-IDF + LogReg/RF, alle Metriken
- ✅ **Score-Predictor**: Ridge/Lasso Regression mit Meta-Features
- ✅ **Clustering**: K-means mit Elbow, Silhouette
- ✅ **Evaluation-Skript**: Confusion Matrix, Residuenplots, p-Werte
- ✅ **Dashboard**: Flask/Dash Template

### Notebooks (100% komplett)
- ✅ `01_exploratory_analysis.ipynb` - EDA mit Chi², ANOVA, Korrelation
- ✅ `02_model_training.ipynb` - Model Training
- ✅ `03_clustering_analysis.ipynb` - K-means, Elbow, Silhouette
- ✅ `04_geo_visualization.ipynb` - Choropleth-Karten nach Land

### Dokumentation (100% komplett)
- ✅ `README.md` - Projekt-Übersicht
- ✅ `QUICKSTART.md` - Detaillierte Setup-Anleitung
- ✅ `PRESENTATION_OUTLINE.md` - Vollständiger Präsentationsleitfaden
- ✅ `requirements.txt` - Alle Dependencies
- ✅ `.env.sample` - Konfigurations-Template
- ✅ `setup_project.py` - Automatisches Setup-Skript

---

## 🔴 WAS DU NOCH MACHEN MUSST

### 1. SETUP & KONFIGURATION (30-60 Minuten)

#### A. API-Keys einrichten
```bash
#gemacht
# 1. TMDb API Key besorgen
# - Gehe zu: https://www.themoviedb.org/settings/api
# - Account erstellen (falls nicht vorhanden)
# - API Key beantragen (Developer Section)
# - Kopiere den API Key (v3 auth)

# 2. .env Datei konfigurieren
cp .env.sample .env
# Editiere .env und füge deinen API Key ein:
TMDB_API_KEY=dein_echter_api_key_hier
```

#### B. PostgreSQL einrichten
```bash
# 1. PostgreSQL installieren (falls noch nicht)
# - Windows: https://www.postgresql.org/download/windows/
# - Installiere mit Standardeinstellungen

# 2. Datenbank erstellen
psql -U postgres
CREATE DATABASE moviemind;
\q

# 3. Schema initialisieren
psql -U postgres -d moviemind -f sql/schema.sql

# ODER automatisch via Setup-Skript:
python setup_project.py
```

#### C. Python-Umgebung
```bash
# Virtual Environment erstellen
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Dependencies installieren
pip install -r requirements.txt

# NLTK Daten herunterladen (automatisch beim ersten Start)
```

---

### 2. DATENSAMMLUNG (2-4 Stunden)

**WICHTIG**: Dies ist der zeitaufwendigste Schritt!

```bash
# Aktiviere Virtual Environment
venv\Scripts\activate

# Sammle 500-1000 Filme + Reviews
python src/data_collection/fetch_movies.py --movies 500 --strategy mixed

# Hinweise:
# - API hat Rate-Limits (ca. 40 Requests/10 Sekunden)
# - Das Skript hat automatische Delays eingebaut
# - Für 500 Filme + Reviews: ca. 2-3 Stunden
# - Läuft im Hintergrund, du kannst währenddessen anderes machen

# Überprüfe Datensammlung
psql -U postgres -d moviemind
SELECT COUNT(*) FROM movies;   -- Sollte ~500 sein
SELECT COUNT(*) FROM reviews;  -- Sollte >5000 sein
\q
```

**Strategie-Optionen**:
- `--strategy popular`: Nur populäre Filme
- `--strategy top_rated`: Nur top-bewertete Filme
- `--strategy mixed`: Mix aus beidem (empfohlen!)

---

### 3. MODELLTRAINING (1-2 Stunden)

```bash
# Nachdem Daten gesammelt sind:

# 1. EDA-Notebook ausführen (optional, aber hilfreich)
jupyter notebook notebooks/01_exploratory_analysis.ipynb
# Führe alle Zellen aus, speichere Plots für Präsentation

# 2. Modelle trainieren
python src/models/train_models.py

# Das trainiert:
# - Sentiment Classifier (TF-IDF + LogReg)
# - Score Predictor (Ridge Regression)
# Speichert Modelle in models/

# 3. Modelle evaluieren
python src/models/evaluate_models.py

# Erstellt:
# - evaluation_results/confusion_matrix_sentiment.png
# - evaluation_results/regression_evaluation.png
```

---

### 4. ERWEITERTE ANALYSEN (2-3 Stunden)

```bash
# Führe alle Notebooks aus (in Reihenfolge):

jupyter notebook

# Öffne und führe aus:
# 1. notebooks/01_exploratory_analysis.ipynb
#    → Speichere wichtige Plots (Rating-Verteilung, Korrelation)
#
# 2. notebooks/03_clustering_analysis.ipynb
#    → Speichere Elbow-Plot, Silhouette-Plot, PCA-Viz
#
# 3. notebooks/04_geo_visualization.ipynb
#    → Speichere Choropleth-Karten (HTML interaktiv)

# Screenshots für Präsentation:
# - Speichere alle wichtigen Plots in presentation/screenshots/
```

---

### 5. PRÄSENTATION ERSTELLEN (4-6 Stunden)

#### A. Folien erstellen (3-4 Stunden)

Nutze `PRESENTATION_OUTLINE.md` als Vorlage!

**Empfohlene Tools**:
- Google Slides / PowerPoint
- LaTeX Beamer (für akademische Optik)
- Canva (für schöne Grafiken)

**Struktur** (siehe PRESENTATION_OUTLINE.md):
1. **Title Slide**: Projekt-Name, Team, Datum
2. **Intro & Motivation**: Problem, Lösung, Wert
3. **Scope**: Datenquellen, Ziele
4. **Methodology**: API → DB → NLP → ML
5. **EDA**: Plots, Chi², ANOVA, Korrelation
6. **ML Models**: Classifier + Regressor Metriken
7. **Clustering**: Elbow, Silhouette, Interpretation
8. **Geo-Insights**: Choropleth-Karten
9. **Demo**: Dashboard-Screenshot oder Live
10. **Results**: Metriken-Zusammenfassung
11. **Challenges**: API-Limits, Bias, Lösungen
12. **Conclusions**: Key Achievements, Future Work
13. **Bonus Checklist**: PostgreSQL ✓, Chi² ✓, k-means ✓...
14. **Q&A**

**Wichtige Inhalte**:
- **Screenshots**:
  - Database schema (`psql -d moviemind` → `\dt`)
  - Confusion Matrix
  - Residuenplots
  - Elbow-Plot
  - Choropleth-Karte
  - Dashboard-Demo

- **Metriken** (fülle mit echten Werten):
  - Sentiment Accuracy: ___%
  - Score R²: ___
  - Silhouette Score: ___
  - Chi² p-value: ___
  - ANOVA p-value: ___

#### B. Bonus-Anhang erstellen (1 Stunde)

Erstelle `appendix_bonus_points.pdf` mit:

1. **PostgreSQL-Nachweis**:
   - Screenshot `\dt` (Tabellen-Liste)
   - Screenshot SQL-View-Code
   - Screenshot Query-Ergebnisse

2. **Geodaten**:
   - Screenshot Choropleth-Karte (Sentiment by Country)
   - Code-Snippet: plotly.express.choropleth()

3. **Statistische Tests**:
   - Chi²-Test Output (mit **p-value < 0.05** hervorgehoben)
   - ANOVA Output (mit **p-value** und **F-statistic**)
   - Korrelation Output (Pearson r, **p-value**)

4. **K-means**:
   - Elbow-Plot (mit optimalem k markiert)
   - Silhouette-Score-Tabelle
   - Cluster-Interpretation

5. **Klassifikation & Regression**:
   - Confusion Matrix (mit Accuracy %)
   - Precision/Recall/F1 Tabelle
   - R²/RMSE/MAE Tabelle
   - Residuenplot

---

### 6. VIDEO-AUFNAHME (2-3 Stunden)

#### Vorbereitung (30 Min)
- [ ] Folien finalisieren (PDF exportieren)
- [ ] Skript/Stichwörter für jeden Sprecher vorbereiten
- [ ] Timer bereitstellen (15 Min genau!)
- [ ] Screen-Recording-Software testen (OBS, Zoom, PowerPoint)

#### Probe-Durchlauf (30 Min)
- [ ] Timing checken (3 Personen × 5 Min)
- [ ] Übergänge zwischen Sprechern üben
- [ ] Dashboard-Demo proben (oder Pre-Record)

#### Finale Aufnahme (1-2 Stunden)
- [ ] Aufnahme starten
- [ ] Präsentation durchführen (15 Min exakt)
- [ ] Video exportieren als MP4 (1080p)
- [ ] Qualitäts-Check (Audio klar? Folien lesbar?)

**Technik-Tipps**:
- Gutes Mikrofon verwenden
- Ruhige Umgebung
- Bildschirm teilen (Folien im Vollbild)
- Wenn Demo-Risiko: Pre-Record Dashboard, füge Screenshot ein

---

### 7. FINALE DELIVERABLES (1 Stunde)

#### Dateien vorbereiten:

```bash
# 1. PDF erstellen
# - Exportiere Folien als: presentation_group_XX.pdf

# 2. Video benennen
# - Benenne Video: video_recording_group_XX.mp4

# 3. Materials ZIP erstellen
# Packe alles in ein ZIP (OHNE data/, models/, venv/):

# Windows:
# - Rechtsklick auf MovieMind-Ordner → Senden an → ZIP
# - Oder nutze 7-Zip

# Beinhalten sollte:
materials_group_XX.zip
  ├── src/               # Alle Python-Skripte
  ├── notebooks/         # Alle Jupyter Notebooks (.ipynb)
  ├── sql/               # schema.sql
  ├── dashboards/        # Dashboard-Code
  ├── README.md
  ├── QUICKSTART.md
  ├── requirements.txt
  ├── .env.sample
  ├── setup_project.py
  └── appendix_bonus_points.pdf

# NICHT einpacken:
# - data/ (zu groß)
# - models/ (zu groß)
# - venv/ (nicht nötig)
# - .git/ (nicht nötig)
# - __pycache__/ (automatisch generiert)
```

#### Upload zu Moodle:

- [ ] `presentation_group_XX.pdf`
- [ ] `video_recording_group_XX.mp4`
- [ ] `materials_group_XX.zip`

**Deadline**: Sonntag, 11. Januar 2026

---

## 📅 ZEITPLAN (Empfehlung)

### Woche 1 (18.-24. Nov): Setup & Datensammlung
- **Tag 1-2**: API-Keys, PostgreSQL, Setup (2 Std)
- **Tag 3-5**: Datensammlung laufen lassen (Hintergrund, 3 Std Arbeit)
- **Tag 6-7**: EDA-Notebook ausführen, erste Plots (2 Std)

### Woche 2-4 (25. Nov - 15. Dez): Modelle & Analysen
- **Woche 2**: Modelltraining, Evaluation (4 Std)
- **Woche 3**: Clustering, Geo-Viz Notebooks (4 Std)
- **Woche 4**: Dashboard testen, Screenshots sammeln (2 Std)

### Woche 5-7 (16. Dez - 05. Jan): Präsentation
- **16.-22. Dez**: Folien erstellen (6 Std)
- **23.-29. Dez**: Bonus-Anhang, Metriken einfügen (3 Std)
- **30. Dez - 05. Jan**: Probe-Durchläufe, Video-Aufnahme (4 Std)

### Woche 8 (06.-11. Jan): Finalisierung
- **06.-08. Jan**: Letzte Korrekturen, Qualitäts-Check
- **09.-10. Jan**: ZIP erstellen, Moodle-Upload vorbereiten
- **11. Jan**: UPLOAD (vor Deadline!)

---

## 🎯 PRIORITÄTEN

### MUST HAVE (kritisch für Bestehen):
1. ✅ Datensammlung (500+ Filme, 5000+ Reviews)
2. ✅ PostgreSQL-Schema funktioniert
3. ✅ Sentiment-Classifier trainiert (Accuracy >80%)
4. ✅ Score-Predictor trainiert (R² >0.5)
5. ✅ EDA mit statistischen Tests (Chi², ANOVA, Korrelation)
6. ✅ Präsentation (15 Min, PDF + Video)
7. ✅ Materials ZIP (vollständig, lauffähig)

### NICE TO HAVE (Bonus-Punkte):
- ✅ K-means Clustering
- ✅ Geo-Visualisierung (Choropleth)
- ✅ Interaktives Dashboard
- ✅ Gut dokumentierter Code
- ✅ README mit Reproduzierbarkeit

---

## 🆘 TROUBLESHOOTING

### Problem: API-Key funktioniert nicht
**Lösung**:
```bash
# Test API-Key:
python src/data_collection/tmdb_client.py
# Sollte "Found X popular movies" zeigen
```

### Problem: PostgreSQL Verbindungsfehler
**Lösung**:
```bash
# Starte PostgreSQL (Windows):
net start postgresql-x64-14

# Überprüfe .env Credentials:
DB_HOST=localhost
DB_USER=postgres
DB_PASSWORD=dein_postgres_passwort
```

### Problem: "No module named 'sklearn'"
**Lösung**:
```bash
venv\Scripts\activate
pip install -r requirements.txt
```

### Problem: NLTK Daten fehlen
**Lösung**:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## 📞 NÄCHSTE SCHRITTE (SOFORT)

1. **JETZT MACHEN** (10 Min):
   ```bash
   # Setup ausführen
   python setup_project.py
   ```

2. **HEUTE** (30 Min):
   - TMDb API Key besorgen
   - PostgreSQL installieren (falls nötig)
   - .env konfigurieren

3. **DIESE WOCHE** (3 Std):
   - Datensammlung starten und laufen lassen
   - EDA-Notebook durchgehen

4. **NÄCHSTE WOCHE**:
   - Modelle trainieren
   - Erste Präsentations-Skizze

---

## ✅ CHECKLISTE VOR ABGABE

### Code
- [ ] Alle Notebooks ausgeführt, Outputs gespeichert
- [ ] Models trainiert und in `models/` gespeichert
- [ ] Evaluation-Plots in `evaluation_results/`
- [ ] Dashboard getestet (Screenshots gemacht)

### Dokumentation
- [ ] README.md aktuell und korrekt
- [ ] requirements.txt vollständig
- [ ] .env.sample vorhanden (OHNE echte Keys!)

### Präsentation
- [ ] PDF-Folien (14+ Slides)
- [ ] Alle Metriken eingefügt (echte Werte!)
- [ ] Screenshots eingebunden
- [ ] Bonus-Anhang erstellt

### Video
- [ ] 15 Minuten exakt
- [ ] Gute Audio-Qualität
- [ ] Folien lesbar
- [ ] MP4-Format, <500 MB

### Upload
- [ ] Dateinamen korrekt: `presentation_group_XX.pdf`
- [ ] Video: `video_recording_group_XX.mp4`
- [ ] ZIP: `materials_group_XX.zip`
- [ ] Moodle-Upload VOR Deadline

---

**VIEL ERFOLG! 🎬📊🚀**

Bei Fragen: Schau in `QUICKSTART.md` oder `README.md`!
