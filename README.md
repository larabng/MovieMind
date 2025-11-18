# MovieMind - End-to-End Movie Review Analytics System

## Overview
MovieMind is a comprehensive analytics system that collects movie reviews and metadata via APIs, processes them using NLP and ML techniques to generate insights about sentiment, score predictions, and audience patterns.

## Features
- **Sentiment Analysis**: Classify reviews as positive/neutral/negative
- **Score Prediction**: Predict movie ratings (0-10 scale)
- **Driver Analysis**: Identify text patterns that drive good/bad reviews
- **Clustering**: Find similar movies and audience reactions
- **Geo-Visualization**: Sentiment analysis by production country
- **Interactive Dashboard**: Live demo with review text input

## Project Structure
```
MovieMind/
├── src/                    # Source code
│   ├── data_collection/    # API data collection scripts
│   ├── preprocessing/      # Data cleaning and NLP preprocessing
│   ├── models/            # ML models (classification & regression)
│   ├── evaluation/        # Model evaluation and metrics
│   └── utils/             # Helper functions
├── notebooks/             # Jupyter notebooks for EDA
├── sql/                   # PostgreSQL schema and queries
├── dashboards/            # Flask/Dash web application
├── data/                  # Data storage (gitignored)
├── tests/                 # Unit tests
└── requirements.txt       # Python dependencies
```

## Setup Instructions

### 1. Prerequisites
- Python 3.9+
- PostgreSQL 14+
- TMDb API Key (get from https://www.themoviedb.org/settings/api)

### 2. Installation
```bash
# Clone repository
git clone <your-repo-url>
cd MovieMind

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
```bash
# Copy environment template
cp .env.sample .env

# Edit .env and add your API keys
```

### 4. Database Setup
```bash
# Create PostgreSQL database
createdb moviemind

# Run schema
psql -d moviemind -f sql/schema.sql
```

### 5. Data Collection
```bash
# Collect movie data from TMDb API
python src/data_collection/fetch_movies.py

# Collect reviews
python src/data_collection/fetch_reviews.py
```

### 6. Run Analysis
```bash
# Run EDA notebook
jupyter notebook notebooks/01_exploratory_analysis.ipynb

# Train models
python src/models/train_classifier.py
python src/models/train_regressor.py
```

### 7. Launch Dashboard
```bash
python dashboards/app.py
# Open browser to http://localhost:5000
```

## Key Results
- **Classification**: >80% accuracy on sentiment analysis
- **Regression**: R² and RMSE metrics for score prediction
- **Insights**: Genre-specific patterns, runtime correlations
- **Statistical Tests**: ANOVA, Chi², correlation analyses with p-values

## Team & Roles
- **API & DB Lead**: Data collection, ETL, PostgreSQL management
- **NLP & EDA Lead**: Text preprocessing, exploratory analysis
- **Modeling & Storytelling Lead**: ML models, demo app, presentation

## Technologies Used
- **Languages**: Python 3.9+
- **Database**: PostgreSQL
- **Libraries**:
  - Data: pandas, numpy, psycopg2
  - NLP: nltk, scikit-learn (TF-IDF)
  - ML: scikit-learn (LogReg, RandomForest)
  - Viz: matplotlib, seaborn, plotly
  - Web: Flask/Dash
- **APIs**: TMDb API

## License
MIT

## Acknowledgments
This project was developed as part of a data science course focusing on NLP and end-to-end analytics pipelines.
