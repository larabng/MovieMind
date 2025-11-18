# MovieMind - Quick Start Guide

## Prerequisites

Before you begin, ensure you have:

- **Python 3.9+** installed
- **PostgreSQL 14+** installed and running
- **TMDb API Key** - Get one from [https://www.themoviedb.org/settings/api](https://www.themoviedb.org/settings/api)

## Step-by-Step Setup

### 1. Environment Setup

```bash
# Navigate to project directory
cd MovieMind

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
# Copy the sample environment file
copy .env.sample .env  # Windows
# cp .env.sample .env  # Mac/Linux

# Edit .env file and add your credentials
```

Edit `.env` file:
```
# API Keys
TMDB_API_KEY=your_actual_tmdb_api_key_here

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=moviemind
DB_USER=postgres
DB_PASSWORD=your_postgres_password
```

### 3. Database Setup

```bash
# Create PostgreSQL database
createdb moviemind

# Or using psql:
psql -U postgres
CREATE DATABASE moviemind;
\q

# Run schema to create tables
psql -U postgres -d moviemind -f sql/schema.sql
```

### 4. Collect Data

```bash
# Fetch movies from TMDb (this will take a few minutes)
python src/data_collection/fetch_movies.py --movies 500 --strategy mixed

# Fetch reviews for the movies
python src/data_collection/fetch_movies.py --reviews --max-reviews 30
```

**Note:** The API has rate limits. The script includes delays to respect them.

### 5. Explore Data

```bash
# Launch Jupyter Notebook
jupyter notebook

# Open and run: notebooks/01_exploratory_analysis.ipynb
```

This notebook includes:
- Univariate analysis
- Bivariate analysis
- Genre analysis
- Correlation analysis
- Statistical tests (Chi², ANOVA)

### 6. Train ML Models

```bash
# Train both sentiment classifier and score predictor
python src/models/train_models.py

# Or train with limited data for quick testing:
python src/models/train_models.py --limit 1000
```

This will:
- Train a sentiment classifier (positive/neutral/negative)
- Train a score predictor (0-10 scale)
- Save models to `models/` directory
- Print evaluation metrics

### 7. Launch Dashboard

```bash
# Start the web application
python dashboards/app.py
```

Open your browser to: [http://localhost:5000](http://localhost:5000)

The dashboard includes:
- **Live Prediction**: Analyze any review text
- **Database Statistics**: View data overview
- **Visualizations**: Interactive charts

## Common Commands

### Data Collection

```bash
# Fetch only popular movies
python src/data_collection/fetch_movies.py --strategy popular --movies 200

# Fetch only top-rated movies
python src/data_collection/fetch_movies.py --strategy top_rated --movies 200

# Fetch more reviews per movie
python src/data_collection/fetch_movies.py --reviews --max-reviews 50
```

### Model Training

```bash
# Train only classifier
python src/models/train_models.py --skip-regressor

# Train only regressor
python src/models/train_models.py --skip-classifier

# Train with limited data
python src/models/train_models.py --limit 500
```

### Database Queries

```bash
# Connect to database
psql -U postgres -d moviemind

# Check data
SELECT COUNT(*) FROM movies;
SELECT COUNT(*) FROM reviews;

# View aggregated stats
SELECT * FROM movie_review_stats LIMIT 10;

# Genre sentiment analysis
SELECT * FROM genre_sentiment_analysis;
```

## Project Structure

```
MovieMind/
├── src/
│   ├── data_collection/     # API clients and data fetching
│   │   ├── tmdb_client.py
│   │   └── fetch_movies.py
│   ├── preprocessing/       # Text preprocessing
│   │   └── text_processor.py
│   ├── models/             # ML models
│   │   ├── sentiment_classifier.py
│   │   ├── score_predictor.py
│   │   ├── clustering.py
│   │   └── train_models.py
│   └── utils/              # Helper functions
│       └── db_manager.py
├── notebooks/              # Jupyter notebooks
│   └── 01_exploratory_analysis.ipynb
├── dashboards/            # Web application
│   └── app.py
├── sql/                   # Database schema
│   └── schema.sql
├── models/               # Trained models (created after training)
└── data/                 # Data files (gitignored)
```

## Troubleshooting

### Database Connection Error

```
Error: could not connect to server
```

**Solution**: Ensure PostgreSQL is running:
```bash
# Windows (if installed as service)
net start postgresql-x64-14

# Mac
brew services start postgresql

# Linux
sudo systemctl start postgresql
```

### API Rate Limit Error

```
Error: 429 Too Many Requests
```

**Solution**: The script has built-in delays, but if you still hit limits:
- Increase `API_RATE_LIMIT_DELAY` in `.env` (e.g., `1.0`)
- Reduce `--movies` parameter
- Wait a few minutes and retry

### Import Errors

```
ModuleNotFoundError: No module named 'sklearn'
```

**Solution**: Ensure virtual environment is activated and dependencies installed:
```bash
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### NLTK Data Missing

```
LookupError: Resource punkt not found
```

**Solution**: The script downloads NLTK data automatically, but if needed:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## Next Steps

1. **Explore EDA**: Run the Jupyter notebook to understand your data
2. **Tune Models**: Experiment with different parameters in the model classes
3. **Add More Data**: Fetch more movies and reviews for better accuracy
4. **Clustering Analysis**: Use `src/models/clustering.py` to find movie clusters
5. **Custom Analysis**: Create your own notebooks in `notebooks/`

## Key Features Implemented

✓ API data collection (TMDb)
✓ PostgreSQL database with views
✓ Text preprocessing (cleaning, tokenization, lemmatization)
✓ Sentiment classification (TF-IDF + Logistic Regression/Random Forest)
✓ Score prediction (Ridge/Lasso/Linear Regression)
✓ EDA with statistical tests (Chi², ANOVA, Correlation with p-values)
✓ K-means clustering with elbow analysis
✓ Interactive dashboard (Flask/Dash)
✓ Geo-visualization support

## Resources

- **TMDb API Docs**: https://developers.themoviedb.org/3
- **Scikit-learn Docs**: https://scikit-learn.org/
- **Dash Documentation**: https://dash.plotly.com/

## Need Help?

Check the main README.md for more detailed information, or review the code comments in each module.

Happy analyzing! 🎬📊
