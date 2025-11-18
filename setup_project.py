"""
MovieMind Project Setup Script
Automates initial project setup, dependency checks, and database initialization
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_python_version():
    """Check if Python version is 3.9+"""
    logger.info("Checking Python version...")
    version = sys.version_info

    if version.major < 3 or (version.major == 3 and version.minor < 9):
        logger.error(f"Python 3.9+ required, but found {version.major}.{version.minor}")
        return False

    logger.info(f"✓ Python {version.major}.{version.minor}.{version.micro} found")
    return True


def create_directories():
    """Create necessary project directories"""
    logger.info("Creating project directories...")

    directories = [
        'data',
        'data/raw',
        'data/processed',
        'models',
        'evaluation_results',
        'logs',
        'presentation',
        'presentation/screenshots'
    ]

    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Created/verified: {directory}/")

    return True


def check_env_file():
    """Check if .env file exists, create from sample if not"""
    logger.info("Checking environment configuration...")

    env_file = Path('.env')
    env_sample = Path('.env.sample')

    if not env_file.exists():
        if env_sample.exists():
            logger.warning(".env not found, creating from .env.sample")
            env_file.write_text(env_sample.read_text())
            logger.info("✓ Created .env from template")
            logger.warning("⚠ IMPORTANT: Edit .env and add your API keys!")
            return False
        else:
            logger.error(".env.sample not found!")
            return False
    else:
        logger.info("✓ .env file exists")

        # Check if API keys are set
        env_content = env_file.read_text()
        if 'your_tmdb_api_key_here' in env_content:
            logger.warning("⚠ WARNING: API keys not configured in .env!")
            return False

        return True


def check_dependencies():
    """Check if required dependencies are installed"""
    logger.info("Checking dependencies...")

    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'psycopg2', 'nltk',
        'flask', 'dash', 'plotly', 'sqlalchemy', 'requests'
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package} installed")
        except ImportError:
            logger.warning(f"✗ {package} NOT installed")
            missing.append(package)

    if missing:
        logger.warning(f"Missing packages: {', '.join(missing)}")
        logger.info("Run: pip install -r requirements.txt")
        return False

    logger.info("✓ All dependencies installed")
    return True


def download_nltk_data():
    """Download required NLTK data"""
    logger.info("Downloading NLTK data...")

    try:
        import nltk

        resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']

        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
                logger.info(f"✓ NLTK {resource} already downloaded")
            except LookupError:
                logger.info(f"Downloading NLTK {resource}...")
                nltk.download(resource, quiet=True)
                logger.info(f"✓ Downloaded {resource}")

        return True
    except Exception as e:
        logger.error(f"Failed to download NLTK data: {e}")
        return False


def check_postgresql():
    """Check if PostgreSQL is accessible"""
    logger.info("Checking PostgreSQL connection...")

    try:
        import psycopg2
        from dotenv import load_dotenv

        load_dotenv()

        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', '5432'),
            database=os.getenv('DB_NAME', 'moviemind'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', '')
        )

        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        logger.info(f"✓ PostgreSQL connected: {version.split(',')[0]}")

        cursor.close()
        conn.close()
        return True

    except Exception as e:
        logger.error(f"PostgreSQL connection failed: {e}")
        logger.warning("Make sure PostgreSQL is running and .env is configured correctly")
        return False


def initialize_database():
    """Initialize database schema"""
    logger.info("Initializing database schema...")

    schema_file = Path('sql/schema.sql')

    if not schema_file.exists():
        logger.error("sql/schema.sql not found!")
        return False

    try:
        import psycopg2
        from dotenv import load_dotenv

        load_dotenv()

        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', '5432'),
            database=os.getenv('DB_NAME', 'moviemind'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', '')
        )

        cursor = conn.cursor()

        # Execute schema
        schema_sql = schema_file.read_text()
        cursor.execute(schema_sql)
        conn.commit()

        logger.info("✓ Database schema initialized")

        # Verify tables
        cursor.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
        """)
        tables = cursor.fetchall()
        logger.info(f"✓ Tables created: {', '.join([t[0] for t in tables])}")

        cursor.close()
        conn.close()
        return True

    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False


def create_project_summary():
    """Create PROJECT_STATUS.md summary file"""
    logger.info("Creating project status summary...")

    status_content = f"""# MovieMind - Project Status

**Last Updated**: {Path.ctime(Path('.'))}

## Project Structure

```
MovieMind/
├── src/
│   ├── data_collection/    # ✓ API clients (TMDb)
│   ├── preprocessing/      # ✓ Text processing (NLP)
│   ├── models/            # ✓ ML models (sentiment, score, clustering)
│   └── utils/             # ✓ Database manager
├── notebooks/             # ✓ EDA, training, clustering, geo-viz
├── dashboards/            # ✓ Flask/Dash demo app
├── sql/                   # ✓ PostgreSQL schema & views
├── data/                  # Data storage (gitignored)
├── models/               # Trained models (created after training)
├── evaluation_results/   # Model evaluation outputs
└── presentation/         # Presentation materials
```

## Setup Checklist

### Environment
- [ ] Python 3.9+ installed
- [ ] Virtual environment created (`python -m venv venv`)
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] NLTK data downloaded

### Configuration
- [ ] `.env` file created from `.env.sample`
- [ ] TMDb API key configured
- [ ] PostgreSQL credentials configured

### Database
- [ ] PostgreSQL installed and running
- [ ] Database `moviemind` created
- [ ] Schema initialized (`sql/schema.sql`)
- [ ] Tables verified: `movies`, `reviews`, `countries`
- [ ] Views created: `movie_review_stats`, `genre_sentiment_analysis`, etc.

### Data Collection
- [ ] TMDb API key tested
- [ ] Initial movies fetched (target: 500-1000)
- [ ] Reviews collected (target: 30+ per movie)

### Analysis & Models
- [ ] EDA notebook executed (`01_exploratory_analysis.ipynb`)
- [ ] Sentiment classifier trained
- [ ] Score predictor trained
- [ ] Clustering analysis completed
- [ ] Geographic visualization created

### Deliverables
- [ ] Presentation slides created
- [ ] Bonus points appendix prepared
- [ ] Video recording (15 min)
- [ ] Materials ZIP prepared
- [ ] Uploaded to Moodle

## Quick Start Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Initialize database
python setup_project.py --init-db
```

### Data Collection
```bash
# Collect movies and reviews
python src/data_collection/fetch_movies.py --movies 500 --strategy mixed
```

### Training
```bash
# Train models
python src/models/train_models.py
```

### Evaluation
```bash
# Evaluate models
python src/models/evaluate_models.py
```

### Dashboard
```bash
# Launch dashboard
python dashboards/app.py
```

## Next Steps

1. **Configure API Keys**: Edit `.env` with your TMDb API key
2. **Collect Data**: Run `fetch_movies.py` to gather initial dataset
3. **Run EDA**: Execute `notebooks/01_exploratory_analysis.ipynb`
4. **Train Models**: Run `src/models/train_models.py`
5. **Evaluate**: Run `src/models/evaluate_models.py`
6. **Create Presentation**: Use `PRESENTATION_OUTLINE.md` as guide

## Key Files

- `README.md` - Project overview
- `QUICKSTART.md` - Step-by-step setup guide
- `PRESENTATION_OUTLINE.md` - Detailed presentation structure
- `requirements.txt` - Python dependencies
- `.env.sample` - Environment template

## Support

For issues or questions:
1. Check `QUICKSTART.md` troubleshooting section
2. Review code comments and docstrings
3. Check logs in `logs/` directory
"""

    Path('PROJECT_STATUS.md').write_text(status_content)
    logger.info("✓ Created PROJECT_STATUS.md")


def main():
    """Main setup function"""
    logger.info("=" * 80)
    logger.info("MovieMind Project Setup")
    logger.info("=" * 80)

    steps = [
        ("Python version", check_python_version),
        ("Project directories", create_directories),
        ("Environment file", check_env_file),
        ("Dependencies", check_dependencies),
        ("NLTK data", download_nltk_data),
    ]

    results = {}

    for name, func in steps:
        logger.info(f"\n--- {name} ---")
        try:
            results[name] = func()
        except Exception as e:
            logger.error(f"Failed: {e}", exc_info=True)
            results[name] = False

    # Optional steps (require database connection)
    logger.info("\n--- Optional: PostgreSQL ---")
    try:
        if check_postgresql():
            results["PostgreSQL"] = True

            user_input = input("\nInitialize database schema? (y/n): ").lower()
            if user_input == 'y':
                results["Database Init"] = initialize_database()
        else:
            results["PostgreSQL"] = False
    except Exception as e:
        logger.error(f"PostgreSQL setup failed: {e}")
        results["PostgreSQL"] = False

    # Create summary
    create_project_summary()

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SETUP SUMMARY")
    logger.info("=" * 80)

    for step, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{status:10} | {step}")

    all_passed = all(results.values())

    if all_passed:
        logger.info("\n✓ Setup complete! Ready to start data collection.")
        logger.info("\nNext steps:")
        logger.info("  1. Edit .env and add your TMDb API key")
        logger.info("  2. Run: python src/data_collection/fetch_movies.py")
        logger.info("  3. Check QUICKSTART.md for detailed instructions")
    else:
        logger.warning("\n⚠ Some setup steps failed. Review errors above.")

    logger.info("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Setup MovieMind project')
    parser.add_argument('--init-db', action='store_true', help='Initialize database without prompt')
    args = parser.parse_args()

    if args.init_db:
        # Direct database initialization
        if check_postgresql():
            initialize_database()
    else:
        main()
