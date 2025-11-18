"""
Training script for MovieMind ML models
Trains both sentiment classifier and score predictor
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import argparse

from src.utils.db_manager import DatabaseManager
from src.preprocessing.text_processor import TextProcessor
from src.models.sentiment_classifier import SentimentClassifier
from src.models.score_predictor import ScorePredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data_from_db(limit: int = None):
    """
    Load data from PostgreSQL database

    Args:
        limit: Maximum number of reviews to load (None for all)

    Returns:
        DataFrame with reviews and movie metadata
    """
    logger.info("Loading data from database...")

    with DatabaseManager() as db:
        query = """
        SELECT
            r.review_id,
            r.content,
            r.rating,
            r.text_length,
            r.word_count,
            m.title,
            m.genres,
            m.vote_average,
            m.runtime
        FROM reviews r
        JOIN movies m ON r.movie_id = m.movie_id
        WHERE r.content IS NOT NULL
        """

        if limit:
            query += f" LIMIT {limit}"

        reviews = db.execute_query(query)

    df = pd.DataFrame(reviews)
    logger.info(f"Loaded {len(df)} reviews")

    return df


def train_sentiment_classifier(df: pd.DataFrame, save_path: str = 'models/sentiment_classifier'):
    """
    Train sentiment classification model

    Args:
        df: DataFrame with review data
        save_path: Path to save model
    """
    logger.info("=" * 60)
    logger.info("Training Sentiment Classifier")
    logger.info("=" * 60)

    # Prepare text processor
    processor = TextProcessor()

    # Clean reviews
    logger.info("Preprocessing text...")
    df['cleaned_content'] = df['content'].apply(
        lambda x: processor.clean_text(x) if pd.notna(x) else ""
    )

    # Prepare labels from ratings
    # Use vote_average if rating not available
    df['score'] = df['rating'].fillna(df['vote_average'])

    # Filter out rows without scores
    df_labeled = df[df['score'].notna()].copy()

    logger.info(f"Reviews with labels: {len(df_labeled)}")

    # Create sentiment labels
    classifier = SentimentClassifier(model_type='logistic', max_features=5000)
    df_labeled['sentiment'] = classifier.prepare_sentiment_labels(
        df_labeled['score'],
        threshold_pos=7.0,
        threshold_neg=5.0
    )

    # Check label distribution
    logger.info("Label distribution:")
    logger.info(df_labeled['sentiment'].value_counts())

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df_labeled['cleaned_content'],
        df_labeled['sentiment'],
        test_size=0.2,
        random_state=42,
        stratify=df_labeled['sentiment']
    )

    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Train model
    train_metrics = classifier.train(X_train, y_train, validation_split=0.2)

    # Evaluate
    test_metrics = classifier.evaluate(X_test, y_test)

    # Get feature importance
    importance = classifier.get_feature_importance(top_n=20)
    logger.info("\nTop features per class:")
    for class_name, features in importance.items():
        logger.info(f"\n{class_name}:")
        for feat, score in features[:10]:
            logger.info(f"  {feat}: {score:.4f}")

    # Save model
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    classifier.save_model(save_path)

    return classifier, test_metrics


def train_score_predictor(df: pd.DataFrame, save_path: str = 'models/score_predictor'):
    """
    Train score prediction model

    Args:
        df: DataFrame with review data
        save_path: Path to save model
    """
    logger.info("=" * 60)
    logger.info("Training Score Predictor")
    logger.info("=" * 60)

    # Prepare text processor
    processor = TextProcessor()

    # Clean reviews
    logger.info("Preprocessing text...")
    df['cleaned_content'] = df['content'].apply(
        lambda x: processor.clean_text(x) if pd.notna(x) else ""
    )

    # Use rating or vote_average as target
    df['score'] = df['rating'].fillna(df['vote_average'])

    # Filter out rows without scores
    df_scored = df[df['score'].notna()].copy()

    logger.info(f"Reviews with scores: {len(df_scored)}")

    # Prepare metadata features
    meta_features = ['text_length', 'word_count']
    df_scored[meta_features] = df_scored[meta_features].fillna(0)

    # Split data
    X_text_train, X_text_test, X_meta_train, X_meta_test, y_train, y_test = train_test_split(
        df_scored['cleaned_content'],
        df_scored[meta_features],
        df_scored['score'],
        test_size=0.2,
        random_state=42
    )

    logger.info(f"Train size: {len(X_text_train)}, Test size: {len(X_text_test)}")

    # Train model
    predictor = ScorePredictor(model_type='ridge', max_features=3000)
    train_metrics = predictor.train(
        X_text_train,
        y_train,
        X_meta_train,
        validation_split=0.2
    )

    # Evaluate
    test_metrics = predictor.evaluate(X_text_test, y_test, X_meta_test)

    # Get feature importance
    importance = predictor.get_feature_importance(top_n=20)
    logger.info("\nTop features:")
    for feat, score in importance[:15]:
        logger.info(f"  {feat}: {score:.4f}")

    # Save model
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    predictor.save_model(save_path)

    return predictor, test_metrics


def main(limit: int = None, train_classifier: bool = True, train_regressor: bool = True):
    """
    Main training function

    Args:
        limit: Limit number of reviews (None for all)
        train_classifier: Whether to train classifier
        train_regressor: Whether to train regressor
    """
    # Load data
    df = load_data_from_db(limit=limit)

    if len(df) == 0:
        logger.error("No data loaded! Please run data collection first.")
        return

    # Train sentiment classifier
    if train_classifier:
        try:
            classifier, metrics = train_sentiment_classifier(df)
            logger.info("\n✓ Sentiment classifier trained successfully!")
            logger.info(f"  Test Accuracy: {metrics['accuracy']:.4f}")
        except Exception as e:
            logger.error(f"Error training classifier: {e}", exc_info=True)

    # Train score predictor
    if train_regressor:
        try:
            predictor, metrics = train_score_predictor(df)
            logger.info("\n✓ Score predictor trained successfully!")
            logger.info(f"  Test R²: {metrics['r2']:.4f}")
            logger.info(f"  Test RMSE: {metrics['rmse']:.4f}")
        except Exception as e:
            logger.error(f"Error training predictor: {e}", exc_info=True)

    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MovieMind ML models')
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of reviews to load (default: all)'
    )
    parser.add_argument(
        '--skip-classifier',
        action='store_true',
        help='Skip training sentiment classifier'
    )
    parser.add_argument(
        '--skip-regressor',
        action='store_true',
        help='Skip training score predictor'
    )

    args = parser.parse_args()

    main(
        limit=args.limit,
        train_classifier=not args.skip_classifier,
        train_regressor=not args.skip_regressor
    )
