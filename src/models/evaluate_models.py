"""
Model Evaluation Script for MovieMind
Comprehensive evaluation with all metrics, plots, and statistical analysis
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)
from scipy import stats
import logging
import argparse

from src.utils.db_manager import DatabaseManager
from src.preprocessing.text_processor import TextProcessor
from src.models.sentiment_classifier import SentimentClassifier
from src.models.score_predictor import ScorePredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_test_data(limit: int = None):
    """Load test data from database"""
    logger.info("Loading test data from database...")

    with DatabaseManager() as db:
        query = """
        SELECT
            r.review_id,
            r.content,
            r.rating,
            r.sentiment,
            r.sentiment_score,
            r.predicted_rating,
            m.vote_average,
            m.genres
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


def evaluate_sentiment_classifier(model_path: str = 'models/sentiment_classifier'):
    """
    Evaluate sentiment classification model

    Args:
        model_path: Path to saved model
    """
    logger.info("=" * 80)
    logger.info("SENTIMENT CLASSIFIER EVALUATION")
    logger.info("=" * 80)

    # Load model
    try:
        classifier = SentimentClassifier()
        classifier.load_model(model_path)
        logger.info(f"✓ Model loaded from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # Load data
    df = load_test_data(limit=1000)

    # Prepare data
    processor = TextProcessor()
    df['cleaned_content'] = df['content'].apply(
        lambda x: processor.clean_text(x) if pd.notna(x) else ""
    )

    # Create labels from ratings
    df['score'] = df['rating'].fillna(df['vote_average'])
    df_labeled = df[df['score'].notna()].copy()

    df_labeled['true_sentiment'] = classifier.prepare_sentiment_labels(
        df_labeled['score'],
        threshold_pos=7.0,
        threshold_neg=5.0
    )

    # Predict
    logger.info("Making predictions...")
    predictions = classifier.predict(df_labeled['cleaned_content'])

    # Calculate metrics
    accuracy = accuracy_score(df_labeled['true_sentiment'], predictions)
    precision = precision_score(df_labeled['true_sentiment'], predictions, average='weighted', zero_division=0)
    recall = recall_score(df_labeled['true_sentiment'], predictions, average='weighted', zero_division=0)
    f1 = f1_score(df_labeled['true_sentiment'], predictions, average='weighted', zero_division=0)

    logger.info("\n--- CLASSIFICATION METRICS ---")
    logger.info(f"Accuracy:  {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1-Score:  {f1:.4f}")

    # Classification report
    logger.info("\n--- DETAILED CLASSIFICATION REPORT ---")
    logger.info("\n" + classification_report(df_labeled['true_sentiment'], predictions))

    # Confusion matrix
    cm = confusion_matrix(df_labeled['true_sentiment'], predictions)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classifier.model.classes_,
                yticklabels=classifier.model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Sentiment Classification')
    plt.tight_layout()
    plt.savefig('evaluation_results/confusion_matrix_sentiment.png', dpi=300, bbox_inches='tight')
    logger.info("✓ Confusion matrix saved to evaluation_results/confusion_matrix_sentiment.png")
    plt.show()

    # Per-class accuracy
    logger.info("\n--- PER-CLASS ACCURACY ---")
    for i, class_name in enumerate(classifier.model.classes_):
        class_accuracy = cm[i, i] / cm[i, :].sum()
        logger.info(f"  {class_name}: {class_accuracy:.4f}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }


def evaluate_score_predictor(model_path: str = 'models/score_predictor'):
    """
    Evaluate score prediction model

    Args:
        model_path: Path to saved model
    """
    logger.info("\n" + "=" * 80)
    logger.info("SCORE PREDICTOR EVALUATION")
    logger.info("=" * 80)

    # Load model
    try:
        predictor = ScorePredictor()
        predictor.load_model(model_path)
        logger.info(f"✓ Model loaded from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # Load data
    df = load_test_data(limit=1000)

    # Prepare data
    processor = TextProcessor()
    df['cleaned_content'] = df['content'].apply(
        lambda x: processor.clean_text(x) if pd.notna(x) else ""
    )

    df['score'] = df['rating'].fillna(df['vote_average'])
    df_scored = df[df['score'].notna()].copy()

    # Metadata features
    meta_features = ['text_length', 'word_count']
    for col in meta_features:
        if col not in df_scored.columns:
            if col == 'text_length':
                df_scored[col] = df_scored['content'].str.len()
            elif col == 'word_count':
                df_scored[col] = df_scored['content'].str.split().str.len()

    df_scored[meta_features] = df_scored[meta_features].fillna(0)

    # Predict
    logger.info("Making predictions...")
    predictions = predictor.predict(
        df_scored['cleaned_content'],
        df_scored[meta_features]
    )

    # Calculate metrics
    mse = mean_squared_error(df_scored['score'], predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(df_scored['score'], predictions)
    r2 = r2_score(df_scored['score'], predictions)

    logger.info("\n--- REGRESSION METRICS ---")
    logger.info(f"R² Score: {r2:.4f}")
    logger.info(f"RMSE:     {rmse:.4f}")
    logger.info(f"MAE:      {mae:.4f}")
    logger.info(f"MSE:      {mse:.4f}")

    # Residuals
    residuals = df_scored['score'] - predictions

    # Residual statistics
    logger.info("\n--- RESIDUAL STATISTICS ---")
    logger.info(f"Mean residual:   {residuals.mean():.4f}")
    logger.info(f"Median residual: {residuals.median():.4f}")
    logger.info(f"Std residual:    {residuals.std():.4f}")

    # Create residual plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Predicted vs Actual
    axes[0, 0].scatter(df_scored['score'], predictions, alpha=0.5)
    axes[0, 0].plot([df_scored['score'].min(), df_scored['score'].max()],
                     [df_scored['score'].min(), df_scored['score'].max()],
                     'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Score')
    axes[0, 0].set_ylabel('Predicted Score')
    axes[0, 0].set_title('Predicted vs Actual Scores')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Residual plot
    axes[0, 1].scatter(predictions, residuals, alpha=0.5)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted Score')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residual Plot')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Residual distribution
    axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(residuals.mean(), color='red', linestyle='--', label=f'Mean: {residuals.mean():.2f}')
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Residuals')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot (Normality Check)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('evaluation_results/regression_evaluation.png', dpi=300, bbox_inches='tight')
    logger.info("✓ Regression plots saved to evaluation_results/regression_evaluation.png")
    plt.show()

    # Statistical tests on residuals
    logger.info("\n--- RESIDUAL NORMALITY TEST (Shapiro-Wilk) ---")
    if len(residuals) <= 5000:  # Shapiro-Wilk works best with smaller samples
        stat, p_value = stats.shapiro(residuals)
        logger.info(f"Test statistic: {stat:.4f}")
        logger.info(f"p-value: {p_value:.4f}")
        logger.info(f"Residuals are {'NOT ' if p_value < 0.05 else ''}normally distributed (α=0.05)")
    else:
        logger.info("Sample too large for Shapiro-Wilk, using Kolmogorov-Smirnov instead")
        stat, p_value = stats.kstest(residuals, 'norm')
        logger.info(f"Test statistic: {stat:.4f}")
        logger.info(f"p-value: {p_value:.4f}")

    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'mse': mse,
        'residuals': residuals
    }


def main(classifier: bool = True, regressor: bool = True):
    """
    Main evaluation function

    Args:
        classifier: Whether to evaluate classifier
        regressor: Whether to evaluate regressor
    """
    # Create output directory
    Path('evaluation_results').mkdir(exist_ok=True)

    results = {}

    # Evaluate classifier
    if classifier:
        try:
            results['classifier'] = evaluate_sentiment_classifier()
        except Exception as e:
            logger.error(f"Classifier evaluation failed: {e}", exc_info=True)

    # Evaluate regressor
    if regressor:
        try:
            results['regressor'] = evaluate_score_predictor()
        except Exception as e:
            logger.error(f"Regressor evaluation failed: {e}", exc_info=True)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 80)

    if 'classifier' in results:
        logger.info("\nSentiment Classifier:")
        logger.info(f"  Accuracy:  {results['classifier']['accuracy']:.4f}")
        logger.info(f"  Precision: {results['classifier']['precision']:.4f}")
        logger.info(f"  Recall:    {results['classifier']['recall']:.4f}")
        logger.info(f"  F1-Score:  {results['classifier']['f1']:.4f}")

    if 'regressor' in results:
        logger.info("\nScore Predictor:")
        logger.info(f"  R² Score: {results['regressor']['r2']:.4f}")
        logger.info(f"  RMSE:     {results['regressor']['rmse']:.4f}")
        logger.info(f"  MAE:      {results['regressor']['mae']:.4f}")

    logger.info("\n" + "=" * 80)
    logger.info("✓ Evaluation complete! Check evaluation_results/ for plots.")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate MovieMind ML models')
    parser.add_argument(
        '--skip-classifier',
        action='store_true',
        help='Skip evaluating sentiment classifier'
    )
    parser.add_argument(
        '--skip-regressor',
        action='store_true',
        help='Skip evaluating score predictor'
    )

    args = parser.parse_args()

    main(
        classifier=not args.skip_classifier,
        regressor=not args.skip_regressor
    )
