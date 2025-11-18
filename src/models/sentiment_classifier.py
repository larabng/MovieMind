"""
Sentiment Classification Model for MovieMind
Classifies reviews as positive, neutral, or negative
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from typing import Tuple, Dict, Any
import joblib
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentClassifier:
    """Sentiment classification model using TF-IDF and ML"""

    def __init__(self, model_type: str = 'logistic', max_features: int = 5000):
        """
        Initialize sentiment classifier

        Args:
            model_type: Type of model ('logistic' or 'random_forest')
            max_features: Maximum number of TF-IDF features
        """
        self.model_type = model_type
        self.max_features = max_features

        # TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=2,
            max_df=0.8,
            strip_accents='unicode',
            lowercase=True
        )

        # Initialize model
        if model_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced',
                C=1.0
            )
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.is_trained = False

    def prepare_sentiment_labels(self, ratings: pd.Series, threshold_pos: float = 7.0,
                                 threshold_neg: float = 5.0) -> pd.Series:
        """
        Convert ratings to sentiment labels

        Args:
            ratings: Series of numerical ratings
            threshold_pos: Minimum rating for positive sentiment
            threshold_neg: Maximum rating for negative sentiment

        Returns:
            Series of sentiment labels
        """
        def to_sentiment(rating):
            if pd.isna(rating):
                return 'neutral'
            elif rating >= threshold_pos:
                return 'positive'
            elif rating <= threshold_neg:
                return 'negative'
            else:
                return 'neutral'

        return ratings.apply(to_sentiment)

    def train(self, X_train: pd.Series, y_train: pd.Series,
             validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the sentiment classifier

        Args:
            X_train: Training texts
            y_train: Training labels
            validation_split: Proportion of data for validation

        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training {self.model_type} sentiment classifier...")

        # Split training data for validation
        if validation_split > 0:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train,
                test_size=validation_split,
                random_state=42,
                stratify=y_train
            )
        else:
            X_tr, y_tr = X_train, y_train
            X_val, y_val = None, None

        # Fit vectorizer and transform
        logger.info("Fitting TF-IDF vectorizer...")
        X_tr_tfidf = self.vectorizer.fit_transform(X_tr)

        # Train model
        logger.info("Training model...")
        self.model.fit(X_tr_tfidf, y_tr)
        self.is_trained = True

        # Training metrics
        train_pred = self.model.predict(X_tr_tfidf)
        train_accuracy = accuracy_score(y_tr, train_pred)

        metrics = {
            'train_accuracy': train_accuracy,
            'train_samples': len(X_tr),
            'n_features': X_tr_tfidf.shape[1]
        }

        # Validation metrics
        if X_val is not None:
            X_val_tfidf = self.vectorizer.transform(X_val)
            val_pred = self.model.predict(X_val_tfidf)
            val_accuracy = accuracy_score(y_val, val_pred)

            metrics['val_accuracy'] = val_accuracy
            metrics['val_samples'] = len(X_val)

            logger.info(f"Training accuracy: {train_accuracy:.4f}")
            logger.info(f"Validation accuracy: {val_accuracy:.4f}")
        else:
            logger.info(f"Training accuracy: {train_accuracy:.4f}")

        return metrics

    def predict(self, texts: pd.Series) -> np.ndarray:
        """
        Predict sentiment labels

        Args:
            texts: Input texts

        Returns:
            Array of predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")

        X_tfidf = self.vectorizer.transform(texts)
        return self.model.predict(X_tfidf)

    def predict_proba(self, texts: pd.Series) -> np.ndarray:
        """
        Predict sentiment probabilities

        Args:
            texts: Input texts

        Returns:
            Array of probabilities for each class
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")

        X_tfidf = self.vectorizer.transform(texts)
        return self.model.predict_proba(X_tfidf)

    def evaluate(self, X_test: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate model on test data

        Args:
            X_test: Test texts
            y_test: True labels

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating sentiment classifier...")

        # Transform and predict
        X_test_tfidf = self.vectorizer.transform(X_test)
        y_pred = self.model.predict(X_test_tfidf)
        y_pred_proba = self.model.predict_proba(X_test_tfidf)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Classification report
        report = classification_report(y_test, y_pred, zero_division=0)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': report,
            'test_samples': len(X_test)
        }

        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Test Precision: {precision:.4f}")
        logger.info(f"Test Recall: {recall:.4f}")
        logger.info(f"Test F1-Score: {f1:.4f}")

        print("\nClassification Report:")
        print(report)

        print("\nConfusion Matrix:")
        print(cm)

        return metrics

    def get_feature_importance(self, top_n: int = 20) -> Dict[str, list]:
        """
        Get most important features per class

        Args:
            top_n: Number of top features to return

        Returns:
            Dictionary mapping class to top features
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet.")

        feature_names = self.vectorizer.get_feature_names_out()

        if self.model_type == 'logistic':
            # For logistic regression, use coefficients
            importance_dict = {}

            for idx, class_name in enumerate(self.model.classes_):
                coefs = self.model.coef_[idx]
                top_indices = np.argsort(np.abs(coefs))[-top_n:][::-1]
                top_features = [
                    (feature_names[i], coefs[i])
                    for i in top_indices
                ]
                importance_dict[class_name] = top_features

        elif self.model_type == 'random_forest':
            # For random forest, use feature importances
            importances = self.model.feature_importances_
            top_indices = np.argsort(importances)[-top_n:][::-1]
            top_features = [
                (feature_names[i], importances[i])
                for i in top_indices
            ]
            importance_dict = {'all_classes': top_features}

        return importance_dict

    def save_model(self, filepath: str):
        """
        Save trained model and vectorizer

        Args:
            filepath: Path to save model (without extension)
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet.")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = filepath.with_suffix('.model.pkl')
        joblib.dump(self.model, model_path)

        # Save vectorizer
        vectorizer_path = filepath.with_suffix('.vectorizer.pkl')
        joblib.dump(self.vectorizer, vectorizer_path)

        logger.info(f"Model saved to {model_path}")
        logger.info(f"Vectorizer saved to {vectorizer_path}")

    def load_model(self, filepath: str):
        """
        Load trained model and vectorizer

        Args:
            filepath: Path to model (without extension)
        """
        filepath = Path(filepath)

        # Load model
        model_path = filepath.with_suffix('.model.pkl')
        self.model = joblib.load(model_path)

        # Load vectorizer
        vectorizer_path = filepath.with_suffix('.vectorizer.pkl')
        self.vectorizer = joblib.load(vectorizer_path)

        self.is_trained = True
        logger.info(f"Model loaded from {model_path}")


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import fetch_20newsgroups

    # Load sample data (using newsgroups as example)
    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics']
    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
    newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

    X_train = pd.Series(newsgroups_train.data)
    y_train = pd.Series([newsgroups_train.target_names[i] for i in newsgroups_train.target])
    X_test = pd.Series(newsgroups_test.data)
    y_test = pd.Series([newsgroups_test.target_names[i] for i in newsgroups_test.target])

    # Train classifier
    classifier = SentimentClassifier(model_type='logistic')
    classifier.train(X_train, y_train)

    # Evaluate
    metrics = classifier.evaluate(X_test, y_test)

    # Get feature importance
    importance = classifier.get_feature_importance(top_n=10)
    print("\nTop features per class:")
    for class_name, features in importance.items():
        print(f"\n{class_name}:")
        for feat, score in features[:5]:
            print(f"  {feat}: {score:.4f}")
