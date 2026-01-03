"""
Score Prediction Model for MovieMind
Predicts movie ratings (0-10 scale) from review text and metadata
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)
from typing import Tuple, Dict, Any
import joblib
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScorePredictor:
    """Score prediction model using TF-IDF + metadata features"""

    def __init__(self, model_type: str = 'ridge', max_features: int = 3000):
        """
        Initialize score predictor

        Args:
            model_type: Type of model ('linear', 'ridge', 'lasso', 'random_forest')
            max_features: Maximum number of TF-IDF features
        """
        self.model_type = model_type
        self.max_features = max_features

        # TF-IDF Vectorizer for text
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
            strip_accents='unicode',
            lowercase=True
        )

        # Scaler for metadata features
        self.scaler = StandardScaler()

        # Initialize model
        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'ridge':
            self.model = Ridge(alpha=1.0, random_state=42)
        elif model_type == 'lasso':
            self.model = Lasso(alpha=0.1, random_state=42, max_iter=2000)
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.is_trained = False
        self.use_metadata = False

    def combine_features(self, X_text_tfidf, X_meta=None):
        """
        Combine text and metadata features

        Args:
            X_text_tfidf: TF-IDF transformed text features
            X_meta: Metadata features (optional)

        Returns:
            Combined feature matrix
        """
        if X_meta is not None and len(X_meta) > 0:
            # Convert to array if DataFrame
            if isinstance(X_meta, pd.DataFrame):
                X_meta = X_meta.values

            # Ensure 2D
            if X_meta.ndim == 1:
                X_meta = X_meta.reshape(-1, 1)

            # Combine
            from scipy.sparse import hstack, csr_matrix
            return hstack([X_text_tfidf, csr_matrix(X_meta)])
        else:
            return X_text_tfidf

    def train(self, X_text: pd.Series, y_train: pd.Series,
             X_meta: pd.DataFrame = None,
             validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the score predictor

        Args:
            X_text: Training texts
            y_train: Training scores
            X_meta: Metadata features (optional)
            validation_split: Proportion of data for validation

        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training {self.model_type} score predictor...")

        # Check if using metadata
        self.use_metadata = X_meta is not None

        # Split data
        if validation_split > 0:
            if self.use_metadata:
                X_text_tr, X_text_val, X_meta_tr, X_meta_val, y_tr, y_val = train_test_split(
                    X_text, X_meta, y_train,
                    test_size=validation_split,
                    random_state=42
                )
            else:
                X_text_tr, X_text_val, y_tr, y_val = train_test_split(
                    X_text, y_train,
                    test_size=validation_split,
                    random_state=42
                )
                X_meta_tr, X_meta_val = None, None
        else:
            X_text_tr, y_tr = X_text, y_train
            X_meta_tr = X_meta
            X_text_val, y_val, X_meta_val = None, None, None

        # Fit vectorizer and transform text
        logger.info("Fitting TF-IDF vectorizer...")
        X_text_tfidf_tr = self.vectorizer.fit_transform(X_text_tr)

        # Scale metadata features if present
        if self.use_metadata:
            logger.info("Scaling metadata features...")
            X_meta_scaled_tr = self.scaler.fit_transform(X_meta_tr)
        else:
            X_meta_scaled_tr = None

        # Combine features
        X_combined_tr = self.combine_features(X_text_tfidf_tr, X_meta_scaled_tr)

        # Train model
        logger.info("Training model...")
        self.model.fit(X_combined_tr, y_tr)
        self.is_trained = True

        # Training metrics
        train_pred = self.model.predict(X_combined_tr)
        train_rmse = np.sqrt(mean_squared_error(y_tr, train_pred))
        train_mae = mean_absolute_error(y_tr, train_pred)
        train_r2 = r2_score(y_tr, train_pred)

        metrics = {
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'train_r2': train_r2,
            'train_samples': len(X_text_tr),
            'n_features': X_combined_tr.shape[1]
        }

        # Validation metrics
        if X_text_val is not None:
            X_text_tfidf_val = self.vectorizer.transform(X_text_val)

            if self.use_metadata:
                X_meta_scaled_val = self.scaler.transform(X_meta_val)
            else:
                X_meta_scaled_val = None

            X_combined_val = self.combine_features(X_text_tfidf_val, X_meta_scaled_val)

            val_pred = self.model.predict(X_combined_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            val_mae = mean_absolute_error(y_val, val_pred)
            val_r2 = r2_score(y_val, val_pred)

            metrics['val_rmse'] = val_rmse
            metrics['val_mae'] = val_mae
            metrics['val_r2'] = val_r2
            metrics['val_samples'] = len(X_text_val)

            logger.info(f"Training - RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
            logger.info(f"Validation - RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
        else:
            logger.info(f"Training - RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")

        return metrics

    def predict(self, X_text: pd.Series, X_meta: pd.DataFrame = None) -> np.ndarray:
        """
        Predict scores

        Args:
            X_text: Input texts
            X_meta: Metadata features (optional)

        Returns:
            Array of predicted scores
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")

        # Transform text
        X_text_tfidf = self.vectorizer.transform(X_text)

        # Scale metadata if used during training
        if self.use_metadata:
            if X_meta is None:
                raise ValueError("Model was trained with metadata but none provided for prediction")
            X_meta_scaled = self.scaler.transform(X_meta)
        else:
            X_meta_scaled = None

        # Combine features
        X_combined = self.combine_features(X_text_tfidf, X_meta_scaled)

        # Predict
        predictions = self.model.predict(X_combined)

        # Clip predictions to valid range (0-10)
        predictions = np.clip(predictions, 0, 10)

        return predictions

    def evaluate(self, X_text: pd.Series, y_test: pd.Series,
                X_meta: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Evaluate model on test data

        Args:
            X_text: Test texts
            y_test: True scores
            X_meta: Metadata features (optional)

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating score predictor...")

        # Predict
        y_pred = self.predict(X_text, X_meta)

        # Convert y_test to float to avoid Decimal issues
        y_test = y_test.astype(float)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Residuals
        residuals = y_test - y_pred

        metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'residuals': residuals,
            'predictions': y_pred,
            'test_samples': len(X_text)
        }

        logger.info(f"Test RMSE: {rmse:.4f}")
        logger.info(f"Test MAE: {mae:.4f}")
        logger.info(f"Test R²: {r2:.4f}")

        return metrics

    def get_feature_importance(self, top_n: int = 20) -> list:
        """
        Get most important features

        Args:
            top_n: Number of top features to return

        Returns:
            List of (feature_name, importance) tuples
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet.")

        feature_names = list(self.vectorizer.get_feature_names_out())

        # Add metadata feature names if used
        if self.use_metadata:
            feature_names.extend([f"meta_{i}" for i in range(self.scaler.n_features_in_)])

        if self.model_type in ['linear', 'ridge', 'lasso']:
            # For linear models, use coefficients
            coefs = self.model.coef_

            # Handle 1D vs 2D coefficients
            if coefs.ndim > 1:
                coefs = coefs[0]

            top_indices = np.argsort(np.abs(coefs))[-top_n:][::-1]
            top_features = [
                (feature_names[i], coefs[i])
                for i in top_indices
            ]

        elif self.model_type == 'random_forest':
            # For random forest, use feature importances
            importances = self.model.feature_importances_
            top_indices = np.argsort(importances)[-top_n:][::-1]
            top_features = [
                (feature_names[i], importances[i])
                for i in top_indices
            ]

        return top_features

    def save_model(self, filepath: str):
        """
        Save trained model, vectorizer, and scaler

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

        # Save scaler if used
        if self.use_metadata:
            scaler_path = filepath.with_suffix('.scaler.pkl')
            joblib.dump(self.scaler, scaler_path)

        logger.info(f"Model saved to {model_path}")

    def load_model(self, filepath: str):
        """
        Load trained model, vectorizer, and scaler

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

        # Load scaler if exists
        scaler_path = filepath.with_suffix('.scaler.pkl')
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            self.use_metadata = True
        else:
            self.use_metadata = False

        self.is_trained = True
        logger.info(f"Model loaded from {model_path}")


if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)

    # Create sample data
    n_samples = 1000
    texts = [
        f"This is sample review {i} with some random words" * (i % 5 + 1)
        for i in range(n_samples)
    ]

    # Synthetic scores (0-10)
    scores = np.random.uniform(0, 10, n_samples)

    # Synthetic metadata
    metadata = pd.DataFrame({
        'text_length': [len(t) for t in texts],
        'word_count': [len(t.split()) for t in texts]
    })

    # Split data
    X_text_train, X_text_test, X_meta_train, X_meta_test, y_train, y_test = train_test_split(
        pd.Series(texts), metadata, scores,
        test_size=0.2, random_state=42
    )

    # Train predictor
    predictor = ScorePredictor(model_type='ridge')
    predictor.train(X_text_train, pd.Series(y_train), X_meta_train)

    # Evaluate
    metrics = predictor.evaluate(X_text_test, pd.Series(y_test), X_meta_test)

    # Get feature importance
    importance = predictor.get_feature_importance(top_n=10)
    print("\nTop features:")
    for feat, score in importance[:10]:
        print(f"  {feat}: {score:.4f}")
