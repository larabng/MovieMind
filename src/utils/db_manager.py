"""
Database Manager for MovieMind
Handles PostgreSQL connections and operations
"""

import os
import psycopg2
from psycopg2.extras import execute_values, RealDictCursor
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import logging
from decimal import Decimal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class DatabaseManager:
    """Manages database connections and operations for MovieMind"""

    def __init__(self):
        """Initialize database connection"""
        self.connection_params = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'moviemind'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', '')
        }
        self.conn = None
        self.cursor = None

    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(**self.connection_params)
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def disconnect(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info("Database connection closed")

    def _convert_decimals(self, data: Any) -> Any:
        """
        Recursively convert Decimal objects to float for numpy/matplotlib compatibility

        Args:
            data: Data to convert (can be dict, list, Decimal, or any other type)

        Returns:
            Converted data with Decimals as floats
        """
        if isinstance(data, Decimal):
            return float(data)
        elif isinstance(data, dict):
            return {key: self._convert_decimals(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._convert_decimals(item) for item in data]
        else:
            return data

    def execute_query(self, query: str, params: tuple = None) -> List[Dict]:
        """
        Execute a SELECT query and return results

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            List of dictionaries containing query results (Decimals converted to float)
        """
        try:
            self.cursor.execute(query, params)
            results = self.cursor.fetchall()
            # Convert all Decimal values to float for numpy/matplotlib compatibility
            return [self._convert_decimals(dict(row)) for row in results]
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise

    def execute_insert(self, query: str, params: tuple = None) -> int:
        """
        Execute an INSERT query

        Args:
            query: SQL INSERT query
            params: Query parameters

        Returns:
            Number of rows affected
        """
        try:
            self.cursor.execute(query, params)
            self.conn.commit()
            return self.cursor.rowcount
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Insert failed: {e}")
            raise

    def insert_movie(self, movie_data: Dict[str, Any]) -> bool:
        """
        Insert a movie into the database

        Args:
            movie_data: Dictionary containing movie information

        Returns:
            True if successful, False otherwise
        """
        query = """
        INSERT INTO movies (
            movie_id, title, original_title, release_date, runtime,
            budget, revenue, genres, production_countries, original_language,
            spoken_languages, vote_average, vote_count, popularity,
            overview, tagline, status, adult, video, imdb_id,
            homepage, poster_path, backdrop_path
        ) VALUES (
            %(movie_id)s, %(title)s, %(original_title)s, %(release_date)s, %(runtime)s,
            %(budget)s, %(revenue)s, %(genres)s, %(production_countries)s, %(original_language)s,
            %(spoken_languages)s, %(vote_average)s, %(vote_count)s, %(popularity)s,
            %(overview)s, %(tagline)s, %(status)s, %(adult)s, %(video)s, %(imdb_id)s,
            %(homepage)s, %(poster_path)s, %(backdrop_path)s
        )
        ON CONFLICT (movie_id) DO UPDATE SET
            vote_average = EXCLUDED.vote_average,
            vote_count = EXCLUDED.vote_count,
            popularity = EXCLUDED.popularity,
            updated_at = CURRENT_TIMESTAMP
        """

        try:
            self.cursor.execute(query, movie_data)
            self.conn.commit()
            logger.info(f"Inserted/Updated movie: {movie_data.get('title')}")
            return True
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to insert movie {movie_data.get('title')}: {e}")
            return False

    def insert_review(self, review_data: Dict[str, Any]) -> bool:
        """
        Insert a review into the database

        Args:
            review_data: Dictionary containing review information

        Returns:
            True if successful, False otherwise
        """
        query = """
        INSERT INTO reviews (
            movie_id, author, content, rating, review_date,
            language, url, text_length, word_count
        ) VALUES (
            %(movie_id)s, %(author)s, %(content)s, %(rating)s, %(review_date)s,
            %(language)s, %(url)s, %(text_length)s, %(word_count)s
        )
        ON CONFLICT (movie_id, author, content) DO NOTHING
        """

        try:
            self.cursor.execute(query, review_data)
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to insert review: {e}")
            return False

    def bulk_insert_reviews(self, reviews: List[Dict[str, Any]]) -> int:
        """
        Bulk insert reviews

        Args:
            reviews: List of review dictionaries

        Returns:
            Number of reviews inserted
        """
        query = """
        INSERT INTO reviews (
            movie_id, author, content, rating, review_date,
            language, url, text_length, word_count
        ) VALUES %s
        ON CONFLICT (movie_id, author, content) DO NOTHING
        """

        values = [
            (
                r.get('movie_id'),
                r.get('author'),
                r.get('content'),
                r.get('rating'),
                r.get('review_date'),
                r.get('language'),
                r.get('url'),
                r.get('text_length'),
                r.get('word_count')
            )
            for r in reviews
        ]

        try:
            execute_values(self.cursor, query, values)
            self.conn.commit()
            inserted = self.cursor.rowcount
            logger.info(f"Bulk inserted {inserted} reviews")
            return inserted
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Bulk insert failed: {e}")
            return 0

    def get_movies_without_reviews(self, limit: int = 100) -> List[Dict]:
        """
        Get movies that don't have reviews yet

        Args:
            limit: Maximum number of movies to return

        Returns:
            List of movie dictionaries
        """
        query = """
        SELECT m.movie_id, m.title
        FROM movies m
        LEFT JOIN reviews r ON m.movie_id = r.movie_id
        WHERE r.review_id IS NULL
        LIMIT %s
        """

        return self.execute_query(query, (limit,))

    def update_review_sentiment(self, review_id: int, sentiment: str,
                                sentiment_score: float, predicted_rating: float = None):
        """
        Update sentiment analysis results for a review

        Args:
            review_id: Review ID
            sentiment: Sentiment label (positive/neutral/negative)
            sentiment_score: Sentiment score (-1 to 1)
            predicted_rating: ML predicted rating
        """
        query = """
        UPDATE reviews
        SET sentiment = %s,
            sentiment_score = %s,
            predicted_rating = %s,
            updated_at = CURRENT_TIMESTAMP
        WHERE review_id = %s
        """

        try:
            self.cursor.execute(query, (sentiment, sentiment_score, predicted_rating, review_id))
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to update review sentiment: {e}")

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()
