"""
Main script to fetch movies from TMDb API and store in PostgreSQL
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_collection.tmdb_client import TMDbClient
from src.utils.db_manager import DatabaseManager
from dotenv import load_dotenv
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()


def fetch_and_store_movies(num_movies: int = 1000, fetch_strategy: str = 'mixed'):
    """
    Fetch movies from TMDb and store in database

    Args:
        num_movies: Target number of movies to fetch
        fetch_strategy: Strategy for fetching ('popular', 'top_rated', 'mixed')
    """
    client = TMDbClient()

    with DatabaseManager() as db:
        logger.info(f"Starting movie collection (target: {num_movies} movies)")

        movies_collected = 0
        page = 1
        max_pages = (num_movies // 20) + 1

        with tqdm(total=num_movies, desc="Collecting movies") as pbar:
            while movies_collected < num_movies and page <= max_pages:

                # Fetch movies based on strategy
                if fetch_strategy == 'popular':
                    movies = client.get_popular_movies(page=page)
                elif fetch_strategy == 'top_rated':
                    movies = client.get_top_rated_movies(page=page)
                elif fetch_strategy == 'mixed':
                    # Alternate between popular and top rated
                    if page % 2 == 1:
                        movies = client.get_popular_movies(page=(page // 2) + 1)
                    else:
                        movies = client.get_top_rated_movies(page=page // 2)
                else:
                    raise ValueError(f"Unknown strategy: {fetch_strategy}")

                if not movies:
                    logger.warning(f"No movies returned for page {page}")
                    break

                # Fetch detailed info and store each movie
                for movie in movies:
                    if movies_collected >= num_movies:
                        break

                    movie_id = movie.get('id')

                    # Get detailed information
                    details = client.get_movie_details(movie_id)

                    if details:
                        # Format for database
                        movie_data = client.format_movie_for_db(details)

                        # Insert into database
                        success = db.insert_movie(movie_data)

                        if success:
                            movies_collected += 1
                            pbar.update(1)

                page += 1

        logger.info(f"Movie collection complete! Total movies: {movies_collected}")


def fetch_reviews_for_movies(max_reviews_per_movie: int = 50):
    """
    Fetch reviews for movies that don't have any reviews yet

    Args:
        max_reviews_per_movie: Maximum number of reviews to fetch per movie
    """
    client = TMDbClient()

    with DatabaseManager() as db:
        # Get movies without reviews
        movies = db.get_movies_without_reviews(limit=100)

        logger.info(f"Found {len(movies)} movies without reviews")

        for movie in tqdm(movies, desc="Fetching reviews"):
            movie_id = movie['movie_id']

            # Fetch reviews from API
            reviews_data = client.get_all_movie_reviews(
                movie_id,
                max_pages=(max_reviews_per_movie // 20) + 1
            )

            if reviews_data:
                # Format reviews for database
                reviews = [
                    client.format_review_for_db(review, movie_id)
                    for review in reviews_data
                ]

                # Limit reviews per movie
                reviews = reviews[:max_reviews_per_movie]

                # Bulk insert
                inserted = db.bulk_insert_reviews(reviews)
                logger.info(f"Inserted {inserted} reviews for movie {movie['title']}")

        logger.info("Review collection complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Fetch movies from TMDb API')
    parser.add_argument(
        '--movies',
        type=int,
        default=int(os.getenv('MAX_MOVIES', 1000)),
        help='Number of movies to fetch'
    )
    parser.add_argument(
        '--strategy',
        choices=['popular', 'top_rated', 'mixed'],
        default='mixed',
        help='Movie fetching strategy'
    )
    parser.add_argument(
        '--reviews',
        action='store_true',
        help='Fetch reviews for movies'
    )
    parser.add_argument(
        '--max-reviews',
        type=int,
        default=int(os.getenv('MAX_REVIEWS_PER_MOVIE', 50)),
        help='Maximum reviews per movie'
    )

    args = parser.parse_args()

    try:
        if not args.reviews:
            # Fetch movies
            fetch_and_store_movies(
                num_movies=args.movies,
                fetch_strategy=args.strategy
            )
        else:
            # Fetch reviews
            fetch_reviews_for_movies(max_reviews_per_movie=args.max_reviews)

    except Exception as e:
        logger.error(f"Error during data collection: {e}", exc_info=True)
        sys.exit(1)
