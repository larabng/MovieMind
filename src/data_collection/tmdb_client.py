"""
TMDb API Client for MovieMind
Handles data collection from The Movie Database API
"""

import os
import time
import requests
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class TMDbClient:
    """Client for The Movie Database (TMDb) API"""

    BASE_URL = "https://api.themoviedb.org/3"

    def __init__(self, api_key: str = None):
        """
        Initialize TMDb client

        Args:
            api_key: TMDb API key (optional, will use env variable if not provided)
        """
        self.api_key = api_key or os.getenv('TMDB_API_KEY')
        if not self.api_key:
            raise ValueError("TMDb API key not found. Set TMDB_API_KEY environment variable.")

        self.session = requests.Session()
        self.rate_limit_delay = float(os.getenv('API_RATE_LIMIT_DELAY', '0.5'))

    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """
        Make a request to TMDb API with rate limiting

        Args:
            endpoint: API endpoint
            params: Query parameters

        Returns:
            JSON response or None if failed
        """
        if params is None:
            params = {}

        params['api_key'] = self.api_key

        url = f"{self.BASE_URL}/{endpoint}"

        try:
            time.sleep(self.rate_limit_delay)  # Rate limiting
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None

    def get_popular_movies(self, page: int = 1) -> List[Dict]:
        """
        Get popular movies

        Args:
            page: Page number (1-500)

        Returns:
            List of movie dictionaries
        """
        endpoint = "movie/popular"
        data = self._make_request(endpoint, {'page': page})

        if data and 'results' in data:
            logger.info(f"Fetched {len(data['results'])} popular movies (page {page})")
            return data['results']
        return []

    def get_top_rated_movies(self, page: int = 1) -> List[Dict]:
        """
        Get top rated movies

        Args:
            page: Page number

        Returns:
            List of movie dictionaries
        """
        endpoint = "movie/top_rated"
        data = self._make_request(endpoint, {'page': page})

        if data and 'results' in data:
            logger.info(f"Fetched {len(data['results'])} top rated movies (page {page})")
            return data['results']
        return []

    def discover_movies(self, **kwargs) -> List[Dict]:
        """
        Discover movies with filters

        Args:
            **kwargs: Filter parameters (year, genre_id, sort_by, etc.)

        Returns:
            List of movie dictionaries
        """
        endpoint = "discover/movie"
        data = self._make_request(endpoint, kwargs)

        if data and 'results' in data:
            logger.info(f"Discovered {len(data['results'])} movies")
            return data['results']
        return []

    def get_movie_details(self, movie_id: int) -> Optional[Dict]:
        """
        Get detailed information for a specific movie

        Args:
            movie_id: TMDb movie ID

        Returns:
            Movie details dictionary or None
        """
        endpoint = f"movie/{movie_id}"
        params = {
            'append_to_response': 'credits,keywords,videos,release_dates'
        }

        data = self._make_request(endpoint, params)

        if data:
            logger.info(f"Fetched details for movie ID {movie_id}: {data.get('title')}")
            return data
        return None

    def get_movie_reviews(self, movie_id: int, page: int = 1) -> List[Dict]:
        """
        Get reviews for a specific movie

        Args:
            movie_id: TMDb movie ID
            page: Page number

        Returns:
            List of review dictionaries
        """
        endpoint = f"movie/{movie_id}/reviews"
        data = self._make_request(endpoint, {'page': page})

        if data and 'results' in data:
            logger.info(f"Fetched {len(data['results'])} reviews for movie ID {movie_id}")
            return data['results']
        return []

    def get_all_movie_reviews(self, movie_id: int, max_pages: int = 10) -> List[Dict]:
        """
        Get all available reviews for a movie (up to max_pages)

        Args:
            movie_id: TMDb movie ID
            max_pages: Maximum number of pages to fetch

        Returns:
            List of all review dictionaries
        """
        all_reviews = []
        page = 1

        while page <= max_pages:
            reviews = self.get_movie_reviews(movie_id, page)

            if not reviews:
                break

            all_reviews.extend(reviews)
            page += 1

        logger.info(f"Fetched total of {len(all_reviews)} reviews for movie ID {movie_id}")
        return all_reviews

    def search_movies(self, query: str, page: int = 1) -> List[Dict]:
        """
        Search for movies by title

        Args:
            query: Search query
            page: Page number

        Returns:
            List of movie dictionaries
        """
        endpoint = "search/movie"
        params = {
            'query': query,
            'page': page
        }

        data = self._make_request(endpoint, params)

        if data and 'results' in data:
            logger.info(f"Found {len(data['results'])} movies for query '{query}'")
            return data['results']
        return []

    def format_movie_for_db(self, movie_data: Dict) -> Dict[str, Any]:
        """
        Format movie data for database insertion

        Args:
            movie_data: Raw movie data from API

        Returns:
            Formatted dictionary ready for database
        """
        # Extract genre names
        genres = [g['name'] for g in movie_data.get('genres', [])]

        # Extract production countries
        prod_countries = [c['iso_3166_1'] for c in movie_data.get('production_countries', [])]

        # Extract spoken languages
        spoken_langs = [l['iso_639_1'] for l in movie_data.get('spoken_languages', [])]

        # Parse release date
        release_date = movie_data.get('release_date')
        if release_date:
            try:
                release_date = datetime.strptime(release_date, '%Y-%m-%d').date()
            except ValueError:
                release_date = None

        return {
            'movie_id': movie_data.get('id'),
            'title': movie_data.get('title'),
            'original_title': movie_data.get('original_title'),
            'release_date': release_date,
            'runtime': movie_data.get('runtime'),
            'budget': movie_data.get('budget', 0),
            'revenue': movie_data.get('revenue', 0),
            'genres': genres,
            'production_countries': prod_countries,
            'original_language': movie_data.get('original_language'),
            'spoken_languages': spoken_langs,
            'vote_average': movie_data.get('vote_average'),
            'vote_count': movie_data.get('vote_count'),
            'popularity': movie_data.get('popularity'),
            'overview': movie_data.get('overview'),
            'tagline': movie_data.get('tagline'),
            'status': movie_data.get('status'),
            'adult': movie_data.get('adult', False),
            'video': movie_data.get('video', False),
            'imdb_id': movie_data.get('imdb_id'),
            'homepage': movie_data.get('homepage'),
            'poster_path': movie_data.get('poster_path'),
            'backdrop_path': movie_data.get('backdrop_path')
        }

    def format_review_for_db(self, review_data: Dict, movie_id: int) -> Dict[str, Any]:
        """
        Format review data for database insertion

        Args:
            review_data: Raw review data from API
            movie_id: Associated movie ID

        Returns:
            Formatted dictionary ready for database
        """
        content = review_data.get('content', '')

        # Parse created_at date
        created_at = review_data.get('created_at')
        if created_at:
            try:
                created_at = datetime.strptime(created_at, '%Y-%m-%dT%H:%M:%S.%fZ')
            except ValueError:
                try:
                    created_at = datetime.strptime(created_at, '%Y-%m-%dT%H:%M:%SZ')
                except ValueError:
                    created_at = None

        # Extract rating if available
        rating = None
        author_details = review_data.get('author_details', {})
        if author_details and 'rating' in author_details:
            rating = author_details['rating']

        return {
            'movie_id': movie_id,
            'author': review_data.get('author', 'Anonymous'),
            'content': content,
            'rating': rating,
            'review_date': created_at,
            'language': review_data.get('iso_639_1'),
            'url': review_data.get('url'),
            'text_length': len(content),
            'word_count': len(content.split()) if content else 0
        }


if __name__ == "__main__":
    # Test the client
    client = TMDbClient()

    # Test: Get popular movies
    print("Testing popular movies...")
    movies = client.get_popular_movies(page=1)
    if movies:
        print(f"✓ Found {len(movies)} popular movies")
        print(f"  First movie: {movies[0].get('title')}")

    # Test: Get movie details
    if movies:
        movie_id = movies[0]['id']
        print(f"\nTesting movie details for ID {movie_id}...")
        details = client.get_movie_details(movie_id)
        if details:
            print(f"✓ Got details for: {details.get('title')}")

        # Test: Get reviews
        print(f"\nTesting reviews for movie ID {movie_id}...")
        reviews = client.get_movie_reviews(movie_id)
        print(f"✓ Found {len(reviews)} reviews")
