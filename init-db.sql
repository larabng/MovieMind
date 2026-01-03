-- MovieMind Database Schema
-- PostgreSQL Database Schema for Movie Review Analytics

-- Drop existing tables if they exist (for clean setup)
DROP TABLE IF EXISTS reviews CASCADE;
DROP TABLE IF EXISTS movies CASCADE;
DROP TABLE IF EXISTS countries CASCADE;

-- Countries table (for geo-visualization)
CREATE TABLE countries (
    country_id SERIAL PRIMARY KEY,
    country_code VARCHAR(2) UNIQUE NOT NULL,
    country_name VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Movies table
CREATE TABLE movies (
    movie_id INTEGER PRIMARY KEY,
    title VARCHAR(500) NOT NULL,
    original_title VARCHAR(500),
    release_date DATE,
    runtime INTEGER,
    budget BIGINT,
    revenue BIGINT,

    -- Metadata
    genres TEXT[], -- Array of genre names
    production_countries TEXT[], -- Array of country codes
    original_language VARCHAR(10),
    spoken_languages TEXT[],

    -- Scores and popularity
    vote_average DECIMAL(3,1),
    vote_count INTEGER,
    popularity DECIMAL(10,3),

    -- Text fields
    overview TEXT,
    tagline TEXT,

    -- Additional info
    status VARCHAR(50),
    adult BOOLEAN DEFAULT FALSE,
    video BOOLEAN DEFAULT FALSE,

    -- Technical
    imdb_id VARCHAR(20),
    homepage VARCHAR(500),
    poster_path VARCHAR(200),
    backdrop_path VARCHAR(200),

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Reviews table
CREATE TABLE reviews (
    review_id SERIAL PRIMARY KEY,
    movie_id INTEGER NOT NULL REFERENCES movies(movie_id) ON DELETE CASCADE,

    -- Review content
    author VARCHAR(200),
    content TEXT NOT NULL,

    -- Ratings
    rating DECIMAL(3,1), -- User rating (if available)

    -- Metadata
    review_date TIMESTAMP,
    language VARCHAR(10),
    url VARCHAR(500),

    -- Processed fields (will be filled by NLP pipeline)
    sentiment VARCHAR(20), -- 'positive', 'neutral', 'negative'
    sentiment_score DECIMAL(5,4), -- -1 to 1
    predicted_rating DECIMAL(3,1), -- ML predicted score
    text_length INTEGER,
    word_count INTEGER,

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CONSTRAINT unique_review UNIQUE (movie_id, author, content)
);

-- Create indexes for better query performance
CREATE INDEX idx_movies_release_date ON movies(release_date);
CREATE INDEX idx_movies_genres ON movies USING GIN(genres);
CREATE INDEX idx_movies_vote_average ON movies(vote_average);
CREATE INDEX idx_reviews_movie_id ON reviews(movie_id);
CREATE INDEX idx_reviews_sentiment ON reviews(sentiment);
CREATE INDEX idx_reviews_review_date ON reviews(review_date);

-- Create view for analysis
CREATE OR REPLACE VIEW movie_review_stats AS
SELECT
    m.movie_id,
    m.title,
    m.release_date,
    m.genres,
    m.runtime,
    m.vote_average as tmdb_rating,
    m.vote_count as tmdb_votes,
    COUNT(r.review_id) as review_count,
    AVG(r.sentiment_score) as avg_sentiment,
    SUM(CASE WHEN r.sentiment = 'positive' THEN 1 ELSE 0 END) as positive_reviews,
    SUM(CASE WHEN r.sentiment = 'neutral' THEN 1 ELSE 0 END) as neutral_reviews,
    SUM(CASE WHEN r.sentiment = 'negative' THEN 1 ELSE 0 END) as negative_reviews,
    AVG(r.predicted_rating) as avg_predicted_rating,
    AVG(r.text_length) as avg_review_length
FROM movies m
LEFT JOIN reviews r ON m.movie_id = r.movie_id
GROUP BY m.movie_id, m.title, m.release_date, m.genres, m.runtime, m.vote_average, m.vote_count;

-- Create view for genre analysis
CREATE OR REPLACE VIEW genre_sentiment_analysis AS
SELECT
    unnest(m.genres) as genre,
    COUNT(DISTINCT m.movie_id) as movie_count,
    COUNT(r.review_id) as review_count,
    AVG(r.sentiment_score) as avg_sentiment,
    SUM(CASE WHEN r.sentiment = 'positive' THEN 1 ELSE 0 END)::FLOAT /
        NULLIF(COUNT(r.review_id), 0) as positive_ratio,
    SUM(CASE WHEN r.sentiment = 'negative' THEN 1 ELSE 0 END)::FLOAT /
        NULLIF(COUNT(r.review_id), 0) as negative_ratio,
    AVG(m.vote_average) as avg_tmdb_rating
FROM movies m
LEFT JOIN reviews r ON m.movie_id = r.movie_id
GROUP BY genre
ORDER BY review_count DESC;

-- Create view for temporal analysis
CREATE OR REPLACE VIEW temporal_sentiment_trends AS
SELECT
    DATE_TRUNC('month', r.review_date) as review_month,
    COUNT(r.review_id) as review_count,
    AVG(r.sentiment_score) as avg_sentiment,
    SUM(CASE WHEN r.sentiment = 'positive' THEN 1 ELSE 0 END) as positive_count,
    SUM(CASE WHEN r.sentiment = 'neutral' THEN 1 ELSE 0 END) as neutral_count,
    SUM(CASE WHEN r.sentiment = 'negative' THEN 1 ELSE 0 END) as negative_count
FROM reviews r
WHERE r.review_date IS NOT NULL
GROUP BY review_month
ORDER BY review_month;

-- Insert some common countries for geo-visualization
INSERT INTO countries (country_code, country_name) VALUES
('US', 'United States'),
('GB', 'United Kingdom'),
('FR', 'France'),
('DE', 'Germany'),
('IT', 'Italy'),
('ES', 'Spain'),
('JP', 'Japan'),
('KR', 'South Korea'),
('CN', 'China'),
('IN', 'India'),
('CA', 'Canada'),
('AU', 'Australia'),
('BR', 'Brazil'),
('MX', 'Mexico'),
('RU', 'Russia')
ON CONFLICT (country_code) DO NOTHING;

-- Grant permissions (adjust user as needed)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO moviemind_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO moviemind_user;

COMMENT ON TABLE movies IS 'Stores movie metadata from TMDb API';
COMMENT ON TABLE reviews IS 'Stores user reviews with sentiment analysis results';
COMMENT ON VIEW movie_review_stats IS 'Aggregated statistics per movie for analysis';
COMMENT ON VIEW genre_sentiment_analysis IS 'Sentiment breakdown by genre';
COMMENT ON VIEW temporal_sentiment_trends IS 'Sentiment trends over time';
