"""
MovieMind Demo Dashboard
Interactive web application for sentiment analysis and score prediction
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from flask import Flask, render_template, request, jsonify
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from src.preprocessing.text_processor import TextProcessor
from src.models.sentiment_classifier import SentimentClassifier
from src.models.score_predictor import ScorePredictor
from src.utils.db_manager import DatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask server
server = Flask(__name__)

# Initialize Dash app
app = dash.Dash(
    __name__,
    server=server,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

# Initialize components
text_processor = TextProcessor()

# Try to load pre-trained models (if available)
sentiment_model = None
score_model = None

try:
    sentiment_model = SentimentClassifier()
    sentiment_model.load_model('models/sentiment_classifier')
    logger.info("Sentiment model loaded successfully")
except Exception as e:
    logger.warning(f"Could not load sentiment model: {e}")

try:
    score_model = ScorePredictor()
    score_model.load_model('models/score_predictor')
    logger.info("Score predictor loaded successfully")
except Exception as e:
    logger.warning(f"Could not load score model: {e}")


# App Layout
app.layout = dbc.Container([
    dbc.NavbarSimple(
        brand="MovieMind - Movie Review Analytics",
        brand_href="/",
        color="primary",
        dark=True,
        className="mb-4"
    ),

    dbc.Tabs([
        # Tab 1: Live Prediction
        dbc.Tab(label="Live Prediction", tab_id="tab-prediction", children=[
            dbc.Row([
                dbc.Col([
                    html.H3("Analyze Your Review", className="mt-4 mb-3"),

                    dbc.Card([
                        dbc.CardBody([
                            html.Label("Enter Movie Review Text:"),
                            dbc.Textarea(
                                id='review-input',
                                placeholder='Write your movie review here...',
                                style={'width': '100%', 'height': 150},
                                className="mb-3"
                            ),

                            dbc.Row([
                                dbc.Col([
                                    html.Label("Genre (optional):"),
                                    dcc.Dropdown(
                                        id='genre-dropdown',
                                        options=[
                                            {'label': 'Action', 'value': 'Action'},
                                            {'label': 'Comedy', 'value': 'Comedy'},
                                            {'label': 'Drama', 'value': 'Drama'},
                                            {'label': 'Horror', 'value': 'Horror'},
                                            {'label': 'Romance', 'value': 'Romance'},
                                            {'label': 'Sci-Fi', 'value': 'Sci-Fi'},
                                            {'label': 'Thriller', 'value': 'Thriller'},
                                        ],
                                        placeholder="Select genre"
                                    )
                                ], width=6),

                                dbc.Col([
                                    html.Label("Runtime (minutes, optional):"),
                                    dbc.Input(
                                        id='runtime-input',
                                        type='number',
                                        placeholder='e.g., 120',
                                        min=0,
                                        max=300
                                    )
                                ], width=6)
                            ], className="mb-3"),

                            dbc.Button(
                                "Analyze Review",
                                id='analyze-button',
                                color="primary",
                                size="lg",
                                className="w-100"
                            )
                        ])
                    ]),

                    # Results Section
                    html.Div(id='prediction-results', className="mt-4")

                ], width=12)
            ])
        ]),

        # Tab 2: Database Statistics
        dbc.Tab(label="Database Statistics", tab_id="tab-stats", children=[
            dbc.Row([
                dbc.Col([
                    html.H3("Database Overview", className="mt-4 mb-3"),
                    dbc.Button("Refresh Stats", id="refresh-stats-button", color="info", className="mb-3"),
                    html.Div(id='stats-content')
                ], width=12)
            ])
        ]),

        # Tab 3: Visualizations
        dbc.Tab(label="Visualizations", tab_id="tab-viz", children=[
            dbc.Row([
                dbc.Col([
                    html.H3("Data Visualizations", className="mt-4 mb-3"),
                    dbc.Button("Load Charts", id="load-viz-button", color="info", className="mb-3"),
                    html.Div(id='viz-content')
                ], width=12)
            ])
        ])
    ], id="tabs", active_tab="tab-prediction")

], fluid=True)


# Callbacks
@app.callback(
    Output('prediction-results', 'children'),
    Input('analyze-button', 'n_clicks'),
    State('review-input', 'value'),
    State('genre-dropdown', 'value'),
    State('runtime-input', 'value'),
    prevent_initial_call=True
)
def analyze_review(n_clicks, review_text, genre, runtime):
    """Analyze review and return predictions"""

    if not review_text or len(review_text.strip()) == 0:
        return dbc.Alert("Please enter a review text!", color="warning")

    try:
        # Clean text
        cleaned_text = text_processor.clean_text(review_text)

        # Extract features
        features = text_processor.extract_features(review_text)

        # Predict sentiment
        sentiment = "neutral"
        sentiment_proba = None
        if sentiment_model and sentiment_model.is_trained:
            sentiment_pred = sentiment_model.predict(pd.Series([cleaned_text]))
            sentiment = sentiment_pred[0]
            sentiment_proba = sentiment_model.predict_proba(pd.Series([cleaned_text]))[0]

        # Predict score
        predicted_score = None
        if score_model and score_model.is_trained:
            # Prepare metadata if available
            meta_df = pd.DataFrame([{
                'text_length': features['text_length'],
                'word_count': features['word_count']
            }])

            predicted_score = score_model.predict(pd.Series([cleaned_text]), meta_df)[0]

        # Build results
        results = dbc.Card([
            dbc.CardHeader(html.H4("Analysis Results")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H5("Sentiment"),
                        html.H3(
                            sentiment.upper(),
                            className="text-success" if sentiment == "positive"
                            else "text-danger" if sentiment == "negative"
                            else "text-warning"
                        ),
                        html.P(f"Confidence: {max(sentiment_proba)*100:.1f}%" if sentiment_proba is not None else "N/A")
                    ], width=6),

                    dbc.Col([
                        html.H5("Predicted Score"),
                        html.H3(f"{predicted_score:.1f} / 10" if predicted_score else "N/A"),
                        html.P("Based on text analysis")
                    ], width=6)
                ]),

                html.Hr(),

                html.H5("Text Statistics"),
                dbc.Row([
                    dbc.Col([
                        html.P(f"Character count: {features['text_length']}"),
                        html.P(f"Word count: {features['word_count']}"),
                        html.P(f"Unique words: {features['unique_word_count']}"),
                    ], width=6),
                    dbc.Col([
                        html.P(f"Sentences: {features['sentence_count']}"),
                        html.P(f"Avg word length: {features['avg_word_length']:.1f}"),
                        html.P(f"Exclamations: {features['exclamation_count']}")
                    ], width=6)
                ]),

                html.Hr(),

                html.H5("Cleaned Text Preview"),
                html.P(cleaned_text[:500] + "..." if len(cleaned_text) > 500 else cleaned_text,
                      className="text-muted",
                      style={'font-style': 'italic'})
            ])
        ], className="shadow")

        return results

    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        return dbc.Alert(f"Error during analysis: {str(e)}", color="danger")


@app.callback(
    Output('stats-content', 'children'),
    Input('refresh-stats-button', 'n_clicks'),
    prevent_initial_call=True
)
def load_stats(n_clicks):
    """Load and display database statistics"""

    try:
        with DatabaseManager() as db:
            # Get movie count
            movie_count = db.execute_query("SELECT COUNT(*) as count FROM movies")[0]['count']

            # Get review count
            review_count = db.execute_query("SELECT COUNT(*) as count FROM reviews")[0]['count']

            # Get average rating
            avg_rating = db.execute_query(
                "SELECT AVG(vote_average) as avg FROM movies WHERE vote_average IS NOT NULL"
            )[0]['avg']

            # Get genre distribution
            genre_stats = db.execute_query("""
                SELECT unnest(genres) as genre, COUNT(*) as count
                FROM movies
                WHERE genres IS NOT NULL
                GROUP BY genre
                ORDER BY count DESC
                LIMIT 10
            """)

        # Build stats cards
        stats_cards = dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(movie_count, className="card-title text-primary"),
                        html.P("Total Movies", className="card-text")
                    ])
                ], className="text-center shadow-sm")
            ], width=4),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(review_count, className="card-title text-success"),
                        html.P("Total Reviews", className="card-text")
                    ])
                ], className="text-center shadow-sm")
            ], width=4),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{float(avg_rating):.2f}" if avg_rating else "N/A",
                               className="card-title text-warning"),
                        html.P("Average Rating", className="card-text")
                    ])
                ], className="text-center shadow-sm")
            ], width=4)
        ], className="mb-4")

        # Genre table
        genre_df = pd.DataFrame(genre_stats)
        genre_table = dbc.Table.from_dataframe(
            genre_df,
            striped=True,
            bordered=True,
            hover=True,
            className="mt-4"
        )

        return html.Div([
            stats_cards,
            html.H5("Top 10 Genres", className="mt-4"),
            genre_table
        ])

    except Exception as e:
        logger.error(f"Error loading stats: {e}", exc_info=True)
        return dbc.Alert(f"Error loading statistics: {str(e)}", color="danger")


@app.callback(
    Output('viz-content', 'children'),
    Input('load-viz-button', 'n_clicks'),
    prevent_initial_call=True
)
def load_visualizations(n_clicks):
    """Load and display visualizations"""

    try:
        with DatabaseManager() as db:
            # Get rating distribution
            ratings = db.execute_query("""
                SELECT vote_average
                FROM movies
                WHERE vote_average IS NOT NULL
                LIMIT 1000
            """)

            df_ratings = pd.DataFrame(ratings)

        # Create histogram
        fig_hist = px.histogram(
            df_ratings,
            x='vote_average',
            nbins=30,
            title='Distribution of Movie Ratings',
            labels={'vote_average': 'Rating', 'count': 'Frequency'}
        )

        # Create box plot
        fig_box = px.box(
            df_ratings,
            y='vote_average',
            title='Box Plot of Ratings',
            labels={'vote_average': 'Rating'}
        )

        return html.Div([
            dcc.Graph(figure=fig_hist),
            dcc.Graph(figure=fig_box)
        ])

    except Exception as e:
        logger.error(f"Error loading visualizations: {e}", exc_info=True)
        return dbc.Alert(f"Error loading visualizations: {str(e)}", color="danger")


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=5000)
