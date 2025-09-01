# Flask Sentiment Analysis Web Application
# A comprehensive web app for sentiment analysis using TextBlob

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from textblob import TextBlob
import datetime
import json
import os
from werkzeug.utils import secure_filename
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import sqlite3
from contextlib import contextmanager
import re
from collections import Counter
import tempfile
# Add this code right after your imports in Sentiment_Analyzer.py

import nltk
import ssl

# Fix SSL certificate issues if any
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data
def download_nltk_data():
    """Download required NLTK data for TextBlob."""
    try:
        # Required for TextBlob sentiment analysis
        nltk.download('punkt', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('brown', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        
        # Required for POS tagging (part-of-speech)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        
        # Required for language detection and additional features
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        
        print("✅ NLTK data downloaded successfully!")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not download some NLTK data: {e}")
        print("You may need to run: python -c 'import nltk; nltk.download_shell()'")

# Download NLTK data when the module is imported
download_nltk_data()

# ============================================================================
# APPLICATION SETUP
# ============================================================================

app = Flask(__name__)
app.secret_key = 'sentiment_analysis_secret_key_2024'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configuration
DATABASE_PATH = 'sentiment_history.db'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'csv'}

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ============================================================================
# DATABASE SETUP
# ============================================================================

@contextmanager
def get_db_connection():
    """Context manager for database connections."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_database():
    """Initialize the database with required tables."""
    with get_db_connection() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS sentiment_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                polarity REAL NOT NULL,
                subjectivity REAL NOT NULL,
                sentiment TEXT NOT NULL,
                confidence TEXT NOT NULL,
                word_count INTEGER NOT NULL,
                sentence_count INTEGER DEFAULT 1,
                detected_language TEXT DEFAULT 'unknown',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()

# ============================================================================
# SENTIMENT ANALYSIS CORE FUNCTIONS
# ============================================================================

class SentimentAnalyzer:
    """Enhanced sentiment analysis using TextBlob with additional metrics."""
    
    @staticmethod
    def analyze_text(text):
        """Perform comprehensive sentiment analysis on text."""
        if not text or not text.strip():
            return None
        
        # Create TextBlob object
        blob = TextBlob(text)
        
        # Basic sentiment scores
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Classify sentiment
        sentiment = SentimentAnalyzer.classify_sentiment(polarity)
        
        # Calculate confidence level
        confidence = SentimentAnalyzer.calculate_confidence(polarity)
        
        # Additional metrics
        word_count = len(blob.words)
        sentence_count = len(blob.sentences)
        
        # Language detection
        try:
            detected_language = blob.detect_language()
        except:
            detected_language = 'unknown'
        
        # Extract key phrases (nouns and adjectives)
        key_phrases = SentimentAnalyzer.extract_key_phrases(blob)
        
        return {
            'text': text,
            'polarity': round(polarity, 4),
            'subjectivity': round(subjectivity, 4),
            'sentiment': sentiment,
            'confidence': confidence,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'detected_language': detected_language,
            'key_phrases': key_phrases,
            'timestamp': datetime.datetime.now().isoformat()
        }
    
    @staticmethod
    def classify_sentiment(polarity):
        """Classify sentiment based on polarity score."""
        if polarity > 0.1:
            return 'Positive'
        elif polarity < -0.1:
            return 'Negative'
        else:
            return 'Neutral'
    
    @staticmethod
    def calculate_confidence(polarity):
        """Calculate confidence level based on polarity magnitude."""
        abs_polarity = abs(polarity)
        
        if abs_polarity >= 0.5:
            return 'High'
        elif abs_polarity >= 0.2:
            return 'Medium'
        else:
            return 'Low'
    
    @staticmethod
    def extract_key_phrases(blob):
        """Extract important nouns and adjectives from the text."""
        try:
            # Get part-of-speech tags
            pos_tags = blob.tags
            
            # Extract nouns and adjectives
            key_words = []
            for word, pos in pos_tags:
                if pos in ['NN', 'NNP', 'NNS', 'NNPS', 'JJ', 'JJR', 'JJS']:
                    key_words.append(word.lower())
            
            # Get most common key phrases
            word_freq = Counter(key_words)
            return [word for word, count in word_freq.most_common(10)]
        except:
            return []

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_sentiment_chart(analysis_results):
    """Create a sentiment analysis visualization."""
    if not analysis_results:
        return None
    
    try:
        # Set style with fallback
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('default')
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Polarity and Subjectivity Scatter Plot
        polarity = analysis_results['polarity']
        subjectivity = analysis_results['subjectivity']
        sentiment = analysis_results['sentiment']
        
        color_map = {'Positive': 'green', 'Negative': 'red', 'Neutral': 'gray'}
        color = color_map.get(sentiment, 'blue')
        
        ax1.scatter([polarity], [subjectivity], c=color, s=200, alpha=0.7, edgecolors='black')
        ax1.set_xlabel('Polarity (-1 to 1)')
        ax1.set_ylabel('Subjectivity (0 to 1)')
        ax1.set_title('Polarity vs Subjectivity')
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax1.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
        ax1.set_xlim(-1, 1)
        ax1.set_ylim(0, 1)
        
        # Add quadrant labels
        ax1.text(0.7, 0.8, 'Subjective\nPositive', ha='center', va='center', alpha=0.6)
        ax1.text(-0.7, 0.8, 'Subjective\nNegative', ha='center', va='center', alpha=0.6)
        ax1.text(0.7, 0.2, 'Objective\nPositive', ha='center', va='center', alpha=0.6)
        ax1.text(-0.7, 0.2, 'Objective\nNegative', ha='center', va='center', alpha=0.6)
        
        # 2. Sentiment Classification Pie Chart
        sentiments = ['Positive', 'Negative', 'Neutral']
        current_sentiment = [1 if s == sentiment else 0 for s in sentiments]
        colors = ['#2ecc71', '#e74c3c', '#95a5a6']
        
        # Only show non-zero values
        non_zero_sentiments = [s for i, s in enumerate(sentiments) if current_sentiment[i] > 0]
        non_zero_values = [v for v in current_sentiment if v > 0]
        non_zero_colors = [colors[i] for i, v in enumerate(current_sentiment) if v > 0]
        
        if non_zero_values:
            ax2.pie(non_zero_values, labels=non_zero_sentiments, colors=non_zero_colors, 
                    autopct='%1.0f%%', startangle=90)
        ax2.set_title('Sentiment Classification')
        
        # 3. Confidence Level Bar Chart
        confidence_levels = ['Low', 'Medium', 'High']
        confidence_values = [1 if analysis_results['confidence'] == level else 0 for level in confidence_levels]
        colors_conf = ['#f39c12', '#3498db', '#27ae60']
        
        bars = ax3.bar(confidence_levels, confidence_values, color=colors_conf, alpha=0.7)
        ax3.set_ylabel('Confidence')
        ax3.set_title('Confidence Level')
        ax3.set_ylim(0, 1.2)
        
        # Highlight current confidence
        for i, bar in enumerate(bars):
            if confidence_levels[i] == analysis_results['confidence']:
                bar.set_alpha(1.0)
                bar.set_edgecolor('black')
                bar.set_linewidth(2)
        
        # 4. Text Statistics
        stats_labels = ['Words', 'Sentences']
        stats_values = [analysis_results['word_count'], analysis_results['sentence_count']]
        
        bars_stats = ax4.bar(stats_labels, stats_values, color=['#9b59b6', '#34495e'], alpha=0.7)
        ax4.set_ylabel('Count')
        ax4.set_title('Text Statistics')
        
        # Add value labels on bars
        for bar, value in zip(bars_stats, stats_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(stats_values)*0.01,
                    f'{value}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Convert plot to base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plot_data = buffer.getvalue()
        buffer.close()
        plt.close()
        
        plot_url = base64.b64encode(plot_data).decode()
        return plot_url
    
    except Exception as e:
        print(f"Error creating chart: {e}")
        return None

def create_history_chart():
    """Create a chart showing sentiment analysis history."""
    try:
        with get_db_connection() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM sentiment_analysis ORDER BY timestamp DESC LIMIT 50", 
                conn
            )
        
        if df.empty:
            return None
        
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('default')
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Sentiment distribution
        sentiment_counts = df['sentiment'].value_counts()
        colors = ['#2ecc71', '#e74c3c', '#95a5a6']
        
        # Map colors to sentiments
        sentiment_colors = []
        for sentiment in sentiment_counts.index:
            if sentiment == 'Positive':
                sentiment_colors.append('#2ecc71')
            elif sentiment == 'Negative':
                sentiment_colors.append('#e74c3c')
            else:
                sentiment_colors.append('#95a5a6')
        
        ax1.pie(sentiment_counts.values, labels=sentiment_counts.index, colors=sentiment_colors,
                autopct='%1.1f%%', startangle=90)
        ax1.set_title('Overall Sentiment Distribution')
        
        # 2. Polarity over time
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df_sorted = df.sort_values('timestamp')
        
        ax2.plot(range(len(df_sorted)), df_sorted['polarity'], marker='o', alpha=0.7, linewidth=2)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Analysis Number')
        ax2.set_ylabel('Polarity Score')
        ax2.set_title('Polarity Trend Over Time')
        ax2.grid(True, alpha=0.3)
        
        # 3. Confidence level distribution
        confidence_counts = df['confidence'].value_counts()
        conf_colors = ['#f39c12', '#3498db', '#27ae60']
        
        # Map colors to confidence levels
        confidence_color_map = {'Low': '#f39c12', 'Medium': '#3498db', 'High': '#27ae60'}
        conf_colors_mapped = [confidence_color_map.get(conf, '#95a5a6') for conf in confidence_counts.index]
        
        ax3.bar(confidence_counts.index, confidence_counts.values, color=conf_colors_mapped, alpha=0.7)
        ax3.set_xlabel('Confidence Level')
        ax3.set_ylabel('Count')
        ax3.set_title('Confidence Level Distribution')
        
        # 4. Word count vs Polarity scatter
        colors_scatter = ['green' if s == 'Positive' else 'red' if s == 'Negative' else 'gray' 
                         for s in df['sentiment']]
        
        ax4.scatter(df['word_count'], df['polarity'], c=colors_scatter, alpha=0.6, s=50)
        ax4.set_xlabel('Word Count')
        ax4.set_ylabel('Polarity Score')
        ax4.set_title('Word Count vs Polarity')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plot_data = buffer.getvalue()
        buffer.close()
        plt.close()
        
        return base64.b64encode(plot_data).decode()
    
    except Exception as e:
        print(f"Error creating history chart: {e}")
        return None

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_analysis_to_db(analysis_result):
    """Save analysis result to database."""
    try:
        with get_db_connection() as conn:
            conn.execute('''
                INSERT INTO sentiment_analysis 
                (text, polarity, subjectivity, sentiment, confidence, word_count, sentence_count, detected_language)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                analysis_result['text'],
                analysis_result['polarity'],
                analysis_result['subjectivity'],
                analysis_result['sentiment'],
                analysis_result['confidence'],
                analysis_result['word_count'],
                analysis_result['sentence_count'],
                analysis_result.get('detected_language', 'unknown')
            ))
            conn.commit()
    except Exception as e:
        print(f"Error saving to database: {e}")

def get_analysis_history(limit=50):
    """Get recent analysis history from database."""
    try:
        with get_db_connection() as conn:
            cursor = conn.execute('''
                SELECT * FROM sentiment_analysis 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            return [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        print(f"Error getting history: {e}")
        return []

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def index():
    """Home page with sentiment analysis form."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze sentiment of submitted text."""
    try:
        text = request.form.get('text', '').strip()
        
        if not text:
            flash('Please enter some text to analyze.', 'error')
            return redirect(url_for('index'))
        
        if len(text) > 10000:
            flash('Text is too long. Please limit to 10,000 characters.', 'error')
            return redirect(url_for('index'))
        
        # Perform sentiment analysis
        analysis_result = SentimentAnalyzer.analyze_text(text)
        
        if not analysis_result:
            flash('Error analyzing text. Please try again.', 'error')
            return redirect(url_for('index'))
        
        # Save to database
        save_analysis_to_db(analysis_result)
        
        # Create visualization
        chart_url = create_sentiment_chart(analysis_result)
        
        return render_template('results.html', 
                             analysis=analysis_result,
                             chart_url=chart_url)
    
    except Exception as e:
        flash(f'An error occurred during analysis: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for sentiment analysis."""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text'].strip()
        
        if not text:
            return jsonify({'error': 'Empty text provided'}), 400
        
        if len(text) > 10000:
            return jsonify({'error': 'Text too long (max 10,000 characters)'}), 400
        
        # Perform analysis
        analysis_result = SentimentAnalyzer.analyze_text(text)
        
        if not analysis_result:
            return jsonify({'error': 'Analysis failed'}), 500
        
        # Save to database
        save_analysis_to_db(analysis_result)
        
        return jsonify(analysis_result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch')
def batch_analysis():
    """Page for batch text analysis."""
    return render_template('batch.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload for batch analysis."""
    try:
        if 'file' not in request.files:
            flash('No file selected.', 'error')
            return redirect(url_for('batch_analysis'))
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected.', 'error')
            return redirect(url_for('batch_analysis'))
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process file
            results = process_uploaded_file(filepath)
            
            # Clean up uploaded file
            try:
                os.remove(filepath)
            except:
                pass
            
            if not results:
                flash('No valid text found in file.', 'error')
                return redirect(url_for('batch_analysis'))
            
            # Save results to database
            for result in results:
                save_analysis_to_db(result)
            
            return render_template('batch_results.html', results=results)
        
        else:
            flash('Invalid file type. Please upload a .txt or .csv file.', 'error')
            return redirect(url_for('batch_analysis'))
    
    except Exception as e:
        flash(f'Error processing file: {str(e)}', 'error')
        return redirect(url_for('batch_analysis'))

def process_uploaded_file(filepath):
    """Process uploaded file and return analysis results."""
    results = []
    
    try:
        if filepath.endswith('.txt'):
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Split into sentences or paragraphs for analysis
            sentences = content.split('\n')
            sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
            
            for sentence in sentences[:50]:  # Limit to 50 items
                analysis = SentimentAnalyzer.analyze_text(sentence)
                if analysis:
                    results.append(analysis)
        
        elif filepath.endswith('.csv'):
            try:
                df = pd.read_csv(filepath, encoding='utf-8', errors='ignore')
            except:
                df = pd.read_csv(filepath, encoding='latin-1', errors='ignore')
            
            # Find text columns
            text_columns = []
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Check if column contains text data
                    sample_values = df[col].dropna().head(10)
                    if len(sample_values) > 0:
                        avg_length = sample_values.astype(str).str.len().mean()
                        if avg_length > 20:  # Assume columns with avg length > 20 chars are text
                            text_columns.append(col)
            
            if not text_columns:
                return results
            
            # Use the first text column found
            text_column = text_columns[0]
            texts = df[text_column].dropna().head(100).tolist()  # Limit to 100 items
            
            for text in texts:
                if isinstance(text, str) and len(text.strip()) > 10:
                    analysis = SentimentAnalyzer.analyze_text(text)
                    if analysis:
                        results.append(analysis)
    
    except Exception as e:
        print(f"Error processing file: {e}")
    
    return results

@app.route('/history')
def history():
    """View analysis history."""
    try:
        history_data = get_analysis_history(100)
        chart_url = create_history_chart()
        
        return render_template('history.html', 
                             history=history_data,
                             chart_url=chart_url)
    
    except Exception as e:
        flash(f'Error loading history: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/api/history')
def api_history():
    """API endpoint for getting analysis history."""
    try:
        limit = request.args.get('limit', 50, type=int)
        limit = min(limit, 200)  # Max 200 results
        
        history_data = get_analysis_history(limit)
        return jsonify(history_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear analysis history."""
    try:
        with get_db_connection() as conn:
            conn.execute('DELETE FROM sentiment_analysis')
            conn.commit()
        
        flash('History cleared successfully!', 'success')
        return redirect(url_for('history'))
    
    except Exception as e:
        flash(f'Error clearing history: {str(e)}', 'error')
        return redirect(url_for('history'))

@app.route('/stats')
def stats():
    """View detailed statistics."""
    try:
        with get_db_connection() as conn:
            # Get overall statistics
            total_query = 'SELECT COUNT(*) as total FROM sentiment_analysis'
            cursor = conn.execute(total_query)
            total_count = cursor.fetchone()['total']
            
            if total_count == 0:
                return render_template('stats.html', 
                                     sentiment_stats=[],
                                     daily_stats=[],
                                     total_count=0)
            
            # Get sentiment statistics
            sentiment_query = '''
                SELECT 
                    sentiment,
                    COUNT(*) as sentiment_count,
                    AVG(polarity) as avg_polarity,
                    AVG(subjectivity) as avg_subjectivity
                FROM sentiment_analysis 
                GROUP BY sentiment
            '''
            cursor = conn.execute(sentiment_query)
            sentiment_stats = cursor.fetchall()
            
            # Get daily statistics for the last 7 days
            daily_query = '''
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(*) as daily_count,
                    AVG(polarity) as daily_avg_polarity
                FROM sentiment_analysis 
                WHERE timestamp >= date('now', '-7 days')
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
            '''
            cursor = conn.execute(daily_query)
            daily_stats = cursor.fetchall()
        
        return render_template('stats.html', 
                             sentiment_stats=sentiment_stats,
                             daily_stats=daily_stats,
                             total_count=total_count)
    
    except Exception as e:
        flash(f'Error loading statistics: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', 
                         error_code=404,
                         error_message="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', 
                         error_code=500,
                         error_message="Internal server error"), 500

# ============================================================================
# TEMPLATE CREATION
# ============================================================================

def create_missing_templates():
    """Create missing HTML templates."""
    
    os.makedirs('templates', exist_ok=True)
    
    # Base template
    base_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Sentiment Analysis App{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .sentiment-positive { background-color: #d4edda; color: #155724; }
        .sentiment-negative { background-color: #f8d7da; color: #721c24; }
        .sentiment-neutral { background-color: #e2e3e5; color: #383d41; }
        .confidence-high { background-color: #d4edda; }
        .confidence-medium { background-color: #fff3cd; }
        .confidence-low { background-color: #f8d7da; }
        .navbar-brand { font-weight: bold; }
        .card-hover:hover { transform: translateY(-5px); transition: transform 0.3s; }
        .stats-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
        .chart-container { max-width: 100%; overflow-x: auto; }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="{{ url_for('index') }}">
                <i class="fas fa-brain"></i> Sentiment Analyzer
            </a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav ms-auto">
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('index') }}"><i class="fas fa-home"></i> Home</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('batch_analysis') }}"><i class="fas fa-upload"></i> Batch Analysis</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('history') }}"><i class="fas fa-history"></i> History</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('stats') }}"><i class="fas fa-chart-bar"></i> Statistics</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <main class="container mt-4">
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                    <div class="alert alert-{{ 'danger' if category == 'error' else 'success' }} alert-dismissible fade show">
                        {{ message }}
                        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                    </div>
                {% endfor %}
            {% endif %}
        {% endwith %}

        {% block content %}{% endblock %}
    </main>

    <footer class="bg-light mt-5 py-4">
        <div class="container text-center">
            <p class="text-muted">&copy; 2024 Sentiment Analysis App. Built with Flask and TextBlob.</p>
        </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    {% block scripts %}{% endblock %}
</body>
</html>'''
    
    # Create base template
    with open('templates/base.html', 'w', encoding='utf-8') as f:
        f.write(base_template)
    
    # Batch results template
    batch_results_template = '''{% extends "base.html" %}

{% block title %}Batch Analysis Results - Sentiment Analysis App{% endblock %}

{% block content %}
<div class="row">
    <div class="col-12">
        <div class="d-flex justify-content-between align-items-center mb-4">
            <h1><i class="fas fa-chart-line"></i> Batch Analysis Results</h1>
            <a href="{{ url_for('batch_analysis') }}" class="btn btn-primary">
                <i class="fas fa-upload"></i> New Batch Analysis
            </a>
        </div>
    </div>
</div>

<div class="row mb-4">
    <div class="col-12">
        <div class="alert alert-success">
            <h5><i class="fas fa-check-circle"></i> Analysis Complete!</h5>
            <p class="mb-0">Successfully analyzed {{ results|length }} text(s) from your uploaded file.</p>
        </div>
    </div>
</div>

<div class="row">
    <div class="col-12">
        <div class="card shadow">
            <div class="card-header bg-info text-white">
                <h5 class="mb-0"><i class="fas fa-list"></i> Analysis Results</h5>
            </div>
            <div class="card-body p-0">
                <div class="table-responsive">
                    <table class="table table-hover mb-0">
                        <thead class="table-light">
                            <tr>
                                <th>#</th>
                                <th>Text Preview</th>
                                <th>Sentiment</th>
                                <th>Polarity</th>
                                <th>Subjectivity</th>
                                <th>Words</th>
                                <th>Confidence</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for result in results %}
                            <tr>
                                <td>{{ loop.index }}</td>
                                <td>
                                    <div style="max-width: 300px;">
                                        {{ result.text[:80] }}{% if result.text|length > 80 %}...{% endif %}
                                    </div>
                                </td>
                                <td>
                                    <span class="badge sentiment-{{ result.sentiment.lower() }}">
                                        {{ result.sentiment }}
                                    </span>
                                </td>
                                <td>
                                    <span class="badge bg-{{ 'success' if result.polarity > 0 else 'danger' if result.polarity < 0 else 'secondary' }}">
                                        {{ result.polarity }}
                                    </span>
                                </td>
                                <td>{{ result.subjectivity }}</td>
                                <td>{{ result.word_count }}</td>
                                <td>
                                    <span class="badge confidence-{{ result.confidence.lower() }} text-dark">
                                        {{ result.confidence }}
                                    </span>
                                </td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
</div>

<div class="row mt-4">
    <div class="col-12">
        <div class="card shadow">
            <div class="card-header bg-secondary text-white">
                <h5 class="mb-0"><i class="fas fa-chart-bar"></i> Summary Statistics</h5>
            </div>
            <div class="card-body">
                {% set positive_count = results | selectattr('sentiment', 'equalto', 'Positive') | list | length %}
                {% set negative_count = results | selectattr('sentiment', 'equalto', 'Negative') | list | length %}
                {% set neutral_count = results | selectattr('sentiment', 'equalto', 'Neutral') | list | length %}
                {% set avg_polarity = (results | sum(attribute='polarity')) / results|length %}
                {% set avg_subjectivity = (results | sum(attribute='subjectivity')) / results|length %}
                
                <div class="row text-center">
                    <div class="col-md-2">
                        <h4 class="text-success">{{ positive_count }}</h4>
                        <small>Positive</small>
                    </div>
                    <div class="col-md-2">
                        <h4 class="text-danger">{{ negative_count }}</h4>
                        <small>Negative</small>
                    </div>
                    <div class="col-md-2">
                        <h4 class="text-secondary">{{ neutral_count }}</h4>
                        <small>Neutral</small>
                    </div>
                    <div class="col-md-3">
                        <h4 class="text-info">{{ "%.3f"|format(avg_polarity) }}</h4>
                        <small>Avg Polarity</small>
                    </div>
                    <div class="col-md-3">
                        <h4 class="text-warning">{{ "%.3f"|format(avg_subjectivity) }}</h4>
                        <small>Avg Subjectivity</small>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}'''
    
    # Error template
    error_template = '''{% extends "base.html" %}

{% block title %}Error {{ error_code }} - Sentiment Analysis App{% endblock %}

{% block content %}
<div class="row">
    <div class="col-md-6 mx-auto">
        <div class="text-center">
            <h1 class="display-1 text-muted">{{ error_code }}</h1>
            <h2>Oops! Something went wrong.</h2>
            <p class="lead">{{ error_message }}</p>
            <a href="{{ url_for('index') }}" class="btn btn-primary">
                <i class="fas fa-home"></i> Go Home
            </a>
        </div>
    </div>
</div>
{% endblock %}'''
    
    # Stats template
    stats_template = '''{% extends "base.html" %}

{% block title %}Statistics - Sentiment Analysis App{% endblock %}

{% block content %}
<div class="row">
    <div class="col-12">
        <h1><i class="fas fa-chart-bar"></i> Analysis Statistics</h1>
        <p class="lead">Overview of your sentiment analysis history and trends.</p>
    </div>
</div>

{% if total_count > 0 %}
<div class="row mb-4">
    <div class="col-md-3">
        <div class="card stats-card text-center">
            <div class="card-body">
                <h2>{{ total_count }}</h2>
                <p class="mb-0">Total Analyses</p>
            </div>
        </div>
    </div>
    {% for stat in sentiment_stats %}
    <div class="col-md-3">
        <div class="card text-center">
            <div class="card-body">
                <h2 class="sentiment-{{ stat.sentiment.lower() }}">{{ stat.sentiment_count }}</h2>
                <p class="mb-0">{{ stat.sentiment }}</p>
                <small class="text-muted">Avg: {{ "%.3f"|format(stat.avg_polarity) }}</small>
            </div>
        </div>
    </div>
    {% endfor %}
</div>

{% if sentiment_stats %}
<div class="row mb-4">
    <div class="col-12">
        <div class="card">
            <div class="card-header">
                <h5>Sentiment Breakdown</h5>
            </div>
            <div class="card-body">
                <div class="table-responsive">
                    <table class="table">
                        <thead>
                            <tr>
                                <th>Sentiment</th>
                                <th>Count</th>
                                <th>Percentage</th>
                                <th>Avg Polarity</th>
                                <th>Avg Subjectivity</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for stat in sentiment_stats %}
                            <tr>
                                <td>
                                    <span class="badge sentiment-{{ stat.sentiment.lower() }}">
                                        {{ stat.sentiment }}
                                    </span>
                                </td>
                                <td>{{ stat.sentiment_count }}</td>
                                <td>{{ "%.1f"|format((stat.sentiment_count / total_count) * 100) }}%</td>
                                <td>{{ "%.4f"|format(stat.avg_polarity) }}</td>
                                <td>{{ "%.4f"|format(stat.avg_subjectivity) }}</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
</div>
{% endif %}

{% if daily_stats %}
<div class="row">
    <div class="col-12">
        <div class="card">
            <div class="card-header">
                <h5>Recent Activity (Last 7 Days)</h5>
            </div>
            <div class="card-body">
                <div class="table-responsive">
                    <table class="table">
                        <thead>
                            <tr>
                                <th>Date</th>
                                <th>Analyses</th>
                                <th>Avg Polarity</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for stat in daily_stats %}
                            <tr>
                                <td>{{ stat.date }}</td>
                                <td>{{ stat.daily_count }}</td>
                                <td>{{ "%.4f"|format(stat.daily_avg_polarity) }}</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
</div>
{% endif %}

{% else %}
<div class="row">
    <div class="col-12">
        <div class="text-center py-5">
            <i class="fas fa-chart-bar fa-5x text-muted mb-3"></i>
            <h3 class="text-muted">No Statistics Available</h3>
            <p class="text-muted">Start analyzing texts to see statistics here.</p>
            <a href="{{ url_for('index') }}" class="btn btn-primary">
                <i class="fas fa-plus"></i> Start Analyzing
            </a>
        </div>
    </div>
</div>
{% endif %}
{% endblock %}'''
    
    # Create batch results template
    with open('templates/batch_results.html', 'w', encoding='utf-8') as f:
        f.write(batch_results_template)
    
    # Create error template
    with open('templates/error.html', 'w', encoding='utf-8') as f:
        f.write(error_template)
    
    # Create stats template
    with open('templates/stats.html', 'w', encoding='utf-8') as f:
        f.write(stats_template)

# ============================================================================
# APPLICATION INITIALIZATION
# ============================================================================

if __name__ == '__main__':
    # Initialize database
    init_database()
    
    # Create missing templates
    create_missing_templates()
    
    print("🚀 Starting Sentiment Analysis Web Application...")
    print("📊 Features available:")
    print("   • Real-time sentiment analysis")
    print("   • Batch file processing")
    print("   • Analysis history and statistics")
    print("   • Interactive visualizations")
    print("   • RESTful API endpoints")
    print("\n🌐 Access the application at: http://localhost:5000")
    print("📚 API Documentation:")
    print("   • POST /api/analyze - Analyze text sentiment")
    print("   • GET /api/history - Get analysis history")
    
    # Run the Flask application
    app.run(debug=True, host='0.0.0.0', port=5000)