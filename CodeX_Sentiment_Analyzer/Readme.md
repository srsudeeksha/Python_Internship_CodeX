# 🧠 Sentiment Analysis Web Application

A comprehensive web application for sentiment analysis using TextBlob and Flask. Analyze the emotional tone of text with advanced natural language processing, batch file processing, and detailed analytics.

## ✨ Features

### 🎯 Core Functionality
- **Real-time Sentiment Analysis** - Analyze individual texts instantly
- **Batch Processing** - Upload CSV/TXT files for bulk analysis
- **Interactive Visualizations** - Charts and graphs for data insights
- **Analysis History** - Track and review past analyses
- **Detailed Statistics** - Comprehensive analytics dashboard
- **RESTful API** - Programmatic access to sentiment analysis

### 📊 Analysis Metrics
- **Sentiment Classification** - Positive, Negative, or Neutral
- **Polarity Score** - Numerical sentiment strength (-1 to +1)
- **Subjectivity Score** - Objective vs subjective content (0 to 1)
- **Confidence Levels** - High, Medium, or Low confidence ratings
- **Language Detection** - Automatic language identification
- **Key Phrase Extraction** - Important words and phrases
- **Text Statistics** - Word count, sentence count, character count

## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Installation

1. **Clone or Download** the project files
2. **Install required packages:**
   ```bash
   pip install flask textblob pandas matplotlib sqlite3 werkzeug
   ```

3. **Download NLTK data** (required for TextBlob):
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('vader_lexicon'); nltk.download('brown'); nltk.download('averaged_perceptron_tagger')"
   ```

4. **Run the application:**
   ```bash
   python Sentiment_Analyzer.py
   ```

5. **Access the application:**
   Open your browser and go to `http://localhost:5000`

## 📁 Project Structure

```
sentiment-analyzer/
├── Sentiment_Analyzer.py          # Main Flask application
├── templates/                     # HTML templates
│   ├── base.html                 # Base template with navigation
│   ├── index.html                # Home page
│   ├── results.html              # Single analysis results
│   ├── batch.html                # Batch analysis upload
│   ├── batch_results.html        # Batch analysis results
│   ├── history.html              # Analysis history
│   ├── stats.html                # Statistics dashboard
│   └── error.html                # Error pages
├── uploads/                       # Temporary file uploads (auto-created)
├── sentiment_history.db           # SQLite database (auto-created)
└── README.md                     # This file
```

## 🎨 User Interface

### 🏠 Home Page (`/`)
- Clean, intuitive text input interface
- Character counter (10,000 character limit)
- Example texts for quick testing
- Sentiment classification guide

### 📊 Results Page (`/analyze`)
- Detailed sentiment analysis results
- Interactive visualizations and charts
- Text statistics and key phrases
- Analysis interpretation guide

### 📤 Batch Analysis (`/batch`)
- Drag-and-drop file upload
- Support for TXT and CSV files
- File validation and preview
- Progress indicator during processing

### 📚 History (`/history`)
- Searchable analysis history
- Advanced filtering options
- Detailed view modals
- Trend visualizations

### 📈 Statistics (`/stats`)
- Overall sentiment distribution
- Time-based analysis trends
- Confidence level breakdowns
- Daily activity summaries

## 🔧 API Endpoints

### Single Text Analysis
```http
POST /api/analyze
Content-Type: application/json

{
  "text": "I love this amazing product!"
}
```

**Response:**
```json
{
  "text": "I love this amazing product!",
  "polarity": 0.625,
  "subjectivity": 0.6,
  "sentiment": "Positive",
  "confidence": "High",
  "word_count": 5,
  "sentence_count": 1,
  "detected_language": "en",
  "key_phrases": ["love", "amazing", "product"],
  "timestamp": "2024-08-23T15:30:45.123456"
}
```

### Analysis History
```http
GET /api/history?limit=50
```

**Response:**
```json
[
  {
    "id": 1,
    "text": "Sample text...",
    "sentiment": "Positive",
    "polarity": 0.5,
    "confidence": "High",
    "timestamp": "2024-08-23T15:30:45"
  }
]
```

## 📄 File Formats

### TXT Files
- Plain text format
- One text per line or paragraph
- Minimum 10 characters per text
- Maximum 50 texts processed
- UTF-8 encoding recommended

**Example:**
```
I love this product!
The service was terrible.
This is an okay experience.
```

### CSV Files
- Comma-separated values
- Text columns auto-detected
- Headers supported
- Maximum 100 rows processed
- UTF-8 or Latin-1 encoding

**Example:**
```csv
review,rating
"Great product, highly recommend!",5
"Not what I expected, disappointed.",2
"Average quality, nothing special.",3
```

## 🎯 Sentiment Classification

### Polarity Scores
- **Positive**: Polarity > 0.1
- **Neutral**: -0.1 ≤ Polarity ≤ 0.1  
- **Negative**: Polarity < -0.1

### Confidence Levels
- **High**: |Polarity| ≥ 0.5
- **Medium**: 0.2 ≤ |Polarity| < 0.5
- **Low**: |Polarity| < 0.2

### Subjectivity Scale
- **0.0**: Completely objective (facts only)
- **0.5**: Mixed objective and subjective
- **1.0**: Completely subjective (opinions only)

## 🛠️ Configuration

### Application Settings
```python
# Maximum file upload size
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

# Supported file types
ALLOWED_EXTENSIONS = {'txt', 'csv'}

# Database location
DATABASE_PATH = 'sentiment_history.db'

# Upload directory
UPLOAD_FOLDER = 'uploads'
```

### Processing Limits
- **Single Text**: 10,000 characters maximum
- **TXT Files**: 50 texts maximum
- **CSV Files**: 100 rows maximum
- **File Size**: 16MB maximum
- **History Display**: 100 recent analyses

## 🔍 Troubleshooting

### Common Issues

**1. TextBlob/NLTK Data Missing**
```bash
# Download required NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('vader_lexicon')"
```

**2. Template Not Found Error**
- Ensure all HTML templates are in the `templates/` folder
- Check file names match exactly (case-sensitive)

**3. Database Issues**
- Delete `sentiment_history.db` file to reset database
- Check write permissions in application directory

**4. File Upload Problems**
- Verify file format (TXT/CSV only)
- Check file size (maximum 16MB)
- Ensure proper text encoding (UTF-8 recommended)

**5. Port Already in Use**
```python
# Change port in the main application
app.run(debug=True, host='0.0.0.0', port=5001)  # Use port 5001
```

### Performance Tips

- **Large Files**: Break into smaller chunks for better performance
- **Memory Usage**: Restart application periodically for long-running sessions
- **Database**: Clear history regularly to maintain performance

## 🔒 Security Considerations

### Input Validation
- Text length limits enforced
- File type restrictions
- SQL injection prevention with parameterized queries
- XSS protection with template escaping

### File Handling
- Secure filename generation
- Temporary file cleanup
- Upload size restrictions
- File type validation

## 🚀 Advanced Usage

### Custom Sentiment Models
Replace the `SentimentAnalyzer` class with your own implementation:

```python
class CustomSentimentAnalyzer:
    @staticmethod
    def analyze_text(text):
        # Your custom analysis logic here
        return {
            'text': text,
            'polarity': 0.0,
            'subjectivity': 0.0,
            'sentiment': 'Neutral',
            'confidence': 'Low',
            # ... other required fields
        }
```

### Database Integration
The application uses SQLite by default. To use a different database:

```python
# Update the connection string
DATABASE_PATH = 'postgresql://user:password@localhost/sentiment_db'
```

### API Authentication
Add authentication to API endpoints:

```python
from functools import wraps

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

@app.route('/api/analyze', methods=['POST'])
@require_auth
def api_analyze():
    # Protected endpoint
```

## 📱 Mobile Compatibility

The application is fully responsive and works on:
- Desktop browsers (Chrome, Firefox, Safari, Edge)
- Tablet devices (iPad, Android tablets)
- Mobile phones (iOS Safari, Android Chrome)

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Code Style
- Follow PEP 8 for Python code
- Use Bootstrap classes for styling
- Comment complex logic
- Write descriptive commit messages


## 🙏 Acknowledgments

- **TextBlob** - Natural language processing library
- **Flask** - Web application framework
- **Bootstrap** - Frontend UI framework
- **Chart.js/Matplotlib** - Data visualization
- **Font Awesome** - Icon library

## 📞 Support

### Getting Help
- Check the troubleshooting section above
- Review the code comments for implementation details
- Test with the provided example texts

### Feature Requests
- Single text analysis improvements
- Additional file format support
- Enhanced visualization options
- Custom sentiment model integration

## 📞 Contact

- **Sudeeksha** - srsudeeksha@gmail.com
- **Project Link**: https://github.com/srsudeeksha/Python_Internship_CodeX/tree/main/CodeX_Sentiment_Analyzer

---

**Built with ❤️ using Flask and TextBlob**

