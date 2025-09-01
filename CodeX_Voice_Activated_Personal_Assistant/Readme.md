
# 🎤 Voice-Activated Personal Assistant (ARIA)

A comprehensive, intelligent personal assistant built in Python that combines speech recognition, text-to-speech, and various APIs to create an interactive voice-controlled experience. Meet **ARIA** - your Advanced Responsive Intelligence Assistant!

![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-brightgreen)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)

## 📋 Table of Contents

- [Features](#-features)
- [Demo](#-demo)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Commands](#-available-commands)
- [API Integration](#-api-integration)
- [Architecture](#-architecture)
- [Customization](#-customization)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

### 🎯 Core Capabilities
- **🗣️ Natural Voice Recognition** - Advanced speech-to-text using Google Speech Recognition
- **🔊 Text-to-Speech Engine** - Configurable voice synthesis with personality options
- **⏰ Smart Reminder System** - Natural language time parsing and background scheduling
- **🌤️ Weather Integration** - Real-time weather data with location support
- **📰 News Reader** - Latest headlines from multiple sources
- **🌐 Web Navigation** - Voice-controlled website opening
- **😄 Entertainment** - Joke telling and interactive conversations
- **📜 Conversation History** - Complete interaction logging and playback

### 🧠 Intelligence Features
- **Wake Word Activation** - "Hey ARIA" or "ARIA" to start interactions
- **Context Awareness** - Understands follow-up questions and commands
- **Error Recovery** - Graceful handling of recognition errors and network issues
- **Multi-threading** - Non-blocking operations for smooth user experience
- **Ambient Noise Adaptation** - Automatic microphone calibration

### 🔧 Technical Features
- **Modular Architecture** - Easy to extend and customize
- **Cross-Platform Support** - Works on Windows, macOS, and Linux
- **Demo Mode** - Test all features without microphone
- **Comprehensive Logging** - Detailed conversation history and debugging
- **API Ready** - Built-in support for weather and news APIs

## 🎬 Demo

### Voice Interaction Example
```
🎤 Listening...
👤 User: "Hey ARIA"
🔊 ARIA: "Yes? How can I help?"

👤 User: "What's the weather like?"
🔊 ARIA: "The weather in Hyderabad is sunny with a temperature of 28 degrees Celsius and humidity of 65%"

👤 User: "Remind me to call mom in 30 minutes"
🔊 ARIA: "Reminder set: 'call mom' for 3:45 PM on Monday, August 23"

👤 User: "Tell me a joke"
🔊 ARIA: "Why don't scientists trust atoms? Because they make up everything!"
```

### Screenshot
```
🎤 VOICE-ACTIVATED PERSONAL ASSISTANT
==================================================
Available commands:
• 'Hey ARIA' or 'ARIA' - Wake word
• 'What time is it?' - Get current time
• 'What's the weather?' - Get weather info
• 'Read the news' - Get latest news
• 'Set reminder' - Create a reminder
• 'Tell me a joke' - Random joke
• 'Open website' - Open a website
• 'Stop listening' - Exit assistant

🎯 Calibrating microphone for ambient noise...
✅ Microphone calibrated!
🔊 ARIA: Hello! I'm ARIA, your voice assistant. Say 'Hey ARIA' to wake me up!
```

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- Microphone (for voice input)
- Internet connection (for weather and news features)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/voice-assistant-aria.git
cd voice-assistant-aria
```

### Step 2: Install Python Dependencies
```bash
# Core dependencies
pip install speechrecognition pyttsx3 requests

# Audio support (choose based on your system)
pip install pyaudio  # Most systems
```

### Step 3: System-Specific Audio Setup

#### Windows
```bash
pip install pywin32
```

#### macOS (with Homebrew)
```bash
brew install portaudio
pip install pyaudio
```

#### Ubuntu/Debian Linux
```bash
sudo apt-get update
sudo apt-get install python3-pyaudio python3-dev libasound2-dev
```

#### CentOS/RHEL Linux
```bash
sudo yum install python3-pyaudio python3-devel alsa-lib-devel
```

### Step 4: Verify Installation
```bash
python voice_assistant.py
# Choose option 2 for demo mode to test without microphone
```

## ⚙️ Configuration

### API Keys Setup (Optional but Recommended)

#### Weather API (OpenWeatherMap)
1. Visit [OpenWeatherMap](https://openweathermap.org/api) and create a free account
2. Get your API key
3. Replace `your_openweather_api_key` in the code:
```python
self.weather_api_key = "your_actual_api_key_here"
```

#### News API
1. Visit [NewsAPI](https://newsapi.org/) and sign up
2. Get your API key
3. Replace `your_newsapi_key` in the code:
```python
self.news_api_key = "your_actual_api_key_here"
```

### Assistant Customization
```python
# Change assistant name
assistant = VoiceAssistant("YourName")

# Customize default location
self.default_city = "YourCity"

# Modify wake words
wake_words = ['hey yourname', 'yourname', 'assistant']
```

### Voice Settings
```python
# Adjust speech rate (words per minute)
self.tts_engine.setProperty('rate', 180)  # Default: 180

# Adjust volume (0.0 to 1.0)
self.tts_engine.setProperty('volume', 0.9)  # Default: 0.9
```

## 🎮 Usage

### Method 1: Voice Mode (Full Experience)
```bash
python voice_assistant.py
# Choose option 1
# Say "Hey ARIA" followed by your command
```

### Method 2: Demo Mode (Testing)
```bash
python voice_assistant.py
# Choose option 2 for automatic demonstration
```

### Method 3: Import as Module
```python
from voice_assistant import VoiceAssistant

# Create assistant instance
assistant = VoiceAssistant("ARIA")

# Start listening
assistant.start_listening()

# Or process single command
response = assistant.process_command("what time is it")
```

## 🎯 Available Commands

### ⏰ Time & Date
- `"What time is it?"`
- `"What's the date?"`
- `"What day is it?"`

### 🌤️ Weather
- `"What's the weather?"`
- `"What's the weather in [city]?"`
- `"How's the temperature?"`
- `"Weather forecast"`

### 📰 News
- `"Read the news"`
- `"What's in the news?"`
- `"Latest headlines"`
- `"News update"`

### ⏰ Reminders
- `"Set reminder to [task] in [time]"`
  - Example: `"Set reminder to call John in 30 minutes"`
  - Example: `"Remind me to take medicine at 8 PM"`
- `"Check my reminders"`
- `"What are my reminders?"`

#### Supported Time Formats
- `"in X minutes"` (e.g., "in 15 minutes")
- `"in X hours"` (e.g., "in 2 hours")
- `"at X:XX"` (e.g., "at 3:30")
- `"at X:XX PM/AM"` (e.g., "at 9:15 PM")

### 🌐 Web Navigation
- `"Open [website]"`
  - Supported: Google, YouTube, GitHub, Wikipedia, Gmail, Facebook, Twitter, LinkedIn
- `"Open [custom-url.com]"`

### 😄 Entertainment
- `"Tell me a joke"`
- `"Make me laugh"`
- `"Something funny"`

### ℹ️ System
- `"Help"`
- `"What can you do?"`
- `"Stop listening"` / `"Goodbye"` / `"Exit"`

### 💬 Conversation
- `"Hello"` / `"Hi"` / `"Hey"`
- `"Thank you"`
- `"How are you?"`

## 🔗 API Integration

### Weather Service
- **Provider**: OpenWeatherMap
- **Data**: Current temperature, weather conditions, humidity
- **Coverage**: Global cities
- **Fallback**: Mock data when API unavailable

### News Service
- **Provider**: NewsAPI
- **Content**: Top headlines, multiple categories
- **Sources**: Reputable news organizations
- **Fallback**: Sample headlines when API unavailable

### Speech Services
- **Recognition**: Google Speech Recognition
- **Synthesis**: System TTS engine (pyttsx3)
- **Languages**: Configurable (default: English)

## 🏗️ Architecture

### Core Components

```
voice_assistant.py
├── VoiceAssistant (Main Class)
│   ├── Speech Recognition (SpeechRecognition)
│   ├── Text-to-Speech (pyttsx3)
│   ├── Command Processing
│   ├── API Integrations
│   └── Reminder System
├── Reminder (Data Class)
└── Utility Functions
```

### Data Flow
```
Voice Input → Speech Recognition → Command Processing → API Calls → Response Generation → Text-to-Speech → Audio Output
```

### Threading Model
- **Main Thread**: Voice recognition loop
- **Timer Threads**: Reminder scheduling
- **TTS Thread**: Speech synthesis
- **Network Threads**: API calls

## 🎨 Customization

### Adding New Commands

```python
def process_command(self, command: str) -> str:
    # Add your custom command here
    if 'your custom trigger' in command:
        response = self.your_custom_function()
        self.speak(response)
        return response
```

### Creating Custom Functions

```python
def your_custom_function(self) -> str:
    """Your custom functionality."""
    # Implement your feature
    result = "Your response"
    return result
```

### Adding New APIs

```python
def integrate_new_api(self, query: str) -> str:
    """Integrate with a new API service."""
    try:
        # API call logic
        url = f"https://api.example.com/endpoint?q={query}&key={self.api_key}"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            # Process data
            result = self.format_api_response(data)
            return result
    except Exception as e:
        return f"Error accessing service: {str(e)}"
```

### Voice Personality Customization

```python
# Modify response variations
greetings = [
    "Hello! How can I assist you today?",
    "Hi there! What do you need help with?",
    "Hey! I'm ready to help!"
]

# Customize error messages
error_responses = [
    "I didn't catch that. Could you repeat?",
    "Sorry, I'm not sure what you mean.",
    "Can you rephrase that for me?"
]
```

## 🔧 Troubleshooting

### Common Issues

#### Microphone Not Working
```bash
# Test microphone
python -c "import speech_recognition as sr; print('Mic test:', sr.Microphone.list_microphone_names())"

# Linux: Check audio permissions
sudo usermod -a -G audio $USER
```

#### PyAudio Installation Issues
```bash
# Windows with pip issues
pip install pipwin
pipwin install pyaudio

# macOS with Homebrew
brew install portaudio
pip install --global-option='build_ext' --global-option='-I/usr/local/include' --global-option='-L/usr/local/lib' pyaudio
```

#### Speech Recognition Errors
- Check internet connection
- Verify microphone permissions
- Test in demo mode first
- Adjust microphone sensitivity

#### TTS Not Working
```python
# List available voices
import pyttsx3
engine = pyttsx3.init()
voices = engine.getProperty('voices')
for voice in voices:
    print(f"ID: {voice.id}, Name: {voice.name}")
```

### Performance Optimization

#### Reduce Memory Usage
```python
# Limit conversation history
max_history = 50
if len(self.conversation_history) > max_history:
    self.conversation_history = self.conversation_history[-max_history:]
```

#### Improve Response Time
```python
# Reduce timeout values for faster interaction
command = self.listen(timeout=3, phrase_timeout=2)
```

### Debug Mode
```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Development Setup
```bash
# Fork the repository
git clone https://github.com/yourusername/voice-assistant-aria.git
cd voice-assistant-aria

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt
```

### Contribution Guidelines

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Code Style
- Follow PEP 8 guidelines
- Add docstrings to all functions
- Include type hints where appropriate
- Write unit tests for new features

### Feature Requests
- Voice command improvements
- New API integrations
- UI/UX enhancements
- Performance optimizations
- Language support

## 🙏 Acknowledgments

- **Speech Recognition**: Google Speech Recognition API
- **Text-to-Speech**: pyttsx3 library
- **Weather Data**: OpenWeatherMap API
- **News Data**: NewsAPI
- **Audio Processing**: PyAudio library
- **HTTP Requests**: Requests library

## 📊 Project Stats

- **Languages**: Python
- **Dependencies**: 6 core libraries
- **Features**: 15+ voice commands
- **Platforms**: Windows, macOS, Linux
- **Code Quality**: Type hints, documentation, error handling
- **Testing**: Demo mode, unit tests ready

## 🔮 Future Roadmap

### Upcoming Features
- [ ] **Multi-language support** - Spanish, French, German
- [ ] **Calendar integration** - Google Calendar, Outlook
- [ ] **Smart home control** - IoT device integration
- [ ] **Music control** - Spotify, Apple Music integration
- [ ] **Email management** - Read and send emails
- [ ] **Advanced NLP** - Better command understanding
- [ ] **Mobile app** - Android/iOS companion
- [ ] **Cloud sync** - Cross-device conversations

### Performance Improvements
- [ ] **Offline mode** - Local speech recognition
- [ ] **Faster wake word** - Custom wake word detection
- [ ] **Memory optimization** - Reduced resource usage
- [ ] **Voice training** - Personalized recognition

### Developer Features
- [ ] **Plugin system** - Easy third-party integrations
- [ ] **REST API** - Web service interface
- [ ] **Configuration GUI** - Visual settings management
- [ ] **Voice analytics** - Usage statistics and insights

## 📞 Contact

- **Sudeeksha** - srsudeeksha@gmail.com
- **Project Link**: https://github.com/srsudeeksha/Python_Internship_CodeX/tree/main/CodeX_Voice_Activated_Personal_Assistant

---


⭐ **Star this repository if you found it helpful!** ⭐

