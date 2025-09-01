# Voice-Activated Personal Assistant
# A comprehensive personal assistant with speech recognition and text-to-speech capabilities

import speech_recognition as sr
import pyttsx3
import datetime
import requests
import json
import threading
import time
import re
import webbrowser
import os
import random
from dataclasses import dataclass
from typing import List, Dict, Optional
import schedule

# ============================================================================
# CONFIGURATION AND SETUP
# ============================================================================

@dataclass
class Reminder:
    id: str
    message: str
    time: datetime.datetime
    is_active: bool = True

class VoiceAssistant:
    def __init__(self, name: str = "ARIA"):
        self.name = name
        self.is_listening = False
        self.reminders: List[Reminder] = []
        self.conversation_history = []
        
        # Initialize speech recognition and text-to-speech
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.tts_engine = pyttsx3.init()
        
        # Configure TTS settings
        self.setup_tts()
        
        # API Keys (You'll need to replace these with your actual keys)
        self.weather_api_key = "your_openweather_api_key"  # Get from openweathermap.org
        self.news_api_key = "your_newsapi_key"  # Get from newsapi.org
        
        # Default location for weather
        self.default_city = "Hyderabad"
        
        # Calibrate microphone
        self.calibrate_microphone()
        
        print(f"🎤 {self.name} Voice Assistant Initialized!")
        print("Available commands:")
        print("• 'Hey ARIA' or 'ARIA' - Wake word")
        print("• 'What time is it?' - Get current time")
        print("• 'What's the weather?' - Get weather info")
        print("• 'Read the news' - Get latest news")
        print("• 'Set reminder' - Create a reminder")
        print("• 'Check reminders' - List active reminders")
        print("• 'Tell me a joke' - Random joke")
        print("• 'Open website' - Open a website")
        print("• 'Stop listening' - Exit assistant")
        
    def setup_tts(self):
        """Configure text-to-speech settings."""
        voices = self.tts_engine.getProperty('voices')
        
        # Try to set a female voice if available
        for voice in voices:
            if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                self.tts_engine.setProperty('voice', voice.id)
                break
        
        # Set speech rate and volume
        self.tts_engine.setProperty('rate', 180)  # Speed of speech
        self.tts_engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)
    
    def calibrate_microphone(self):
        """Calibrate microphone for ambient noise."""
        print("🎯 Calibrating microphone for ambient noise...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        print("✅ Microphone calibrated!")
    
    def speak(self, text: str):
        """Convert text to speech."""
        print(f"🔊 {self.name}: {text}")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()
    
    def listen(self, timeout: int = 5, phrase_timeout: int = 2) -> Optional[str]:
        """Listen for audio input and convert to text."""
        try:
            with self.microphone as source:
                print("🎤 Listening...")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_timeout)
                
            print("🔍 Recognizing...")
            text = self.recognizer.recognize_google(audio).lower()
            print(f"👤 User: {text}")
            return text
            
        except sr.WaitTimeoutError:
            print("⏱️ Listening timeout")
            return None
        except sr.UnknownValueError:
            print("❓ Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"❌ Error with speech recognition service: {e}")
            return None
    
    def get_current_time(self):
        """Get current time and date."""
        now = datetime.datetime.now()
        time_str = now.strftime("%I:%M %p")
        date_str = now.strftime("%A, %B %d, %Y")
        
        response = f"The current time is {time_str} on {date_str}"
        self.speak(response)
        return response
    
    def get_weather(self, city: str = None) -> str:
        """Get weather information for a city."""
        if not city:
            city = self.default_city
        
        # Note: This is a demo implementation
        # In a real application, you would use your actual API key
        if self.weather_api_key == "your_openweather_api_key":
            # Mock weather data for demonstration
            weather_conditions = ["sunny", "cloudy", "rainy", "partly cloudy"]
            temperature = random.randint(20, 35)
            condition = random.choice(weather_conditions)
            
            response = f"The weather in {city} is {condition} with a temperature of {temperature} degrees Celsius."
            self.speak(response)
            return response
        
        try:
            # Real API call (uncomment when you have a valid API key)
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={self.weather_api_key}&units=metric"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                temp = data['main']['temp']
                description = data['weather'][0]['description']
                humidity = data['main']['humidity']
                
                weather_info = f"The weather in {city} is {description} with a temperature of {temp:.1f} degrees Celsius and humidity of {humidity}%."
                self.speak(weather_info)
                return weather_info
            else:
                error_msg = f"Sorry, I couldn't get weather information for {city}."
                self.speak(error_msg)
                return error_msg
                
        except Exception as e:
            error_msg = f"There was an error getting weather information: {str(e)}"
            self.speak(error_msg)
            return error_msg
    
    def get_news(self, category: str = "general") -> str:
        """Get latest news headlines."""
        # Note: This is a demo implementation
        # In a real application, you would use your actual API key
        if self.news_api_key == "your_newsapi_key":
            # Mock news data for demonstration
            mock_headlines = [
                "Technology stocks rise as AI adoption accelerates",
                "New renewable energy project announced in India",
                "Space mission successfully launches communication satellite",
                "Healthcare breakthrough in cancer treatment research",
                "Local weather patterns show seasonal changes"
            ]
            
            headlines = random.sample(mock_headlines, 3)
            news_text = "Here are today's top headlines: "
            for i, headline in enumerate(headlines, 1):
                news_text += f"{i}. {headline}. "
            
            self.speak(news_text)
            return news_text
        
        try:
            # Real API call (uncomment when you have a valid API key)
            url = f"https://newsapi.org/v2/top-headlines?country=in&category={category}&apiKey={self.news_api_key}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])[:5]  # Get top 5 articles
                
                if articles:
                    news_text = "Here are today's top headlines: "
                    for i, article in enumerate(articles, 1):
                        title = article.get('title', 'No title')
                        news_text += f"{i}. {title}. "
                    
                    self.speak(news_text)
                    return news_text
                else:
                    no_news_msg = "Sorry, I couldn't find any news articles right now."
                    self.speak(no_news_msg)
                    return no_news_msg
            else:
                error_msg = "Sorry, I couldn't fetch the news right now."
                self.speak(error_msg)
                return error_msg
                
        except Exception as e:
            error_msg = f"There was an error getting news: {str(e)}"
            self.speak(error_msg)
            return error_msg
    
    def set_reminder(self, message: str, when: str) -> str:
        """Set a reminder for a specific time."""
        try:
            # Parse different time formats
            now = datetime.datetime.now()
            reminder_time = None
            
            # Handle "in X minutes/hours"
            if "in" in when.lower():
                time_match = re.search(r'in (\d+) (minute|minutes|hour|hours)', when.lower())
                if time_match:
                    value = int(time_match.group(1))
                    unit = time_match.group(2)
                    
                    if 'minute' in unit:
                        reminder_time = now + datetime.timedelta(minutes=value)
                    elif 'hour' in unit:
                        reminder_time = now + datetime.timedelta(hours=value)
            
            # Handle "at X:XX" format
            elif "at" in when.lower():
                time_match = re.search(r'at (\d{1,2}):(\d{2})', when.lower())
                if time_match:
                    hour = int(time_match.group(1))
                    minute = int(time_match.group(2))
                    
                    # Assume PM if hour is less than 8 (business hours assumption)
                    if hour < 8 and "pm" not in when.lower():
                        hour += 12
                    
                    reminder_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    
                    # If the time has passed today, set for tomorrow
                    if reminder_time <= now:
                        reminder_time += datetime.timedelta(days=1)
            
            if reminder_time:
                reminder_id = str(len(self.reminders) + 1)
                reminder = Reminder(
                    id=reminder_id,
                    message=message,
                    time=reminder_time
                )
                
                self.reminders.append(reminder)
                
                # Schedule the reminder
                self.schedule_reminder(reminder)
                
                time_str = reminder_time.strftime("%I:%M %p on %A, %B %d")
                response = f"Reminder set: '{message}' for {time_str}"
                self.speak(response)
                return response
            else:
                error_msg = "Sorry, I couldn't understand the time format. Try saying 'in 30 minutes' or 'at 3:30 PM'."
                self.speak(error_msg)
                return error_msg
                
        except Exception as e:
            error_msg = f"Error setting reminder: {str(e)}"
            self.speak(error_msg)
            return error_msg
    
    def schedule_reminder(self, reminder: Reminder):
        """Schedule a reminder to trigger at the specified time."""
        def reminder_alert():
            if reminder.is_active:
                alert_msg = f"Reminder: {reminder.message}"
                print(f"🔔 {alert_msg}")
                self.speak(alert_msg)
                reminder.is_active = False
        
        # Calculate delay until reminder time
        delay = (reminder.time - datetime.datetime.now()).total_seconds()
        
        if delay > 0:
            timer = threading.Timer(delay, reminder_alert)
            timer.daemon = True
            timer.start()
    
    def check_reminders(self) -> str:
        """List all active reminders."""
        active_reminders = [r for r in self.reminders if r.is_active]
        
        if not active_reminders:
            response = "You have no active reminders."
            self.speak(response)
            return response
        
        response = f"You have {len(active_reminders)} active reminder"
        if len(active_reminders) > 1:
            response += "s"
        response += ": "
        
        for i, reminder in enumerate(active_reminders, 1):
            time_str = reminder.time.strftime("%I:%M %p on %A")
            response += f"{i}. {reminder.message} at {time_str}. "
        
        self.speak(response)
        return response
    
    def tell_joke(self) -> str:
        """Tell a random joke."""
        jokes = [
            "Why don't scientists trust atoms? Because they make up everything!",
            "I told my wife she was drawing her eyebrows too high. She seemed surprised.",
            "Why don't eggs tell jokes? They'd crack each other up!",
            "I'm reading a book about anti-gravity. It's impossible to put down!",
            "Why did the scarecrow win an award? He was outstanding in his field!",
            "What do you call a fake noodle? An impasta!",
            "Why don't some couples go to the gym? Because some relationships don't work out!",
            "I used to hate facial hair, but then it grew on me.",
            "What do you call a bear with no teeth? A gummy bear!",
            "Why did the math book look so sad? Because it had too many problems!"
        ]
        
        joke = random.choice(jokes)
        self.speak(joke)
        return joke
    
    def open_website(self, site_name: str) -> str:
        """Open a website in the default browser."""
        websites = {
            'google': 'https://www.google.com',
            'youtube': 'https://www.youtube.com',
            'github': 'https://www.github.com',
            'wikipedia': 'https://www.wikipedia.org',
            'news': 'https://news.google.com',
            'weather': 'https://weather.com',
            'gmail': 'https://mail.google.com',
            'facebook': 'https://www.facebook.com',
            'twitter': 'https://www.twitter.com',
            'linkedin': 'https://www.linkedin.com'
        }
        
        site_name = site_name.lower()
        
        if site_name in websites:
            webbrowser.open(websites[site_name])
            response = f"Opening {site_name} in your browser."
            self.speak(response)
            return response
        else:
            # Try to open as URL
            if '.' in site_name:
                if not site_name.startswith('http'):
                    site_name = 'https://' + site_name
                webbrowser.open(site_name)
                response = f"Opening {site_name} in your browser."
                self.speak(response)
                return response
            else:
                available_sites = ', '.join(websites.keys())
                response = f"I don't recognize that website. Available sites include: {available_sites}"
                self.speak(response)
                return response
    
    def process_command(self, command: str) -> str:
        """Process voice command and return appropriate response."""
        command = command.lower().strip()
        
        # Store command in conversation history
        self.conversation_history.append({
            'timestamp': datetime.datetime.now(),
            'user_input': command,
            'response': None
        })
        
        response = ""
        
        try:
            # Time-related commands
            if any(phrase in command for phrase in ['time', 'what time']):
                response = self.get_current_time()
            
            # Weather commands
            elif any(phrase in command for phrase in ['weather', 'temperature']):
                # Extract city name if mentioned
                city_match = re.search(r'in ([a-zA-Z\s]+)', command)
                city = city_match.group(1).strip() if city_match else None
                response = self.get_weather(city)
            
            # News commands
            elif any(phrase in command for phrase in ['news', 'headlines']):
                response = self.get_news()
            
            # Reminder commands
            elif 'set reminder' in command or 'remind me' in command:
                # Parse reminder message and time
                reminder_pattern = r'(?:set reminder|remind me)(?: to)?\s+(.+?)\s+(?:in|at)\s+(.+)'
                match = re.search(reminder_pattern, command)
                
                if match:
                    message = match.group(1).strip()
                    when = match.group(2).strip()
                    response = self.set_reminder(message, when)
                else:
                    response = "Please specify what to remind you about and when. For example: 'Remind me to call mom in 30 minutes'"
                    self.speak(response)
            
            elif 'check reminder' in command or 'my reminder' in command:
                response = self.check_reminders()
            
            # Joke command
            elif any(phrase in command for phrase in ['joke', 'funny', 'laugh']):
                response = self.tell_joke()
            
            # Website commands
            elif 'open' in command and ('website' in command or any(site in command for site in ['google', 'youtube', 'github'])):
                # Extract website name
                site_pattern = r'open\s+(?:website\s+)?([a-zA-Z0-9\.\-_]+)'
                match = re.search(site_pattern, command)
                if match:
                    site_name = match.group(1)
                    response = self.open_website(site_name)
                else:
                    response = "Please specify which website to open."
                    self.speak(response)
            
            # Greeting responses
            elif any(phrase in command for phrase in ['hello', 'hi', 'hey']):
                greetings = [
                    f"Hello! I'm {self.name}, your personal assistant. How can I help you today?",
                    f"Hi there! {self.name} here. What can I do for you?",
                    f"Hey! I'm {self.name}. Ready to assist you!"
                ]
                response = random.choice(greetings)
                self.speak(response)
            
            # Help command
            elif any(phrase in command for phrase in ['help', 'what can you do']):
                help_text = "I can help you with: checking the time, getting weather updates, reading news, setting reminders, telling jokes, opening websites, and much more. Just ask me naturally!"
                self.speak(help_text)
                response = help_text
            
            # Exit commands
            elif any(phrase in command for phrase in ['stop listening', 'goodbye', 'exit', 'quit']):
                farewell_messages = [
                    "Goodbye! Have a great day!",
                    "See you later! Feel free to call me anytime.",
                    "Farewell! I'll be here when you need me."
                ]
                response = random.choice(farewell_messages)
                self.speak(response)
                self.is_listening = False
            
            # Default response for unrecognized commands
            else:
                default_responses = [
                    "I'm not sure how to help with that. Try asking about the time, weather, news, or setting reminders.",
                    "I didn't understand that command. You can ask me about weather, news, time, or say 'help' for more options.",
                    "Sorry, I don't recognize that command. Try asking me to check the weather or tell you the time."
                ]
                response = random.choice(default_responses)
                self.speak(response)
        
        except Exception as e:
            error_response = f"I encountered an error: {str(e)}"
            self.speak(error_response)
            response = error_response
        
        # Update conversation history with response
        if self.conversation_history:
            self.conversation_history[-1]['response'] = response
        
        return response
    
    def start_listening(self):
        """Start the main listening loop."""
        self.is_listening = True
        
        welcome_messages = [
            f"Hello! I'm {self.name}, your voice assistant. Say 'Hey {self.name}' to wake me up!",
            f"{self.name} is now active and ready to help!",
            f"Voice assistant {self.name} is listening. How can I assist you today?"
        ]
        
        self.speak(random.choice(welcome_messages))
        
        wake_words = ['hey aria', 'aria', 'hey assistant', 'assistant']
        
        while self.is_listening:
            try:
                # Listen for wake word
                command = self.listen(timeout=10, phrase_timeout=3)
                
                if command:
                    # Check for wake word
                    if any(wake_word in command for wake_word in wake_words):
                        # Remove wake word from command
                        for wake_word in wake_words:
                            command = command.replace(wake_word, '').strip()
                        
                        if command:
                            # Process the command immediately
                            self.process_command(command)
                        else:
                            # Just acknowledged, wait for actual command
                            acknowledgments = [
                                "Yes? How can I help?",
                                "I'm listening. What do you need?",
                                "How may I assist you?"
                            ]
                            self.speak(random.choice(acknowledgments))
                            
                            # Listen for the actual command
                            follow_up = self.listen(timeout=8, phrase_timeout=4)
                            if follow_up:
                                self.process_command(follow_up)
                    
                    # Direct commands without wake word (for convenience during conversation)
                    elif any(phrase in command for phrase in ['time', 'weather', 'news', 'reminder', 'joke', 'stop listening']):
                        self.process_command(command)
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.1)
                
            except KeyboardInterrupt:
                print("\n🛑 Stopping voice assistant...")
                self.speak("Voice assistant shutting down. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error in main loop: {e}")
                continue
        
        print(f"👋 {self.name} voice assistant stopped.")
    
    def show_conversation_history(self):
        """Display conversation history."""
        if not self.conversation_history:
            print("No conversation history available.")
            return
        
        print(f"\n📜 Conversation History with {self.name}:")
        print("=" * 60)
        
        for i, entry in enumerate(self.conversation_history, 1):
            timestamp = entry['timestamp'].strftime("%H:%M:%S")
            print(f"{i}. [{timestamp}]")
            print(f"   User: {entry['user_input']}")
            print(f"   {self.name}: {entry['response']}")
            print("-" * 60)

# ============================================================================
# DEMO AND TESTING FUNCTIONS
# ============================================================================

def demo_assistant():
    """Demonstrate assistant capabilities without voice input."""
    print("🎭 VOICE ASSISTANT DEMO MODE")
    print("=" * 50)
    
    assistant = VoiceAssistant("ARIA")
    
    # Demo commands
    demo_commands = [
        "what time is it",
        "what's the weather in Mumbai",
        "read the news",
        "set reminder to call mom in 30 minutes",
        "tell me a joke",
        "check my reminders",
        "open google"
    ]
    
    print("\n🎪 Running demo commands:")
    print("-" * 30)
    
    for i, command in enumerate(demo_commands, 1):
        print(f"\n{i}. Demo Command: '{command}'")
        print("-" * 20)
        response = assistant.process_command(command)
        time.sleep(2)  # Pause between demos
    
    print(f"\n✨ Demo completed! {len(demo_commands)} commands processed.")
    
    # Show conversation history
    assistant.show_conversation_history()
    
    return assistant

def main():
    """Main function to run the voice assistant."""
    print("🎤 VOICE-ACTIVATED PERSONAL ASSISTANT")
    print("=" * 50)
    
    print("\nChoose an option:")
    print("1. Start Voice Assistant (requires microphone)")
    print("2. Run Demo Mode (text-based demonstration)")
    print("3. Exit")
    
    try:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            print("\n🎙️ Starting voice-activated mode...")
            print("Make sure your microphone is connected and working!")
            print("Say 'Hey ARIA' to wake up the assistant.")
            print("Press Ctrl+C to stop.\n")
            
            assistant = VoiceAssistant("ARIA")
            assistant.start_listening()
            
        elif choice == "2":
            print("\n🎭 Starting demo mode...")
            demo_assistant()
            
        elif choice == "3":
            print("👋 Goodbye!")
            
        else:
            print("❌ Invalid choice. Please run the program again.")
            
    except KeyboardInterrupt:
        print("\n🛑 Program interrupted by user.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    # Installation instructions
    print("📋 REQUIRED LIBRARIES:")
    print("pip install speechrecognition pyttsx3 requests pyaudio")
    print("\nNote: You may need additional setup for microphone access.")
    print("For Windows: pip install pywin32")
    print("For macOS: brew install portaudio (if using Homebrew)")
    print("For Linux: sudo apt-get install python3-pyaudio")
    print("\n" + "="*60 + "\n")
    
    main()