# News Summarization and Text-to-Speech Application

![Project Banner](https://via.placeholder.com/1200x300?text=News+Summarization+and+TTS+Application)

## Overview
This project is a web-based application that extracts news articles for a given company, performs sentiment analysis, and summarizes the content in English. Additionally, each article’s concise summary is translated into Hindi and then combined to generate a Hindi audio report using text-to-speech (TTS). The final solution includes a user-friendly interface built with Streamlit, API integration, and deployment on Hugging Face Spaces.

## Features
- **News Extraction:** Scrapes article titles, summaries, and publication dates from at least 10 unique news articles using BeautifulSoup.
- **Sentiment Analysis:** Classifies articles as Positive, Negative, or Neutral using Hugging Face Transformers.
- **Summarization:** Generates concise summaries for each article.
- **Translation and TTS:** Translates each article's concise summary from English to Hindi and combines them to generate a Hindi audio report.
- **User Interface:** Provides a clean, responsive web interface built with Streamlit.
- **API Development:** Exposes backend functionality via APIs (see `api.py`).
- **Deployment:** Deployed on Hugging Face Spaces for public testing.

## Project Links
- **GitHub Repository:** [https://github.com/yourusername/news-summarization-tts](https://github.com/yourusername/news-summarization-tts)
- **Hugging Face Space:** [https://your-username-news-summarization-tts.hf.space](https://your-username-news-summarization-tts.hf.space)
- **Video Demo:** [Watch the Demo]([https://youtu.be/yourvideolink](https://drive.google.com/file/d/1DB-nL0k_D5HgH-kf87Pze1r9ZbiNtFU1/view?usp=sharing))

## Installation

### Prerequisites
- Python 3.7+
- pip

### Local Setup Steps
1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/news-summarization-tts.git
   cd news-summarization-tts
