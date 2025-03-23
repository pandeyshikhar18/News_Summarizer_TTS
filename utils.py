import requests
from bs4 import BeautifulSoup
import random, time, os, re
from transformers import pipeline
from gtts import gTTS
from langdetect import detect

# Suppress HF symlink warnings on Windows.
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# ---------------- Model Initializations ---------------- #

# Load summarization pipeline (using CPU; set device=-1 if GPU is not available)
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
# Load sentiment analysis pipeline.
sentiment_analyzer = pipeline("sentiment-analysis")
# Load English-to-Hindi translation pipeline.
translator = pipeline("translation_en_to_hi", model="Helsinki-NLP/opus-mt-en-hi", device=-1)

# ---------------- News Extraction Functions ---------------- #

def get_article_urls(company_name, max_articles=10):
    sample_urls = [
        "https://www.annualreports.com/Company/tesla-inc",
        "https://apnews.com/article/tesla-vandalism-musk-trump-domestic-extremism-7576c03393a733eaf34b793e86ad1a6f",
        "https://www.bbc.com/news/articles/cz61vwjel2zo",
        "https://finance.yahoo.com/quote/TSLA/?guccounter=1&guce_referrer=aHR0cHM6Ly93d3cuZ29vZ2xlLmNvbS8&guce_referrer_sig=AQAAAESBf1sCTic3ABYh4w5uWUE0cdwoJBBjokBtwYW5iRavEgQbXkbECq587sV9M76MBHaXO-SGRNXS90tAX9vaSsrGO0ptVRyuaP2uDkES_je3oOpSPvQYRXrEwARnDxg1GPJPX95MhzmQojqwRsskwj7YZkaEN7wRztWxf2QCPl4Q",
        "https://www.nasdaq.com/market-activity/stocks/tsla/financials",
        "https://www.morningstar.com/stocks/xwbo/tsla/financials",
        "https://www.statista.com/statistics/272120/revenue-of-tesla/",
        "https://medium.com/@nambos3rd/tesla-full-year-2024-analysis-a-review-of-actual-performance-my-financial-forecast-41ee70091b5a",
        "https://www.cnbc.com/2025/01/29/tesla-tsla-2024-q4-earnings.html",
        "https://www.businesswire.com/news/home/20250129018824/en/Tesla-Releases-Fourth-Quarter-and-Full-Year-2024-Financial-Results"
    ]
    return sample_urls[:max_articles]

def scrape_article(url):
    headers = {"User-Agent": random.choice([
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko)"
    ])}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"Failed to retrieve {url}. Status code: {response.status_code}")
            return None
        soup = BeautifulSoup(response.text, "html.parser")
        title = soup.find('h1').get_text(strip=True) if soup.find('h1') else "No Title Found"
        meta_desc = soup.find("meta", attrs={"name": "description"})
        summary = meta_desc["content"] if meta_desc and meta_desc.get("content") else (
            soup.find("p").get_text(strip=True) if soup.find("p") else "No Summary Found"
        )
        time_tag = soup.find("time")
        pub_date = time_tag.get("datetime") if time_tag and time_tag.get("datetime") else "Unknown Publication Date"
        return {"URL": url, "Title": title, "Summary": summary, "Publication Date": pub_date}
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

def scrape_news(company_name):
    urls = get_article_urls(company_name)
    articles = []
    for url in urls:
        print(f"Scraping article: {url}")
        art = scrape_article(url)
        if art:
            articles.append(art)
        time.sleep(random.uniform(1, 3))
    return articles

# ---------------- Sentiment Analysis Function ---------------- #

def analyze_sentiment(text):
    try:
        result = sentiment_analyzer(text[:512])
        label = result[0]['label']
        score = result[0]['score']
        if score < 0.6:
            return "Neutral"
        return "Positive" if label.upper() == "POSITIVE" else "Negative"
    except Exception as e:
        print(f"Error during sentiment analysis: {e}")
        return "Neutral"

# ---------------- Summarization Function ---------------- #

def summarize_text(text):
    try:
        tokens = summarizer.tokenizer(text, return_tensors="pt")
        input_length = tokens.input_ids.size(1)
        if input_length < 50:
            max_len = input_length + 10
            min_len = max(10, max_len - 10)
        else:
            max_len = 50
            min_len = 25
        summary = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Error during summarization: {e}")
        return text

# ---------------- Translation Function ---------------- #

def translate_text(text, target_lang='hi'):
    """
    Translates the provided English text to Hindi using the Helsinki-NLP model.
    If the text is too long, it splits the text into chunks, translates each, and reassembles them.
    """
    try:
        # Clean the text before translation.
        text = re.sub(r'\s+', ' ', text).strip()
        max_tokens = 512
        tokens = translator.tokenizer(text, return_tensors="pt")
        token_length = tokens.input_ids.size(1)
        if token_length > max_tokens:
            sentences = text.split('. ')
            chunks = []
            current_chunk = ""
            for sentence in sentences:
                if not sentence.endswith('.'):
                    sentence += '.'
                candidate = current_chunk + " " + sentence if current_chunk else sentence
                candidate_tokens = translator.tokenizer(candidate, return_tensors="pt").input_ids.size(1)
                if candidate_tokens < max_tokens - 50:
                    current_chunk = candidate
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sentence
            if current_chunk:
                chunks.append(current_chunk)
            translations = []
            for chunk in chunks:
                translation = translator(chunk)
                translations.append(translation[0]['translation_text'])
            translated_text = " ".join(translations)
        else:
            translation = translator(text)
            translated_text = translation[0]['translation_text']
        translated_text = re.sub(r'\s+', ' ', translated_text).strip()
        return translated_text
    except Exception as e:
        print(f"Translation error: {e}")
        return text

# ---------------- New Helper Functions for Detailed Analysis ---------------- #

def comparative_analysis(articles):
    """
    Performs a comparative sentiment analysis over the provided articles.
    Returns a dictionary with counts for Positive, Negative, and Neutral sentiments.
    """
    distribution = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for article in articles:
        sentiment = article.get("Sentiment", "Neutral")
        if sentiment in distribution:
            distribution[sentiment] += 1
        else:
            distribution[sentiment] = 1
    return distribution

def filter_articles_by_keyword(articles, keyword):
    """
    Filters the list of articles to only include those where the keyword appears
    in the title or the concise summary.
    """
    keyword = keyword.lower()
    return [article for article in articles
            if keyword in article['Title'].lower() or keyword in article['Concise Summary'].lower()]

# ---------------- New Helper Functions for Hindi Summaries ---------------- #

def translate_article_summaries(articles):
    """
    Translates the 'Concise Summary' of each article into Hindi.
    Returns a dictionary mapping article title to its Hindi summary.
    """
    hindi_summaries = {}
    for article in articles:
        english_summary = article.get("Concise Summary", "")
        hindi_summary = translate_text(english_summary, target_lang='hi')
        hindi_summaries[article["Title"]] = hindi_summary
    return hindi_summaries

def combine_hindi_summaries(hindi_summaries):
    """
    Combines the individual Hindi summaries into one string for TTS.
    """
    combined = ""
    for summary in hindi_summaries.values():
        combined += summary + " "
    return combined.strip()

def generate_hindi_summaries_and_tts(articles):
    """
    Translates each article's concise summary into Hindi,
    then combines them and generates TTS audio for the combined Hindi summaries.
    
    Returns:
        tuple: (hindi_summaries_dict, audio_file_path)
    """
    hindi_summaries = translate_article_summaries(articles)
    combined_text = combine_hindi_summaries(hindi_summaries)
    audio_path = text_to_speech(combined_text, translate_if_english=False, lang='hi')
    return hindi_summaries, audio_path

# ---------------- Text-to-Speech Function ---------------- #

def text_to_speech(text, translate_if_english=True, lang=None):
    """
    Converts text into an audio file using gTTS.
    
    If 'lang' is provided, that language is used for TTS. Otherwise, if translate_if_english is True
    and the text is detected as English, the text is translated to Hindi before TTS.
    NOTE: When called with lang='hi', it assumes the text is already in Hindi.
    
    Returns:
        str: The file path of the generated audio file, or None if conversion fails.
    """
    try:
        if lang is not None:
            if lang == 'hi':
                text_for_tts = text
                used_lang = 'hi'
            else:
                text_for_tts = text
                used_lang = lang
        else:
            detected_lang = detect(text)
            if translate_if_english and detected_lang == 'en':
                text_for_tts = translate_text(text, target_lang='hi')
                used_lang = 'hi'
            else:
                text_for_tts = text
                used_lang = detected_lang
        
        tts = gTTS(text=text_for_tts, lang=used_lang, slow=False)
        audio_path = "output_audio.mp3"
        tts.save(audio_path)
        print(f"Audio saved at: {audio_path} (Language used: {used_lang})")
        return audio_path
    except Exception as e:
        print(f"Error during text-to-speech conversion: {e}")
        return None

# ---------------- Main Execution Block ---------------- #

if __name__ == "__main__":
    company = "Tesla"
    print(f"Extracting news articles for: {company}\n")
    articles = scrape_news(company)
    for art in articles:
        art["Sentiment"] = analyze_sentiment(art.get("Summary", ""))
        art["Concise Summary"] = summarize_text(art.get("Summary", ""))
        print("="*50)
        print(f"URL: {art['URL']}")
        print(f"Title: {art['Title']}")
        print(f"Original Summary: {art['Summary']}")
        print(f"Concise Summary: {art['Concise Summary']}")
        print(f"Publication Date: {art['Publication Date']}")
        print(f"Sentiment: {art['Sentiment']}")
    final_report = f"News Report for {company}:\n"
    for art in articles:
        final_report += f"\nTitle: {art['Title']}\nSentiment: {art['Sentiment']}\nSummary: {art['Concise Summary']}\n"
    
    # final_report remains in English.
    # Generate Hindi summaries for each article and combine them for Hindi TTS.
    hindi_summaries, audio_file = generate_hindi_summaries_and_tts(articles)
    
    print("\nFinal English Report:")
    print(final_report)
    print("\nHindi Summaries (Per Article):")
    for title, summary in hindi_summaries.items():
        print(f"\nTitle: {title}\nHindi Summary: {summary}")
    if audio_file:
        print(f"\nText-to-Speech conversion completed. Audio saved at: {audio_file}")
    else:
        print("\nText-to-Speech conversion failed.")
