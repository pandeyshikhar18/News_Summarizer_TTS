from flask import Flask, request, jsonify
import utils

app = Flask(__name__)

@app.route('/news', methods=['POST'])
def get_news():
    """
    Endpoint to retrieve news articles, perform sentiment analysis, and generate a Hindi TTS report.
    Expects a JSON payload with a 'company' key.
    Returns a JSON response with:
      - Company name
      - Articles (each with title, sentiment, summarized text, URL, and publication date)
      - Final text report
      - Audio file path for the Hindi TTS output
    """
    data = request.get_json()
    if not data or 'company' not in data:
        return jsonify({'error': 'Missing company parameter'}), 400

    company = data['company']

    # Extract news articles for the given company.
    news_articles = utils.scrape_news(company)

    # Process each article: sentiment analysis and summarization.
    for article in news_articles:
        article['Sentiment'] = utils.analyze_sentiment(article.get("Summary", ""))
        article["Concise Summary"] = utils.summarize_text(article.get("Summary", ""))

    # Build a final text report.
    final_report = f"News Report for {company}:\n"
    for article in news_articles:
        final_report += (
            f"\nTitle: {article['Title']}\n"
            f"Sentiment: {article['Sentiment']}\n"
            f"Summary: {article['Concise Summary']}\n"
        )

    # Convert the final report to Hindi audio.
    audio_file = utils.text_to_speech(final_report, lang='hi')

    response = {
        "Company": company,
        "Articles": news_articles,
        "Final Report": final_report,
        "Audio": audio_file  # This is the path to the generated audio file.
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
