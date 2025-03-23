import streamlit as st
import matplotlib.pyplot as plt
import utils  # Ensure utils.py is in the same directory

def main():
    # Configure the Streamlit page.
    st.set_page_config(
        page_title="News Summarization, Sentiment Analysis & Hindi TTS",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Main header and description.
    st.title("News Summarization, Sentiment Analysis & Hindi TTS")
    st.markdown("""
        This application extracts news articles for a given company, performs sentiment analysis, and summarizes the content in English.
        It then translates each article's concise summary into Hindi and generates a combined Hindi audio report.
        The English report is displayed below, and a separate section shows the translated Hindi summaries organized by article.
    """)
    
    # Sidebar for user input.
    with st.sidebar:
        st.header("Input Parameters")
        company = st.text_input("Company Name", value="Tesla")
        generate_btn = st.button("Generate Report")
    
    if generate_btn:
        with st.spinner("Fetching and processing news articles..."):
            # Extract news articles.
            articles = utils.scrape_news(company)
            if not articles:
                st.error("No articles found. Please try another company or check your network.")
                return
            
            # Process each article for sentiment analysis and summarization.
            for article in articles:
                article["Sentiment"] = utils.analyze_sentiment(article.get("Summary", ""))
                article["Concise Summary"] = utils.summarize_text(article.get("Summary", ""))
            
            # Build the final English report.
            final_report = f"News Report for {company}:\n"
            for article in articles:
                final_report += (
                    f"\nTitle: {article['Title']}\n"
                    f"Sentiment: {article['Sentiment']}\n"
                    f"Summary: {article['Concise Summary']}\n"
                )
            
            # Generate Hindi summaries for each article and combine them for Hindi TTS.
            hindi_summaries, audio_file = utils.generate_hindi_summaries_and_tts(articles)
        
        st.success("Report Generated Successfully!")
        
        # Compute sentiment distribution for detailed analysis.
        sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
        for article in articles:
            sentiment = article.get("Sentiment", "Neutral")
            if sentiment in sentiment_counts:
                sentiment_counts[sentiment] += 1
            else:
                sentiment_counts[sentiment] = 1
        
        # Create a bar chart for sentiment distribution.
        fig, ax = plt.subplots()
        ax.bar(sentiment_counts.keys(), sentiment_counts.values(), color=['green', 'red', 'gray'])
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Count')
        ax.set_title('Sentiment Distribution')
        
        # Layout: Final English report and Hindi audio in two columns.
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("### Final News Report (English)")
            st.markdown(final_report)
        with col2:
            if audio_file:
                st.markdown("### Listen to the Hindi Audio Report")
                st.audio(audio_file)
            else:
                st.error("Audio file generation failed.")
        
        # Display the sentiment distribution graph.
        st.markdown("### Sentiment Distribution")
        st.pyplot(fig)
        
        # Display the Hindi summaries for each article.
        st.markdown("### Translated Hindi Summaries (Per Article)")
        for title, hindi_summary in hindi_summaries.items():
            st.markdown(f"**{title}**")
            st.markdown(hindi_summary)
            st.markdown("---")
        
        # Display detailed article information.
        st.markdown("### Detailed Articles")
        for article in articles:
            with st.expander(article['Title']):
                st.markdown(f"**Publication Date:** {article['Publication Date']}")
                st.markdown(f"**Sentiment:** {article['Sentiment']}")
                st.markdown(f"**Summary:** {article['Concise Summary']}")
                st.markdown(f"[Read Full Article]({article['URL']})")
    
if __name__ == "__main__":
    main()
