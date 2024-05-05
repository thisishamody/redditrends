import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import bigrams, trigrams
import praw
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure required NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize Reddit API client
reddit = praw.Reddit(
    client_id="1u34wifQ-abMQRn63mYXww",
    client_secret="8GqiM6Tet4JszgrAu61RFXLRhHykaQ",
    user_agent="scraper 1.0 by /u/rustydusty10"
)

def fetch_posts(subreddit_name, limit=500):
    subreddit = reddit.subreddit(subreddit_name)
    posts_data = []
    for submission in subreddit.hot(limit=limit):
        post = {
            "title": submission.title,
            "created_utc": submission.created_utc
        }
        posts_data.append(post)
    return pd.DataFrame(posts_data)

def preprocess_text(text):
    custom_stopwords = set(['new', 'technology', 'latest', 'says', 'say'])
    stop_words = list(set(stopwords.words('english')) | custom_stopwords)
    tokenizer = word_tokenize
    lemmatizer = WordNetLemmatizer()
    words = tokenizer(text.lower())
    filtered_words = [lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stop_words]
    bi_tri_grams = [' '.join(gram) for gram in list(bigrams(filtered_words)) + list(trigrams(filtered_words))]
    return filtered_words + bi_tri_grams

def extract_keywords_with_tfidf(dataframe, top_n=10):
    custom_stopwords = set(['new', 'technology', 'latest', 'says', 'say'])
    stop_words = list(set(stopwords.words('english')) | custom_stopwords)
    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=1000, ngram_range=(1, 3))
    tfidf_matrix = tfidf_vectorizer.fit_transform(dataframe['title'])
    feature_names = tfidf_vectorizer.get_feature_names_out()
    dense = tfidf_matrix.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=feature_names)
    importance = df.sum(axis=0).sort_values(ascending=False)[:top_n]
    return importance.index.tolist(), importance.values.tolist()

def app():
    st.title("Reddit Trend Analyzer")
    subreddit_name = st.text_input("Enter the subreddit name:", "technology")
    limit = st.slider("Select how many posts to analyze:", 100, 1000, 500)
    if st.button("Analyze"):
        posts_df = fetch_posts(subreddit_name, limit)
        if not posts_df.empty:
            keywords, scores = extract_keywords_with_tfidf(posts_df, top_n=10)
            fig, ax = plt.subplots()
            ax.bar(keywords, scores, color='blue')
            ax.set_xlabel('Keywords')
            ax.set_ylabel('TF-IDF Score')
            ax.set_title('Top Keywords by TF-IDF Scores')
            ax.set_xticklabels(keywords, rotation=45, ha='right')
            st.pyplot(fig)
        else:
            st.write("No posts fetched, try a different subreddit or increase the limit.")

if __name__ == "__main__":
    app()