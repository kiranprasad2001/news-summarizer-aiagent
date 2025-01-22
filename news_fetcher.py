import requests
import json
import nltk
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext

# --- NLTK Resource Downloads ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# --- API Key ---
API_KEY = "21f582975c7c42d9a9bdd9bf9b9e169c"  # Replace with your actual API key

# --- Core Functionalities ---
def fetch_news_articles(api_key, keyword="technology", language="en", page_size=5):
    """
    Fetches news articles from the NewsAPI based on a keyword.

    Args:
        api_key (str): Your NewsAPI key.
        keyword (str): The keyword to search for in news articles.
        language (str): The language of the articles (e.g., "en" for English).
        page_size (int): Number of articles to fetch for each request.

    Returns:
        list: A list of dictionaries, each containing information about an article.
        or None: If there is any exception
    """
    base_url = "https://newsapi.org/v2/everything"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {
        "q": keyword,
        "language": language,
        "pageSize": page_size,
        "sortBy": "relevancy",
    }
    try:
        response = requests.get(base_url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        if data["status"] == "ok" and data["totalResults"] > 0:
            return data["articles"]
        else:
            print("Error fetching articles or no results.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error during request: {e}")
        return None

def clean_text(text):
    """Cleans the input text."""
    if not text:
        return ""
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower()

def preprocess_text(text):
    """Preprocesses the text by cleaning it and tokenizing it into words."""
    if not text:
        return []
    cleaned_text = clean_text(text)
    tokens = word_tokenize(cleaned_text)
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [w for w in tokens if w not in stop_words]
    return filtered_tokens

def extract_article_text(url):
    """Extracts the text from a given URL using BeautifulSoup."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "lxml")
        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text() for p in paragraphs])
        return text
    except Exception as e:
        print(f"Error extracting text from {url}: {e}")
        return None

def summarize_text(text, summary_ratio=0.2):
    """Summarizes the text using TF-IDF based extractive method."""
    if not text:
        return ""
    sentences = sent_tokenize(text)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
    ranked_sentences = sorted(
        ((score, i) for i, score in enumerate(sentence_scores)), reverse=True
    )
    num_sentences = int(len(sentences) * summary_ratio)
    summary_sentences = [sentences[i] for score, i in ranked_sentences[:num_sentences]]
    summary = " ".join(summary_sentences)
    return summary

def process_articles(api_key, keyword, summary_ratio):
    """Fetches and processes articles based on given input"""
    articles = fetch_news_articles(api_key, keyword=keyword, page_size=2)
    results = []
    if articles:
        for i, article in enumerate(articles):
            article_data = {
                "title": article["title"],
                "source": article["source"]["name"],
                "url": article["url"],
                "summary": ""
            }
            article_text = extract_article_text(article["url"])
            if article_text:
                summary = summarize_text(article_text, summary_ratio=summary_ratio)
                article_data["summary"] = summary
            results.append(article_data)
    return results

# --- GUI Setup and Execution ---
def create_gui():
    """Creates the main GUI window."""
    window = tk.Tk()
    window.title("News Summarizer")
    window.geometry("800x600")

    # Keyword Input
    keyword_label = ttk.Label(window, text="Keyword/Topic:")
    keyword_label.pack(pady=5)
    keyword_entry = ttk.Entry(window, width=50)
    keyword_entry.pack(pady=5)

    # Summary Length Control
    summary_label = ttk.Label(window, text="Summary Length Ratio (0-1):")
    summary_label.pack(pady=5)
    summary_slider = ttk.Scale(
        window, from_=0, to=1, orient="horizontal", length=300
    )
    summary_slider.set(0.2)  # Set default value
    summary_slider.pack(pady=5)

    # Button to Trigger Summarization
    summarize_button = ttk.Button(
        window, text="Summarize", command=lambda: process_and_display(keyword_entry, summary_slider, output_text)
    )
    summarize_button.pack(pady=10)

    # Output Text Area
    output_text = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=90, height=20)
    output_text.pack(pady=10)

    return window


def display_results(results, output_text):
    """Displays the results in the output text area."""
    output_text.delete(1.0, tk.END)  # Clear the text area
    if not results:
        output_text.insert(tk.END, "Could not fetch articles!\n")
    else:
         for i, article in enumerate(results):
            output_text.insert(tk.END, f"--- Article {i+1} ---\n")
            output_text.insert(tk.END, f"Title: {article['title']}\n")
            output_text.insert(tk.END, f"Source: {article['source']}\n")
            output_text.insert(tk.END, f"URL: {article['url']}\n")
            if article["summary"]:
                output_text.insert(tk.END, f"Summary:\n{article['summary']}\n\n")
            else:
                output_text.insert(tk.END, "Could not process this article.\n\n")

def process_and_display(keyword_entry, summary_slider, output_text):
    """Fetches articles, generates summaries and updates output area"""
    keyword = keyword_entry.get()
    summary_ratio = summary_slider.get()
    results = process_articles(API_KEY, keyword, summary_ratio)
    display_results(results, output_text)

if __name__ == "__main__":
    window = create_gui()
    window.mainloop()