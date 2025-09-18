#To import all the required libraries 
import praw
import pandas as pd
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
nltk.download("punkt")
nltk.download("stopwords")

#Authenticating with Reddit by adding client info
reddit = praw.Reddit(
    client_id="C8vTnuXTo0VHG0UBps3Tfw",
    client_secret="owGQSt0ukpICJvgZysxuFwUMxsSZVA",
    user_agent="r_scraper"
)

#Scraping all the Reddit Posts with limit of 2000
def scrape_reddit(subreddits, query, limit=2000):
    posts = []
    for sub in subreddits:
        try:
            subreddit = reddit.subreddit(sub)
            # Trigger a lightweight call to validate existence
            _ = subreddit.display_name  # Raises Redirect if invalid

            for post in subreddit.search(query, limit=limit):
                posts.append({
                    "Date": post.created_utc,
                    "Subreddit": sub,
                    "Title": post.title,
                    "Content": post.selftext,
                    "Score": post.score,
                    "Comments": post.num_comments
                })

        except Exception as e:
            print(f"Skipping subreddit '{sub}' due to error: {e}")
            continue

    return pd.DataFrame(posts)

df = scrape_reddit(
    subreddits = ["IndiaInvestments", "StockMarket", "nifty", "stocktrading"],
    query="nifty OR sensex OR banknifty OR intraday",
    limit=500
)

#Preprocessing the data and use the clean text for further analysis
def clean_text(text):
    text = re.sub(r"http\S+|@\S+|#\S+|RT", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    text = text.lower()
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in text.split() if word not in stop_words])

df["Text"] = df["Title"] + " " + df["Content"]
df["Cleaned"] = df["Text"].apply(clean_text)
df.drop_duplicates(subset="Cleaned", inplace=True)

#Using textblob for Sentiment Analysis: 'BUY', 'SELL' & 'Hold'
df["Sentiment"] = df["Cleaned"].apply(lambda x: TextBlob(x).sentiment.polarity)
df["Signal"] = df["Sentiment"].apply(lambda x: "Buy" if x > 0.2 else "Sell" if x < -0.2 else "Hold")

#Using TF-IDF Vectorization to create numerical vector for the data
tfidf = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf.fit_transform(df["Cleaned"])
tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf.get_feature_names_out())

#By finding Composite Signal and using Confidence Estimate we get normalised interpretable scores for the data vectors
df["CompositeScore"] = (
    df["Sentiment"] * 0.6 +
    np.log1p(df["Score"]) * 0.2 +
    np.log1p(df["Comments"]) * 0.2
)

df["CompositeSignal"] = df["CompositeScore"].apply(lambda x: "Buy" if x > 0.7 else "Sell" if x < 0.2 else "Hold")
df["Confidence"] = df[["Sentiment", "Score", "Comments"]].std(axis=1)

df_combined = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

#We store and save the data in both csv as well as parquet format
df_combined.to_parquet("reddit_market_data.parquet", index=False)
df_combined.to_csv("reddit_market_data.csv", index=False)

#Visualizations for analysis by using sampled data- (1)Bar graph & (2)Word Cloud
sample_size = min(500, len(df))
sample_df = df.sample(n=sample_size, random_state=42)

#Bar Graph
sns.countplot(data=sample_df, x="CompositeSignal", hue="CompositeSignal", palette="Set2", legend=False)
plt.title("Composite Trading Signal Distribution (Sampled)")
plt.savefig("signal_distribution_sampled.png")
plt.show()

#Word Cloud
sample_text = " ".join(sample_df["Cleaned"])
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(sample_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Market Sentiment WordCloud (Sampled)")
plt.savefig("wordcloud_sampled.png")
plt.show()