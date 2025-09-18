#  Technical Documentation

##  Problem Statement

The goal is to extract sentiment-driven trading signals from Reddit discussions focused on Indian stock market indices. The solution is scalable, reproducible, and resilient to platform constraints.

---

##  Data Source

- **Platform**: Reddit
- **Access Method**: PRAW (Python Reddit API Wrapper)
- **Subreddits Used**: `IndiaInvestments`, `StockMarket`, `nifty`, `stocktrading`
- **Query Terms**: `nifty`, `sensex`, `banknifty`, `intraday`

---

##  Script Overview

### 1. **Scraping**
- Authenticated Reddit access using PRAW
- Subreddit validation with exception handling
- Query-based post extraction with title, content, score, and comment count

### 2. **Text Cleaning**
- Removal of URLs, mentions, hashtags, and non-alphabetic characters
- Lowercasing and stopword removal using NLTK

### 3. **Sentiment Analysis**
- Polarity scoring using TextBlob
- Signal classification:
  - `Buy` if polarity > 0.2
  - `Sell` if polarity < -0.2
  - `Hold` otherwise

### 4. **TF-IDF Vectorization**
- `TfidfVectorizer` with `max_features=1000`
- Converts cleaned text into numerical vectors for downstream modeling

### 5. **Composite Signal Generation**
- Weighted score:
  - Sentiment × 0.6
  - Log(Score) × 0.2
  - Log(Comments) × 0.2
- Final signal:
  - `Buy` if score > 0.7
  - `Sell` if score < 0.2
  - `Hold` otherwise

### 6. **Confidence Estimation**
- Standard deviation across sentiment, score, and comments

### 7. **Visualization**
- Bar chart of composite signals (sampled)
- Word cloud of dominant market terms

### 8. **Output Persistence**
- Saved as `.csv` and `.parquet` for flexibility
- Visuals saved as `.png`

---

##  Scalability & Maintainability

- Modular code structure with clear separation of concerns
- Exception handling for invalid subreddits
- Sampling ensures memory efficiency
- Easily extendable to more subreddits, timeframes, or vectorization methods

---

##  Limitations & Future Improvements

- Sentiment analysis via TextBlob may miss sarcasm or financial jargon
- Composite signal weights are heuristic — could be tuned via ML
- Subreddit coverage can be expanded for broader market sentiment
- Could integrate clustering or topic modeling for deeper insights

---

##  Evaluation Criteria Alignment

| Criterion                          | Implementation Highlights                          |
|-----------------------------------|----------------------------------------------------|
| Code quality                      | Modular, documented, exception-handled             |
| Data structure & efficiency       | Pandas, TF-IDF, log-scaled engagement              |
| Market understanding              | Focused on Indian indices and trading terms        |
| Problem-solving                   | Pivoted from Twitter, handled API constraints      |
| Scalability & maintainability     | Clean structure, extensible logic, reproducible    |
