# Reddit based Stock-Market Data Sentiment Analysis

This project analyzes Reddit discussions related to Indian stock market indices such as **Nifty**, **Sensex**, **BankNifty**, and **Intraday trading**. It scrapes Reddit posts using PRAW, performs sentiment analysis, generates composite trading signals, and visualizes market sentiment trends.

---

##  Project Overview

- **Data Source**: Reddit posts from subreddits like `IndiaInvestments`, `StockMarket`, `nifty`, and `stocktrading`
- **Goal**: Extract actionable sentiment signals from retail investor discussions
- **Tech Stack**: Python, PRAW, Pandas, TextBlob, Scikit-learn, Seaborn, WordCloud
- **Outputs**:
  - Cleaned and labeled dataset (`reddit_market_data.csv`, `.parquet`)
  - Composite signal distribution chart
  - Word cloud of dominant market terms

---

##  Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/reddit-market-sentiment.git
cd reddit-market-sentiment
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Script
```bash
python reddit.py
```

## Project Structure
```
reddit-market-sentiment/
├── reddit.py                        # Main python script
├── requirements.txt                 # Python dependencies
├── README.md                        # Project overview and setup
├── TECHNICAL_DOCUMENTATION.md       # Detailed methodology
├── data/
│   ├── reddit_market_data.csv
│   ├── reddit_market_data.parquet
├── visuals/
│   ├── signal_distribution_sampled.png
│   ├── wordcloud_sampled.png
```



