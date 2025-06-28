from typing import Optional

from langchain_core.messages import HumanMessage
from graph.state import AgentState, show_agent_reasoning
from utils.progress import progress
import pandas as pd
import numpy as np
import json
from datetime import datetime
from tools.api import get_insider_trades, get_company_news
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

"""# Load model and tokenizer ONCE at import
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
LABELS = ["negative", "neutral", "positive"]

def finbert_sentiment(text):
    # For headlines, text can be short
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).numpy().flatten()
    label_idx = int(np.argmax(probs))
    return LABELS[label_idx]"""



##### Sentiment Agent #####
def temp_sentiment_agent(state: AgentState, date: datetime = None):
    # Load model and tokenizer ONCE at import
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    LABELS = ["negative", "neutral", "positive"]

    def finbert_sentiment(text):
        # For headlines, text can be short
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1).numpy().flatten()
        label_idx = int(np.argmax(probs))
        return LABELS[label_idx]

    """Analyzes market sentiment and generates trading signals for multiple tickers."""
    data = state.get("data", {})
    end_date = date.strftime("%Y-%m-%d") if date else data.get("end_date")
    tickers = data.get("tickers")

    # Initialize sentiment analysis for each ticker
    sentiment_analysis = {}

    for ticker in tickers:
        progress.update_status("sentiment_agent", ticker, "Fetching insider trades")

        # Get the insider trades
        insider_trades = get_insider_trades(
            ticker=ticker,
            end_date=end_date,
            limit=1000,
        )

        progress.update_status("sentiment_agent", ticker, "Analyzing trading patterns")

        # Get the signals from the insider trades
        transaction_shares = pd.Series([t.transaction_shares for t in insider_trades]).dropna()
        insider_signals = np.where(transaction_shares < 0, "bearish", "bullish").tolist()
        #print("Insider trades:", transaction_shares.tolist())
        #print("Insider signals:", insider_signals)

        # Remove 0.0 trades before computing signals
        nonzero_trades = transaction_shares[transaction_shares != 0]

        filtered_insider_signals = np.where(nonzero_trades < 0, "bearish", "bullish").tolist()
        #print("Filtered Insider trades:", nonzero_trades.tolist())
        #print("Filtered Insider signals:", filtered_insider_signals)

        progress.update_status("sentiment_agent", ticker, "Fetching company news")

        # Get the company news
        # Convert datetime 'date' to string if given, else use end_date string
        date_str = date.strftime("%Y-%m-%d") if date else end_date

        company_news = get_company_news(
            ticker,
            end_date=date_str,
            start_date=date_str,  # fetch news only for this day
            limit=100,
        )

        # Compute FinBERT sentiment ONCE for all headlines
        finbert_sentiments = [finbert_sentiment(n.title or "") for n in company_news]
        sentiment = pd.Series(finbert_sentiments)

        news_signals = np.where(
            sentiment == "negative", "bearish",
            np.where(sentiment == "positive", "bullish", "neutral")
        ).tolist()

        """# Print, using already computed FinBERT sentiments
        for i, (n, fb_sent) in enumerate(zip(company_news[:5], finbert_sentiments[:5])):
            print(f"[{i + 1}] Date: {getattr(n, 'date', None)}")
            print(f"    Headline: {getattr(n, 'headline', getattr(n, 'title', None))}")
            print(f"    Summary: {getattr(n, 'summary', getattr(n, 'content', None))}")
            print(f"    Sentiment (API): {getattr(n, 'sentiment', None)}")
            print(f"    FinBERT Sentiment: {fb_sent}\n")"""

        progress.update_status("sentiment_agent", ticker, "Combining signals")
        # Combine signals from both sources with weights
        insider_weight = 0.0
        news_weight = 1.0

        """# Calculate weighted signal counts
        bullish_signals = (
                filtered_insider_signals.count("bullish") * insider_weight +
                news_signals.count("bullish") * news_weight
        )
        bearish_signals = (
                filtered_insider_signals.count("bearish") * insider_weight +
                news_signals.count("bearish") * news_weight
        )

        if bullish_signals > bearish_signals:
            overall_signal = "bullish"
        elif bearish_signals > bullish_signals:
            overall_signal = "bearish"
        else:
            overall_signal = "neutral"

        # Calculate confidence level based on the weighted proportion
        total_weighted_signals = len(filtered_insider_signals) * insider_weight + len(news_signals) * news_weight
        confidence = 0  # Default confidence when there are no signals
        if total_weighted_signals > 0:
            confidence = round(max(bullish_signals, bearish_signals) / total_weighted_signals, 2) * 100
        reasoning = f"Weighted Bullish signals: {bullish_signals:.1f}, Weighted Bearish signals: {bearish_signals:.1f}"
        """
        ###############################################################################################################
        # Count the signals
        bullish_count = news_signals.count("bullish")
        bearish_count = news_signals.count("bearish")
        neutral_count = news_signals.count("neutral")
        total_signals = bullish_count + bearish_count + neutral_count

        # Set a threshold for majority (e.g., > 40% of all signals must agree)
        majority_threshold = 0.40

        if total_signals == 0:
            overall_signal = "neutral"
            confidence = 0
            reasoning = "No sentiment signals available."
        else:
            # Calculate proportions
            bullish_prop = bullish_count / total_signals
            bearish_prop = bearish_count / total_signals
            neutral_prop = neutral_count / total_signals
            print(f'Bullish:{bullish_prop} Bearish:{bearish_prop} Neutral:{neutral_prop}')

            if bullish_prop > majority_threshold:
                overall_signal = "bullish"
                confidence = round(bullish_prop * 100, 2)
                reasoning = f"{confidence}% bullish signals"
            elif bearish_prop > majority_threshold:
                overall_signal = "bearish"
                confidence = round(bearish_prop * 100, 2)
                reasoning = f"{confidence}% bearish signals"
            else:
                overall_signal = "neutral"
                confidence = round(neutral_prop * 100, 2)
                reasoning = f"{confidence}% neutral signals, not enough consensus for bullish or bearish."

        # Result: overall_signal, confidence, reasoning

        sentiment_analysis[ticker] = {
            "signal": overall_signal,
            "confidence": confidence,
            "reasoning": reasoning,
        }

        progress.update_status("sentiment_agent", ticker, "Done")

    # Create the sentiment message
    message = HumanMessage(
        content=json.dumps(sentiment_analysis),
        name="sentiment_agent",
    )

    # Print the reasoning if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(sentiment_analysis, "Sentiment Analysis Agent")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"]["sentiment_agent"] = sentiment_analysis

    # Also store a simpler version for other agents to use
    state["data"]["sentiment_scores"] = {
        ticker: {
            "score": round(info["confidence"] / 100, 2),
            "summary": f"Sentiment is {info['signal']} based on {info['reasoning']}"
        }
        for ticker, info in sentiment_analysis.items()
    }

    return {
        "messages": [message],
        "data": data,
    }
