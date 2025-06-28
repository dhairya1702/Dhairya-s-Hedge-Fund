from langchain_core.messages import HumanMessage
from graph.state import AgentState, show_agent_reasoning
from utils.progress import progress
import pandas as pd
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tools.api import get_insider_trades, get_company_news


##### Sentiment Agent #####
def sentiment_agent(state: AgentState):
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
    end_date = data.get("end_date")
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

        progress.update_status("sentiment_agent", ticker, "Fetching company news")

        # Get the company news
        company_news = get_company_news(ticker, end_date, limit=100)
        #print(company_news[:5])

        # Get the sentiment from the company news
        # Compute FinBERT sentiment ONCE for all headlines
        finbert_sentiments = [finbert_sentiment(n.title or "") for n in company_news]
        sentiment = pd.Series(finbert_sentiments)

        news_signals = np.where(
            sentiment == "negative", "bearish",
            np.where(sentiment == "positive", "bullish", "neutral")
        ).tolist()

        print("News signals", news_signals)
        
        progress.update_status("sentiment_agent", ticker, "Combining signals")
        # Combine signals from both sources with weights
        insider_weight = 0.3
        news_weight = 0.7
        
        # Calculate weighted signal counts
        print("bullish signals count:",news_signals.count("bullish"))
        print("bullish trades count:", insider_signals.count("bullish"))
        print("bearish signals count:", news_signals.count("bearish"))
        print("bearish trades count:", insider_signals.count("bearish"))
        bullish_signals = (
            insider_signals.count("bullish") * insider_weight +
            news_signals.count("bullish") * news_weight
        )
        bearish_signals = (
            insider_signals.count("bearish") * insider_weight +
            news_signals.count("bearish") * news_weight
        )

        if bullish_signals > bearish_signals:
            overall_signal = "bullish"
        elif bearish_signals > bullish_signals:
            overall_signal = "bearish"
        else:
            overall_signal = "neutral"

        # Calculate confidence level based on the weighted proportion
        total_weighted_signals = len(insider_signals) * insider_weight + len(news_signals) * news_weight
        confidence = 0  # Default confidence when there are no signals
        if total_weighted_signals > 0:
            confidence = round(max(bullish_signals, bearish_signals) / total_weighted_signals, 2) * 100
        reasoning = f"Weighted Bullish signals: {bullish_signals:.1f}, Weighted Bearish signals: {bearish_signals:.1f}"

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
