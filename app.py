
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Use the real_results from your model
REAL_RESULTS = {"date": "2025-07-30", "prediction": "DOWN", "confidence": 51, "rsi": 41.9, "stock_price": np.float64(362.0), "change": np.float64(1.01), "headlines": [{"title": "UBL profit up 10%", "sentiment": "Positive", "score": 0.9}, {"title": "UBL declares dividend", "sentiment": "Positive", "score": 0.85}]}

# Page config
st.set_page_config(page_title="HBL AI Predictor", layout="centered")
st.title("📈 KSE-Insight: HBL Movement Forecast")
st.markdown("Powered by Business Recorder & AI Sentiment Analysis")

# Display key metrics
st.header("📅 Today's Prediction")
col1, col2, col3 = st.columns(3)
col1.metric("Outlook", REAL_RESULTS["prediction"])
col2.metric("Confidence", f"{REAL_RESULTS['confidence']}%")
col3.metric("RSI", f"{REAL_RESULTS['rsi']}")

col4, col5 = st.columns(2)
col4.metric("HBL Price", f"₨ {REAL_RESULTS['stock_price']}")
col5.metric("Today's Change", f"{REAL_RESULTS['change']}%")

# Show headlines
st.header("📰 Relevant News Headlines")
if len(REAL_RESULTS["headlines"]) == 0:
    st.write("No recent headlines available.")
else:
    for h in REAL_RESULTS["headlines"]:
        with st.expander(f"{h['title']} (Sentiment: {h['sentiment']})"):
            st.write(f"**Score:** {h['score']}")

# Sentiment chart
st.header("📊 Sentiment Summary")
if len(REAL_RESULTS["headlines"]) > 0:
    df = pd.DataFrame(REAL_RESULTS["headlines"])
    sent_counts = df['sentiment'].value_counts()
    fig, ax = plt.subplots()
    colors = {"Positive": "green", "Negative": "red", "Neutral": "gray"}
    sent_counts.plot(kind='bar', ax=ax, color=[colors.get(x, "gray") for x in sent_counts.index])
    ax.set_title("Headline Sentiment Impacting HBL")
    ax.set_ylabel("Count")
    st.pyplot(fig)
else:
    st.write("No sentiment data to display.")

# Footer
st.markdown("---")
st.caption(f"Updated: {REAL_RESULTS['date']} | Model: Logistic Regression + FinancialBERT")
st.info("💡 This is a prototype. Future versions will include automation and more stocks.")
