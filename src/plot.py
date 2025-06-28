import matplotlib.pyplot as plt

dates = [
    "2024-04-01", "2024-04-02", "2024-04-03", "2024-04-04", "2024-04-05",
    "2024-04-08", "2024-04-09", "2024-04-10", "2024-04-11", "2024-04-12",
    "2024-04-15", "2024-04-16", "2024-04-17", "2024-04-18", "2024-04-19",
    "2024-04-22", "2024-04-23", "2024-04-24", "2024-04-25", "2024-04-26",
    "2024-04-29", "2024-04-30"
]

# Paste your lists here
without_sentiment = [
    100000.00,   # 2024-04-01
    100000.00,   # 2024-04-02
    119923.12,   # 2024-04-03  ← spike (SHORT)
    99904.42,    # 2024-04-04  ← back to base (COVER)
    99904.42,    # 2024-04-05
    119613.07,   # 2024-04-08  ← spike (SHORT)
    99904.42,    # 2024-04-09  ← back to base (COVER)
    119755.81,   # 2024-04-10  ← spike (SHORT)
    100125.55,   # 2024-04-11  ← base after COVER/BUY
    100125.55,   # 2024-04-12
    119717.00,   # 2024-04-15  ← spike (SELL)
    100238.30,   # 2024-04-16  ← base after SHORT/COVER
    120230.30,   # 2024-04-17  ← spike (COVER)
    100352.54,   # 2024-04-18  ← base after SHORT/COVER
    100352.54,   # 2024-04-19
    100352.54,   # 2024-04-22
    100352.54,   # 2024-04-23
    100098.14,   # 2024-04-24  ← base after SHORT/COVER
    119975.27,   # 2024-04-25  ← spike (COVER)
    99675.77,    # 2024-04-26  ← base after SHORT
    99675.77,    # 2024-04-29  ← base (no trade)
    99675.77     # 2024-04-30
]



one_time_sentiment = [
    100000.00, 119893.51, 100139.23, 120157.93, 100237.17,
    120282.72, 100091.99, 100091.99, 120046.55, 100359.89,
    100359.89, 120346.73, 100522.73, 100522.73, 120589.37,
    100522.73, 100522.73, 100522.73, 100522.73, 120569.75,
    120143.77, 120143.77
]

daily_sentiment = [

    119723.48, 119723.48, 119861.52, 119767.56, 119863.84,
    119906.76, 119906.76, 119765.24, 119984.48, 119142.32,
    119414.92, 119414.92, 119798.88, 119958.96, 120070.32,
    120209.52, 120209.52, 120086.56, 119840.64, 119739.72,
    120126.00, 119320.96
]

plt.figure(figsize=(12, 6))
# Draw a vertical line for the "jump" on first day for daily_sentiment
plt.vlines(
    x=dates[0],
    ymin=100000,
    ymax=119723.48,
    color='tab:blue',
    linestyle='-',
    linewidth=1.8,
)
plt.plot(
    dates, without_sentiment,
    label='Without Sentiment',
    linestyle='-',        # Dotted line
    color='tab:orange',
    alpha=0.6,            # Less dark
    linewidth=2,
    marker=None           # No markers
)

plt.plot(
    dates, one_time_sentiment,
    label='One-Time Sentiment',
    linestyle='--',       # Dashed line
    color='tab:green',
    alpha=0.6,
    linewidth=2,
    marker=None
)

plt.plot(
    dates, daily_sentiment,
    label='Daily Sentiment',
    linestyle='-',       # Dash-dot line
    color='tab:blue',
    alpha=0.9,
    linewidth=2,
    marker=None
)

plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.title("Portfolio Value Over Time – All Strategies")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()
