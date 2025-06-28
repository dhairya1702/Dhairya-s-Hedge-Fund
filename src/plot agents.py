import matplotlib.pyplot as plt
import pandas as pd

buffet = [
    100000.00, 100000.00, 100000.00, 100000.00, 100000.00,
    100000.00, 100000.00, 100000.00, 100000.00, 100000.00,
    100000.00, 100000.00, 119924.02, 140474.26, 179696.08,
    160679.44, 155235.16, 190973.17, 241731.55, 273210.96,
    495763.88, 934154.52, 945794.20, 971983.48, 948927.96,
    1810935.80, 3578376.44, 3440491.00, 7113257.72, 14813353.72,
    14813353.72
]

cathie = [
    100000.00,100000.00,100000.00,100000.00,100000.00,100000.00,100000.00,100258.26,
    100258.26,100480.04,101307.45,102586.95,102160.45,102160.45,101742.48,101540.73,
    111547.53, 114412.38, 114412.38, 111345.78, 114253.84, 113354.44, 113354.44, 112574.96,
    110608.19, 112339.62, 113970.19, 113970.19, 116558.93, 120148.67,119514.12
]

bill=[
 100000.00,100000.00, 101152.75, 101152.75,101152.75, 101152.75, 100935.75, 101194.75, 101194.75, 101408.21,
 101408.21, 102416.21, 102005.21, 102005.21, 101602.43, 101479.13, 101479.13,101479.13,101479.13, 101479.13,
 102518.97, 102197.37, 102197.37, 101775.13, 100825.09, 101661.45, 102597.50, 102597.50, 104083.60, 105994.30,105604.40
]

dates = pd.date_range(start='2023-05-04', periods=31, freq='B')
date_labels = dates.strftime('%Y-%m-%d')

fig, ax1 = plt.subplots(figsize=(14,6))

ax1.set_xlabel('Date')
ax1.set_ylabel('Portfolio value in millions')
line_buffet,=ax1.plot(date_labels, buffet, label='Warren Buffet')
ax1.tick_params(axis='y')
ax1.tick_params(axis='x', rotation=45, labelsize=9)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

line_cathie, = ax2.plot(date_labels, cathie, color='tab:orange', label='Cathie Woods')
line_bill, = ax2.plot(date_labels, bill, color='tab:green', label='Bill Ackman')
ax2.set_ylabel('Portfolio value (Cathie & Bill)')
ax2.tick_params(axis='y')

# Legend: Combine all lines
lines = [line_buffet, line_cathie, line_bill]
labels = [l.get_label() for l in lines]
fig.legend(lines, labels, loc='upper left', bbox_to_anchor=(0.1, 0.93))

plt.title('Portfolio Value over Time: Buffet vs Cathie & Bill')
fig.tight_layout()
plt.show()
