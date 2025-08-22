import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Step 1: Fetch historical stock data
ticker = 'META'
start_date = '2020-01-01'
end_date = '2025-04-01'
data = yf.download(ticker, start=start_date, end=end_date)

# Step 2: Bin prices instead of rounding to capture all levels
bin_width = 1  # Adjust for resolution
data['Price_Bin'] = (data['Close'] // bin_width) * bin_width  # Bin prices into $5 ranges

# Step 3: Apply time decay with proper weighting
decay_factor = 0.99
weights = np.array([decay_factor**i for i in range(len(data))][::-1])

# Compute weighted counts for each price bin
weighted_counts = data.groupby('Price_Bin', group_keys=False).apply(lambda x: np.sum(weights[:len(x)]))

# Step 4: Smooth the data (optional, adjust sigma)
prices = weighted_counts.index.to_numpy()  # Store index before smoothing
smoothed_counts = gaussian_filter(weighted_counts.to_numpy(), sigma=1)  # Reduce sigma

# Step 5: Visualization
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(prices, smoothed_counts, color='blue', label='Stability Depth')
ax.fill_between(prices, smoothed_counts, color='blue', alpha=0.3)
ax.set_xlabel('Price')
ax.set_ylabel('Stability Depth')
ax.set_title(f'{ticker} Price-Time Density Landscape')
plt.gca().invert_yaxis()  # Invert y-axis to show depth
plt.legend()
plt.show()
