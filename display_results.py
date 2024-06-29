import finplot as fplt
import numpy as np
import pandas as pd
import yfinance as yf
import warnings

# Suppress specific FutureWarning from yfinance
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

# Set up symbol and interval
symbol = 'BTC-USD'
interval = '5m'
period = '7d'  # Last 7 days to get enough 5-minute intervals

# Pull some data using yfinance
data = yf.download(tickers=symbol, interval=interval, period=period)

# Format it in pandas
data = data.reset_index()  # Reset index to move 'Date' into a column
data = data.rename(columns={'Datetime': 'time', 'Open': 'open',
                   'Close': 'close', 'High': 'high', 'Low': 'low', 'Volume': 'volume'})

print(type(data.loc[0, 'time']))

# Create the plot
ax = fplt.create_plot(symbol)

# Plot candlesticks
candles = data[['time', 'open', 'close', 'high', 'low']]
fplt.candlestick_ochl(candles, ax=ax)

# Overlay volume on the top plot
volumes = data[['time', 'open', 'close', 'volume']]
fplt.volume_ocv(volumes, ax=ax.overlay())

# Put an MA on the close price
fplt.plot(data['time'], data['close'].rolling(
    25).mean(), ax=ax, legend='ma-25')

lo_wicks = data[['open', 'close']].T.min() - data['low']
data.loc[(lo_wicks > lo_wicks.quantile(0.99)), 'marker'] = data['low']
fplt.plot(data['time'], data['marker'], ax=ax,
          color='#4a5', style='^', legend='dumb mark', width=5)

# Add a line at the trade price
fplt.add_line((700, data['close'].iloc[700]), (850, data['close'].iloc[850]),
              color='g', width=5, style='-', ax=ax)

print(data)
# Restore view (X-position and zoom) if we ever run this example again
fplt.autoviewrestore()

# Show the plot
# fplt.show()
