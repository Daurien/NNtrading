import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from flags_pennants import find_flags_pennants_pips, find_flags_pennants_trendline, runAnalysis


data = pd.read_csv('./TechnicalAnalysisAutomation/BTCUSDT3600.csv')
data['date'] = data['date'].astype('datetime64[s]')
data = data.set_index('date')
# data.index = data.index[::-1]
# data = data.iloc[::-1]

# data = np.log(data)

runAnalysis(data['close'].to_numpy())


