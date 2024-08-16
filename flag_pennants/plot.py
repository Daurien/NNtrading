import pandas as pd
import finplot
from .flagpattern import FlagPattern


def plot_patterns(history: pd.DataFrame, patterns: pd.DataFrame):
    """
    This function plots the top 10 flag or pennant patterns found patterns dataframe.

    Parameters:
    history (pd.DataFrame): A pandas DataFrame containing historical data with columns 'Time', 'Open', 'Close', 'High', 'Low'.
    patterns (pd.DataFrame): A pandas DataFrame containing the flag or pennant patterns found in the historical data.

    Returns:
    None. The function creates a plot using the 'finplot' library to visualize the patterns.
    """
    for i in range(min(10, len(patterns))):
        pattern = patterns.iloc[i]
        flag_width = pattern.pennant_width if 'pennant_width' in pattern else pattern.flag_width
        subset = history.iloc[int(pattern.base_x):int(pattern.base_x + flag_width *
                                                      3 + pattern.pole_width)].reset_index(drop=True)
        # subset.reset_index(drop=True)
        flag = FlagPattern(0,
                           history.iloc[int(pattern.base_x)].Close,
                           pattern.pole_width,
                           history.iloc[int(pattern.base_x + pattern.pole_width)].Close,
                           resist_slope=pattern.resist_slope,
                           support_slope=pattern.support_slope,
                           resist_intercept=pattern.resist_intercept,
                           support_intercept=pattern.support_intercept,
                           flag_width=flag_width)

        # Create the plot
        ax = finplot.create_plot('XAUUSD')

        # Plot candlesticks
        candles = subset[['Time', 'Open', 'Close', 'High', 'Low']]
        finplot.candlestick_ochl(candles, ax=ax)

        # Add a line for the tip
        finplot.add_line((flag.base_x, flag.base_y), (flag.tip_x, flag.tip_y), color='g', width=5, style='-', ax=ax)

        # Add support and resistance lines of the pattern
        finplot.add_line((flag.tip_x, flag.resist_intercept), (flag.tip_x + flag.flag_width,
                                                               flag.resist_intercept + flag.resist_slope * flag.flag_width), color='b', width=5, style='-', ax=ax)
        finplot.add_line((flag.tip_x, flag.support_intercept), (flag.tip_x + flag.flag_width,
                                                                flag.support_intercept + flag.support_slope * flag.flag_width), color='b', width=5, style='-', ax=ax)

        # If return is positive add line for the trade
        if pattern['return'] > 0:
            trade_open = subset.iloc[int(flag.tip_x + flag.flag_width)].Close
            finplot.add_line((flag.tip_x + flag.flag_width, trade_open), (subset.iloc[int(
                flag.tip_x + flag.flag_width):]['Close'].idxmax(), trade_open + pattern['return']), color='r', width=5, style='.', ax=ax)

        finplot.show()
