import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple


#########################################################################################
############# ROLLING WINDOW USING FUTURE AND PAST DATA #################################
#########################################################################################

def pivotid(dataframe, l, n1, n2):
    """
    This function identifies pivot points in a given dataframe based on a rolling window approach using past & FUTURE data.

    Parameters:
    dataframe (pandas.DataFrame): The dataframe containing the price data.
    l (int): The index of the current candle in the dataframe.
    n1 (int): The number of candles to consider before the current candle for identifying pivot low.
    n2 (int): The number of candles to consider after the current candle for identifying pivot high.

    Returns:
    int: A value indicating the type of pivot point.
         - 0: No pivot point.
         - 1: Pivot low.
         - 2: Pivot high.
         - 3: Both pivot low and pivot high.
    """

    # Check if the current index is within the range of the dataframe
    if l-n1 < 0 or l+n2 >= len(dataframe):
        return 0

    # Get the window of candles around the current candle
    window = dataframe.iloc[l-n1:l+n2+1]

    # Check if the current candle is a pivot low
    pivotlow = (dataframe.loc[l, 'Low'] <= window['Low']).all()

    # Check if the current candle is a pivot high
    pivothigh = (dataframe.loc[l, 'High'] >= window['High']).all()

    # Return the type of pivot point based on the conditions
    if pivotlow and pivothigh:
        return 3
    elif pivotlow:
        return 1
    elif pivothigh:
        return 2
    else:
        return 0

#########################################################################################
############# ROLLING WINDOW USING PAST DATA FOR RESISTANCE #############################
#########################################################################################


def rw_top(data: np.array, curr_index: int, order: int) -> bool:
    """
    This function identifies a local top in a given price series using a rolling window approach. It uses only paste data so the top is confirmed only after a lag equal to order.

    Parameters:
    data (np.array): A 1D numpy array containing the price series.
    curr_index (int): The index of the current candle in the series.
    order (int): The number of candles to consider for identifying the local top.

    Returns:
    bool: True if the current candle is a local top, False otherwise.
    """
    # Check if the current index is within the range of the data
    if curr_index < order * 2 + 1:
        return False

    # Initialize a flag to check if the current candle is a local top
    top = True

    # Calculate the index of the candle to start the rolling window
    k = curr_index - order

    # Get the price of the candle at index k
    v = data[k]

    # Iterate over the candles in the rolling window
    for i in range(1, order + 1):
        # If any of the candles in the window are higher than the K candle,
        # it is not a local top
        if data[k + i] > v or data[k - i] > v:
            top = False
            break

    # Return True if the K candle is a local top, False otherwise
    return top


#########################################################################################
############# ROLLING WINDOW USING PAST DATA FOR SUPPPORT ###############################
#########################################################################################

def rw_bottom(data: np.array, curr_index: int, order: int) -> bool:
    """
    This function identifies a local bottom in a given price series using a rolling window approach. It uses only paste data so the bottom is confirmed only after a lag equal to order.

    Parameters:
    data (np.array): A 1D numpy array containing the price series.
    curr_index (int): The index of the current candle in the series.
    order (int): The number of candles to consider for identifying the local bottom.

    Returns:
    bool: True if the current candle is a local bottom, False otherwise.
    """
    # Check if the current index is within the range of the data
    if curr_index < order * 2 + 1:
        return False

    # Initialize a flag to check if the current candle is a local bottom
    top = True

    # Calculate the index of the candle to start the rolling window
    k = curr_index - order

    # Get the price of the candle at index k
    v = data[k]

    # Iterate over the candles in the rolling window
    for i in range(1, order + 1):
        # If any of the candles in the window are lower than the K candle,
        # it is not a local bottom
        if data[k + i] < v or data[k - i] < v:
            top = False
            break

    # Return True if the K candle is a local bottom, False otherwise
    return top


#########################################################################################
########### ROLLING WINDOW USING PAST DATA FOR RESISTANCE & SUPPORT #####################
#########################################################################################

def rw_extremes(data: np.array, order: int):
    """
    This function identifies local tops and bottoms in a given price series using a rolling window approach.

    Parameters:
    data (np.array): A 1D numpy array containing the price series.
    order (int): The number of candles to consider for identifying local tops and bottoms.

    Returns:
    tops (list): A list of tuples, where each tuple represents a local top. Each tuple contains:
        - confirmation index: The index of the candle that confirms the local top.
        - index of top: The index of the local top.
        - price of top: The price of the local top.
    bottoms (list): A list of tuples, where each tuple represents a local bottom. Each tuple contains:
        - confirmation index: The index of the candle that confirms the local bottom.
        - index of bottom: The index of the local bottom.
        - price of bottom: The price of the local bottom.
    """
    # Rolling window local tops and bottoms
    tops = []
    bottoms = []
    for i in range(len(data)):
        if rw_top(data, i, order):
            # top[0] = confirmation index
            # top[1] = index of top
            # top[2] = price of top
            top = [i, i - order, float(data[i - order])]
            tops.append(top)

        if rw_bottom(data, i, order):
            # bottom[0] = confirmation index
            # bottom[1] = index of bottom
            # bottom[2] = price of bottom
            bottom = [i, i - order, float(data[i - order])]
            bottoms.append(bottom)

    return tops, bottoms

#########################################################################################
############################ DIRECTIONNAL CHANGE METHOD #################################
#########################################################################################
# see https://www.youtube.com/watch?v=X31hyMhB-3s for more information


def directional_change(close: np.array, high: np.array, low: np.array, sigma: float):
    """
    This function identifies local tops and bottoms in a given price series using a directional change method.

    Parameters:
    close (np.array): A 1D numpy array containing the closing prices of the series.
    high (np.array): A 1D numpy array containing the high prices of the series.
    low (np.array): A 1D numpy array containing the low prices of the series.
    sigma (float): A float value representing the percentage of price retracement to confirm a top or bottom.

    Returns:
    tops (list): A list of tuples, where each tuple represents a local top. Each tuple contains:
        - confirmation index: The index of the candle that confirms the local top.
        - index of top: The index of the local top.
        - price of top: The price of the local top.
    bottoms (list): A list of tuples, where each tuple represents a local bottom. Each tuple contains:
        - confirmation index: The index of the candle that confirms the local bottom.
        - index of bottom: The index of the local bottom.
        - price of bottom: The price of the local bottom.
    """

    up_zig = True  # Last extreme is a bottom. Next is a top.
    tmp_max = high[0]
    tmp_min = low[0]
    tmp_max_i = 0
    tmp_min_i = 0

    tops = []
    bottoms = []

    for i in range(len(close)):
        if up_zig:  # Last extreme is a bottom
            if high[i] > tmp_max:
                # New high, update
                tmp_max = high[i]
                tmp_max_i = i
            elif close[i] < tmp_max * (1 - sigma):
                # Price retraced by sigma %. Top confirmed, record it
                # top[0] = confirmation index
                # top[1] = index of top
                # top[2] = price of top
                top = [i, tmp_max_i, tmp_max]
                tops.append(top)

                # Setup for next bottom
                up_zig = False
                tmp_min = low[i]
                tmp_min_i = i
        else:  # Last extreme is a top
            if low[i] < tmp_min:
                # New low, update
                tmp_min = low[i]
                tmp_min_i = i
            elif close[i] > tmp_min * (1 + sigma):
                # Price retraced by sigma %. Bottom confirmed, record it
                # bottom[0] = confirmation index
                # bottom[1] = index of bottom
                # bottom[2] = price of bottom
                bottom = [i, tmp_min_i, tmp_min]
                bottoms.append(bottom)

                # Setup for next top
                up_zig = True
                tmp_max = high[i]
                tmp_max_i = i

    return tops, bottoms


def get_extremes(ohlc: pd.DataFrame, sigma: float) -> pd.DataFrame:
    """
    This function identifies local tops and bottoms in a given price series using a directional change method.

    Parameters:
    ohlc (pd.DataFrame): A pandas DataFrame containing the OHLC (Open, High, Low, Close) prices of the series.
        It should have columns named 'close', 'high', and 'low'.
    sigma (float): A float value representing the percentage of price retracement to confirm a top or bottom.

    Returns:
    pd.DataFrame: A pandas DataFrame containing the confirmed local extremes.
        It has columns 'conf_i' (confirmation index), 'ext_i' (index of extreme), 'ext_p' (price of extreme),
        and 'type' (1 for top, -1 for bottom). The DataFrame is indexed by 'conf_i' and sorted by index.
    """
    # Call directional_change function to get local tops and bottoms
    tops, bottoms = directional_change(
        ohlc['close'], ohlc['high'], ohlc['low'], sigma)

    # Convert lists to pandas DataFrames and add type column
    tops = pd.DataFrame(tops, columns=['conf_i', 'ext_i', 'ext_p'])
    bottoms = pd.DataFrame(bottoms, columns=['conf_i', 'ext_i', 'ext_p'])
    tops['type'] = 1
    bottoms['type'] = -1

    # Concatenate tops and bottoms DataFrames and sort by index
    extremes = pd.concat([tops, bottoms])
    extremes = extremes.set_index('conf_i')
    extremes = extremes.sort_index()

    return extremes

#########################################################################################
######################### PERCUPTUALLY IMPORTANT POINTS #################################
#########################################################################################
# see https://www.youtube.com/watch?v=X31hyMhB-3s for more information


def find_pips(data: np.array, n_pips: int, dist_measure: int) -> Tuple[List[int], List[float]]:
    """
    This function identifies Perceptually Important Points (PIPs) in a given price series.
    PIPs are points of significant price movement that are likely to influence future price movements.

    Parameters:
    data (np.array): A 1D numpy array containing the price series.
    n_pips (int): The number of PIPs to identify.
    dist_measure (int): The distance measure to use for identifying PIPs.
        - 1: Euclidean Distance
        - 2: Perpendicular Distance
        - 3: Vertical Distance

    Returns:
    Tuple[List[int], List[float]]: A tuple containing two lists.
        - The first list represents the indices of the identified PIPs.
        - The second list represents the prices of the identified PIPs.
    """

    # Initialize the lists to store the indices and prices of the PIPs
    pips_x = [0, len(data) - 1]  # Index
    pips_y = [data[0], data[-1]]  # Price

    # Iterate to identify the specified number of PIPs
    for curr_point in range(2, n_pips):

        md = 0.0  # Max distance
        md_i = -1  # Max distance index
        insert_index = -1

        # Iterate over the adjacent PIPs to find the PIP with the maximum distance
        for k in range(0, curr_point - 1):

            # Left adjacent, right adjacent indices
            left_adj = k
            right_adj = k + 1

            # Calculate the slope and intercept of the line connecting the adjacent PIPs
            time_diff = pips_x[right_adj] - pips_x[left_adj]
            price_diff = pips_y[right_adj] - pips_y[left_adj]
            slope = price_diff / time_diff
            intercept = pips_y[left_adj] - pips_x[left_adj] * slope

            # Iterate over the points between the adjacent PIPs to find the point with the maximum distance
            for i in range(pips_x[left_adj] + 1, pips_x[right_adj]):

                d = 0.0  # Distance
                # Calculate the distance based on the specified distance measure
                if dist_measure == 1:  # Euclidean distance
                    d = ((pips_x[left_adj] - i) ** 2 +
                         (pips_y[left_adj] - data[i]) ** 2) ** 0.5
                    d += ((pips_x[right_adj] - i) ** 2 +
                          (pips_y[right_adj] - data[i]) ** 2) ** 0.5
                elif dist_measure == 2:  # Perpendicular distance
                    d = abs((slope * i + intercept) -
                            data[i]) / (slope ** 2 + 1) ** 0.5
                else:  # Vertical distance
                    d = abs((slope * i + intercept) - data[i])

                # Update the maximum distance and its index if necessary
                if d > md:
                    md = d
                    md_i = i
                    insert_index = right_adj

        # Insert the PIP with the maximum distance into the list of PIPs
        pips_x.insert(insert_index, md_i)
        pips_y.insert(insert_index, data[md_i])

    # Return the lists of indices and prices of the identified PIPs
    return pips_x, pips_y
