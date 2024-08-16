import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sup_res_id import find_pips
from sup_res_id import rw_top, rw_bottom
from trendline_automation import fit_trendlines_single
from enum import Enum
from typing import List, Tuple
from tqdm import tqdm
from .flagpattern import FlagPattern


class MethodType(Enum):
    PIPS = "perceptually_important_points"
    TRENDLINE = "trendline"


def check_bear_pattern_pips(pending: FlagPattern, data: np.array, i: int, order: int, order_ratio: float = 0.5, flag_pole_width_ratio: float = 0.5, flag_pole_height_ratio: float = 0.5):
    """
    Check if the given data slice forms a bearish flag/pennant pattern.

    Parameters:
    pending (FlagPattern): The pending pattern to be checked.
    data (np.array): The price data.
    i (int): The current index in the data.
    order (int): The order of the pattern.
    order_ratio (float): The ratio of order to be considered for pattern confirmation.
    flag_pole_width_ratio (float): The ratio of flag/pennant width to pole width to be considered for pattern confirmation.
    flag_pole_height_ratio (float): The ratio of flag/pennant height to pole height to be considered for pattern confirmation.

    Returns:
    bool: True if the data slice forms a bearish flag/pennant pattern, False otherwise.
    """

    # Find max price since local bottom, (top of pole)
    data_slice = data[pending.base_x: i + 1]  # i + 1 includes current price
    min_i = data_slice.argmin() + pending.base_x  # Min index since local top

    if i - min_i < max(5, order * order_ratio):  # Far enough from max to draw potential flag/pennant
        return False

    # Test flag width / height
    pole_width = min_i - pending.base_x
    flag_width = i - min_i
    if flag_width > pole_width * flag_pole_width_ratio:  # Flag should be less than half the width of pole
        return False

    pole_height = pending.base_y - data[min_i]
    flag_height = data[min_i:i+1].max() - data[min_i]
    if flag_height > pole_height * flag_pole_height_ratio:  # Flag should smaller vertically than preceding trend
        return False

    # If here width/height are OK.

    # Find perceptually important points from pole to current time
    pips_x, pips_y = find_pips(data[min_i:i+1], 5, 3)  # Finds pips between max and current index (inclusive)

    # Check center pip is less than two adjacent. /\/\
    if not (pips_y[2] < pips_y[1] and pips_y[2] < pips_y[3]):
        return False

    # Find slope and intercept of flag lines
    # intercept is at the max value (top of pole)
    support_rise = pips_y[2] - pips_y[0]
    support_run = pips_x[2] - pips_x[0]
    support_slope = support_rise / support_run
    support_intercept = pips_y[0]

    resist_rise = pips_y[3] - pips_y[1]
    resist_run = pips_x[3] - pips_x[1]
    resist_slope = resist_rise / resist_run
    resist_intercept = pips_y[1] + (pips_x[0] - pips_x[1]) * resist_slope

    # Find x where two lines intersect.
    if resist_slope != support_slope:  # Not parallel
        intersection = (support_intercept - resist_intercept) / (resist_slope - support_slope)
    else:
        intersection = -flag_width * 100

    # No intersection in flag area
    if intersection <= pips_x[4] and intersection >= 0:
        return False

    # Check if current point has a breakout of flag. (confirmation)
    support_endpoint = pips_y[0] + support_slope * pips_x[4]
    if pips_y[4] > support_endpoint:
        return False

    if resist_slope < 0:
        pending.pennant = True
    else:
        pending.pennant = False

    # Filter harshly diverging lines
    if intersection < 0 and intersection > -flag_width:
        return False

    pending.tip_x = min_i
    pending.tip_y = data[min_i]
    pending.conf_x = i
    pending.conf_y = data[i]
    pending.flag_width = flag_width
    pending.flag_height = flag_height
    pending.pole_width = pole_width
    pending.pole_height = pole_height
    pending.support_slope = support_slope
    pending.support_intercept = support_intercept
    pending.resist_slope = resist_slope
    pending.resist_intercept = resist_intercept

    return True


def check_bull_pattern_pips(pending: FlagPattern, data: np.array, i: int, order: int, order_ratio: float = 0.5, flag_pole_width_ratio: float = 0.5, flag_pole_height_ratio: float = 0.5):
    """
    Check if the given data slice forms a bullish flag/pennant pattern.

    Parameters:
    pending (FlagPattern): The pending pattern to be checked.
    data (np.array): The price data.
    i (int): The current index in the data.
    order (int): The order of the pattern.
    order_ratio (float): The ratio of order to be considered for pattern confirmation.
    flag_pole_width_ratio (float): The ratio of flag/pennant width to pole width to be considered for pattern confirmation.
    flag_pole_height_ratio (float): The ratio of flag/pennant height to pole height to be considered for pattern confirmation.

    Returns:
    bool: True if the data slice forms a bullish flag/pennant pattern, False otherwise.
    """

    # Find max price since local bottom, (top of pole)
    data_slice = data[pending.base_x: i + 1]  # i + 1 includes current price
    max_i = data_slice.argmax() + pending.base_x  # Max index since bottom
    pole_width = max_i - pending.base_x

    if i - max_i < max(5, order * order_ratio):  # Far enough from max to draw potential flag/pennant
        return False

    flag_width = i - max_i
    if flag_width > pole_width * flag_pole_width_ratio:  # Flag should be less than half the width of pole
        return False

    pole_height = data[max_i] - pending.base_y
    flag_height = data[max_i] - data[max_i:i+1].min()
    if flag_height > pole_height * flag_pole_height_ratio:  # Flag should smaller vertically than preceding trend
        return False

    pips_x, pips_y = find_pips(data[max_i:i+1], 5, 3)  # Finds PIPS between max and current index (inclusive)

    # Check center pip is greater than two adjacent. \/\/
    if not (pips_y[2] > pips_y[1] and pips_y[2] > pips_y[3]):
        return False

    # Find slope and intercept of flag lines
    # intercept is at the max value (top of pole)
    resist_rise = pips_y[2] - pips_y[0]
    resist_run = pips_x[2] - pips_x[0]
    resist_slope = resist_rise / resist_run
    resist_intercept = pips_y[0]

    support_rise = pips_y[3] - pips_y[1]
    support_run = pips_x[3] - pips_x[1]
    support_slope = support_rise / support_run
    support_intercept = pips_y[1] + (pips_x[0] - pips_x[1]) * support_slope

    # Find x where two lines intersect.
    if resist_slope != support_slope:  # Not parallel
        intersection = (support_intercept - resist_intercept) / (resist_slope - support_slope)
    else:
        intersection = -flag_width * 100

    # No intersection in flag area
    if intersection <= pips_x[4] and intersection >= 0:
        return False

    # Filter harshly diverging lines
    if intersection < 0 and intersection > -1.0 * flag_width:
        return False

    # Check if current point has a breakout of flag. (confirmation)
    resist_endpoint = pips_y[0] + resist_slope * pips_x[4]
    if pips_y[4] < resist_endpoint:
        return False

    # Pattern is confiremd, fill out pattern details in pending
    if support_slope > 0:
        pending.pennant = True
    else:
        pending.pennant = False

    pending.tip_x = max_i
    pending.tip_y = data[max_i]
    pending.conf_x = i
    pending.conf_y = data[i]
    pending.flag_width = flag_width
    pending.flag_height = flag_height
    pending.pole_width = pole_width
    pending.pole_height = pole_height

    pending.support_slope = support_slope
    pending.support_intercept = support_intercept
    pending.resist_slope = resist_slope
    pending.resist_intercept = resist_intercept

    return True


def find_flags_pennants_pips(data: np.array, order: int) -> Tuple[List[FlagPattern], List[FlagPattern], List[FlagPattern], List[FlagPattern]]:
    """
    This function finds flag and pennant patterns in the given price data using PIPs method.

    Parameters:
    data (np.array): The price data array.
    order (int): The order of the rolling window used to identify local extrema.

    Returns:
    Tuple[List[FlagPattern], List[FlagPattern], List[FlagPattern], List[FlagPattern]]:
    - A list of confirmed bull flag patterns.
    - A list of confirmed bear flag patterns.
    - A list of confirmed bull pennant patterns.
    - A list of confirmed bear pennant patterns.
    """
    assert (order >= 3)
    pending_bull = None  # Pending pattern
    pending_bear = None  # Pending pattern

    bull_pennants = []
    bear_pennants = []
    bull_flags = []
    bear_flags = []
    for i in range(len(data)):

        # Pattern data is organized like so:
        if rw_top(data, i, order):
            pending_bear = FlagPattern(i - order, data[i - order])

        if rw_bottom(data, i, order):
            pending_bull = FlagPattern(i - order, data[i - order])

        if pending_bear is not None:
            if check_bear_pattern_pips(pending_bear, data, i, order):
                if pending_bear.pennant:
                    bear_pennants.append(pending_bear)
                else:
                    bear_flags.append(pending_bear)
                pending_bear = None

        if pending_bull is not None:
            if check_bull_pattern_pips(pending_bull, data, i, order):
                if pending_bull.pennant:
                    bull_pennants.append(pending_bull)
                else:
                    bull_flags.append(pending_bull)
                pending_bull = None

    return bull_flags, bear_flags, bull_pennants, bear_pennants


def check_bull_pattern_trendline(pending: FlagPattern, data: np.array, i: int, order: int):

    # Check if data max less than pole tip
    if data[pending.tip_x + 1: i].max() > pending.tip_y:
        return False

    flag_min = data[pending.tip_x:i].min()

    # Find flag/pole height and width
    pole_height = pending.tip_y - pending.base_y
    pole_width = pending.tip_x - pending.base_x

    flag_height = pending.tip_y - flag_min
    flag_width = i - pending.tip_x

    if flag_width > pole_width * 0.5:  # Flag should be less than half the width of pole
        return False

    if flag_height > pole_height * 0.75:  # Flag should smaller vertically than preceding trend
        return False

    # Find trendlines going from flag tip to the previous bar (not including current bar)
    support_coefs, resist_coefs = fit_trendlines_single(data[pending.tip_x:i])
    support_slope, support_intercept = support_coefs[0], support_coefs[1]
    resist_slope, resist_intercept = resist_coefs[0], resist_coefs[1]

    # Check for breakout of upper trendline to confirm pattern
    current_resist = resist_intercept + resist_slope * (flag_width + 1)
    if data[i] <= current_resist:
        return False

    # Pattern is confiremd, fill out pattern details in pending
    if support_slope > 0:
        pending.pennant = True
    else:
        pending.pennant = False

    pending.conf_x = i
    pending.conf_y = data[i]
    pending.flag_width = flag_width
    pending.flag_height = flag_height
    pending.pole_width = pole_width
    pending.pole_height = pole_height

    pending.support_slope = support_slope
    pending.support_intercept = support_intercept
    pending.resist_slope = resist_slope
    pending.resist_intercept = resist_intercept

    return True


def check_bear_pattern_trendline(pending: FlagPattern, data: np.array, i: int, order: int):
    # Check if data max less than pole tip
    if data[pending.tip_x + 1: i].min() < pending.tip_y:
        return False

    flag_max = data[pending.tip_x:i].max()

    # Find flag/pole height and width
    pole_height = pending.base_y - pending.tip_y
    pole_width = pending.tip_x - pending.base_x

    flag_height = flag_max - pending.tip_y
    flag_width = i - pending.tip_x

    if flag_width > pole_width * 0.5:  # Flag should be less than half the width of pole
        return False

    if flag_height > pole_height * 0.75:  # Flag should smaller vertically than preceding trend
        return False

    # Find trendlines going from flag tip to the previous bar (not including current bar)
    support_coefs, resist_coefs = fit_trendlines_single(data[pending.tip_x:i])
    support_slope, support_intercept = support_coefs[0], support_coefs[1]
    resist_slope, resist_intercept = resist_coefs[0], resist_coefs[1]

    # Check for breakout of lower trendline to confirm pattern
    current_support = support_intercept + support_slope * (flag_width + 1)
    if data[i] >= current_support:
        return False

    # Pattern is confiremd, fill out pattern details in pending
    if resist_slope < 0:
        pending.pennant = True
    else:
        pending.pennant = False

    pending.conf_x = i
    pending.conf_y = data[i]
    pending.flag_width = flag_width
    pending.flag_height = flag_height
    pending.pole_width = pole_width
    pending.pole_height = pole_height

    pending.support_slope = support_slope
    pending.support_intercept = support_intercept
    pending.resist_slope = resist_slope
    pending.resist_intercept = resist_intercept

    return True


def find_flags_pennants_trendline(data: np.array, order: int) -> Tuple[List[FlagPattern], List[FlagPattern], List[FlagPattern], List[FlagPattern]]:
    """
    This function finds flag and pennant patterns in the given price data using trendline method.

    Parameters:
    data (np.array): The price data array.
    order (int): The order of the rolling window used to identify local extrema.

    Returns:
    Tuple[List[FlagPattern], List[FlagPattern], List[FlagPattern], List[FlagPattern]]:
    - A list of confirmed bull flag patterns.
    - A list of confirmed bear flag patterns.
    - A list of confirmed bull pennant patterns.
    - A list of confirmed bear pennant patterns.
    """
    assert (order >= 3)
    pending_bull = None  # Pending pattern
    pending_bear = None  # Pending pattern

    last_bottom = -1
    last_top = -1

    bull_pennants = []
    bear_pennants = []
    bull_flags = []
    bear_flags = []
    for i in range(len(data)):

        # Pattern data is organized like so:
        if rw_top(data, i, order):
            last_top = i - order
            if last_bottom != -1:
                pending = FlagPattern(last_bottom, data[last_bottom])
                pending.tip_x = last_top
                pending.tip_y = data[last_top]
                pending_bull = pending

        if rw_bottom(data, i, order):
            last_bottom = i - order
            if last_top != -1:
                pending = FlagPattern(last_top, data[last_top])
                pending.tip_x = last_bottom
                pending.tip_y = data[last_bottom]
                pending_bear = pending

        if pending_bear is not None:
            if check_bear_pattern_trendline(pending_bear, data, i, order):
                if pending_bear.pennant:
                    bear_pennants.append(pending_bear)
                else:
                    bear_flags.append(pending_bear)
                pending_bear = None

        if pending_bull is not None:
            if check_bull_pattern_trendline(pending_bull, data, i, order):
                if pending_bull.pennant:
                    bull_pennants.append(pending_bull)
                else:
                    bull_flags.append(pending_bull)
                pending_bull = None

    return bull_flags, bear_flags, bull_pennants, bear_pennants


def runAnalysis(dat_slice: np.ndarray, type: MethodType = MethodType.PIPS, orders: List = list(range(3, 49)), print_results: bool = True, save_results: bool = True, savePath='./Data/results/') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    assert (type == MethodType.PIPS or type == MethodType.TRENDLINE), "Make sure MethodType is either PIPS or TRENDLINE"

    bull_flag_wr = []
    bull_pennant_wr = []
    bear_flag_wr = []
    bear_pennant_wr = []

    bull_flag_avg = []
    bull_pennant_avg = []
    bear_flag_avg = []
    bear_pennant_avg = []

    bull_flag_count = []
    bull_pennant_count = []
    bear_flag_count = []
    bear_pennant_count = []

    bull_flag_total_ret = []
    bull_pennant_total_ret = []
    bear_flag_total_ret = []
    bear_pennant_total_ret = []

    bear_pennant_df_full = []
    bull_pennant_df_full = []
    bear_flag_df_full = []
    bull_flag_df_full = []

    for order in tqdm(orders):
        if type == MethodType.PIPS:
            bull_flags, bear_flags, bull_pennants, bear_pennants = find_flags_pennants_pips(dat_slice, order)
        else:
            bull_flags, bear_flags, bull_pennants, bear_pennants = find_flags_pennants_trendline(dat_slice, order)

        bull_flag_df = pd.DataFrame()
        bull_pennant_df = pd.DataFrame()
        bear_flag_df = pd.DataFrame()
        bear_pennant_df = pd.DataFrame()

        # Assemble data into dataframe
        hold_mult = 2.0  # Multipler of flag width to hold for after a pattern
        for i, flag in enumerate(bull_flags):
            bull_flag_df.loc[i, 'base_x'] = flag.base_x
            bull_flag_df.loc[i, 'flag_width'] = flag.flag_width
            bull_flag_df.loc[i, 'flag_height'] = flag.flag_height
            bull_flag_df.loc[i, 'pole_width'] = flag.pole_width
            bull_flag_df.loc[i, 'pole_height'] = flag.pole_height
            bull_flag_df.loc[i, 'support_slope'] = flag.support_slope
            bull_flag_df.loc[i, 'resist_slope'] = flag.resist_slope
            bull_flag_df.loc[i, 'support_intercept'] = flag.support_intercept
            bull_flag_df.loc[i, 'resist_intercept'] = flag.resist_intercept

            hp = int(flag.flag_width * hold_mult)
            if flag.conf_x + hp >= len(dat_slice):
                bull_flag_df.loc[i, 'return_min'] = np.nan
                bull_flag_df.loc[i, 'return'] = np.nan
                bull_flag_df.loc[i, 'return_mean'] = np.nan
            else:
                # ret = dat_slice[flag.conf_x + hp] - dat_slice[flag.conf_x]
                id_max = dat_slice[flag.conf_x+1:flag.conf_x + hp].argmax()
                ret_max = dat_slice[flag.conf_x+1:flag.conf_x + hp].max() - dat_slice[flag.conf_x]
                ret_min = dat_slice[flag.conf_x+1:flag.conf_x + hp].min() - dat_slice[flag.conf_x]
                ret_mean = (dat_slice[flag.conf_x+1:flag.conf_x + hp] - dat_slice[flag.conf_x]).mean()
                bull_flag_df.loc[i, 'return'] = ret_max
                bull_flag_df.loc[i, 'return_min'] = ret_min
                bull_flag_df.loc[i, 'return_mean'] = ret_mean
                bull_flag_df.loc[i, 'id_max'] = id_max

        for i, flag in enumerate(bear_flags):
            bear_flag_df.loc[i, 'base_x'] = flag.base_x
            bear_flag_df.loc[i, 'flag_width'] = flag.flag_width
            bear_flag_df.loc[i, 'flag_height'] = flag.flag_height
            bear_flag_df.loc[i, 'pole_width'] = flag.pole_width
            bear_flag_df.loc[i, 'pole_height'] = flag.pole_height
            bear_flag_df.loc[i, 'support_slope'] = flag.support_slope
            bear_flag_df.loc[i, 'resist_slope'] = flag.resist_slope
            bear_flag_df.loc[i, 'support_intercept'] = flag.support_intercept
            bear_flag_df.loc[i, 'resist_intercept'] = flag.resist_intercept

            hp = int(flag.flag_width * hold_mult)
            if flag.conf_x + hp >= len(dat_slice):
                bear_flag_df.loc[i, 'return_min'] = np.nan
                bear_flag_df.loc[i, 'return'] = np.nan
                bear_flag_df.loc[i, 'return_mean'] = np.nan
            else:
                id_max = dat_slice[flag.conf_x+1:flag.conf_x + hp].argmin()
                ret_max = dat_slice[flag.conf_x] - dat_slice[flag.conf_x+1:flag.conf_x + hp].min()
                ret_min = dat_slice[flag.conf_x] - dat_slice[flag.conf_x+1:flag.conf_x + hp].max()
                ret_mean = (dat_slice[flag.conf_x] - dat_slice[flag.conf_x+1:flag.conf_x + hp]).mean()
                bear_flag_df.loc[i, 'return'] = ret_max
                bear_flag_df.loc[i, 'return_min'] = ret_min
                bear_flag_df.loc[i, 'return_mean'] = ret_mean
                bear_flag_df.loc[i, 'id_max'] = id_max

        for i, pennant in enumerate(bull_pennants):
            bull_pennant_df.loc[i, 'base_x'] = pennant.base_x
            bull_pennant_df.loc[i, 'pennant_width'] = pennant.flag_width
            bull_pennant_df.loc[i, 'pennant_height'] = pennant.flag_height
            bull_pennant_df.loc[i, 'pole_width'] = pennant.pole_width
            bull_pennant_df.loc[i, 'pole_height'] = pennant.pole_height
            bull_pennant_df.loc[i, 'pole_height'] = pennant.resist_slope
            bull_pennant_df.loc[i, 'support_slope'] = pennant.support_slope
            bull_pennant_df.loc[i, 'resist_slope'] = pennant.resist_slope
            bull_pennant_df.loc[i, 'support_intercept'] = pennant.support_intercept
            bull_pennant_df.loc[i, 'resist_intercept'] = pennant.resist_intercept

            hp = int(pennant.flag_width * hold_mult)
            if pennant.conf_x + hp >= len(dat_slice):
                bull_pennant_df.loc[i, 'return_min'] = np.nan
                bull_pennant_df.loc[i, 'return'] = np.nan
                bull_pennant_df.loc[i, 'return_mean'] = np.nan
            else:
                id_max = dat_slice[pennant.conf_x+1:pennant.conf_x + hp].argmax()
                ret_max = dat_slice[pennant.conf_x+1:pennant.conf_x + hp].max() - dat_slice[pennant.conf_x]
                ret_min = dat_slice[pennant.conf_x+1:pennant.conf_x + hp].min() - dat_slice[pennant.conf_x]
                ret_mean = (dat_slice[pennant.conf_x+1:pennant.conf_x + hp] - dat_slice[pennant.conf_x]).mean()
                bull_pennant_df.loc[i, 'return'] = ret_max
                bull_pennant_df.loc[i, 'return_min'] = ret_min
                bull_pennant_df.loc[i, 'return_mean'] = ret_mean
                bull_pennant_df.loc[i, 'id_max'] = id_max

        for i, pennant in enumerate(bear_pennants):
            bear_pennant_df.loc[i, 'base_x'] = pennant.base_x
            bear_pennant_df.loc[i, 'pennant_width'] = pennant.flag_width
            bear_pennant_df.loc[i, 'pennant_height'] = pennant.flag_height
            bear_pennant_df.loc[i, 'pole_width'] = pennant.pole_width
            bear_pennant_df.loc[i, 'pole_height'] = pennant.pole_height
            bear_pennant_df.loc[i, 'support_slope'] = pennant.support_slope
            bear_pennant_df.loc[i, 'resist_slope'] = pennant.resist_slope
            bear_pennant_df.loc[i, 'support_intercept'] = pennant.support_intercept
            bear_pennant_df.loc[i, 'resist_intercept'] = pennant.resist_intercept

            hp = int(pennant.flag_width * hold_mult)
            if pennant.conf_x + hp >= len(dat_slice):
                bear_pennant_df.loc[i, 'return_min'] = np.nan
                bear_pennant_df.loc[i, 'return'] = np.nan
                bear_pennant_df.loc[i, 'return_mean'] = np.nan
            else:
                id_max = dat_slice[pennant.conf_x+1:pennant.conf_x + hp].argmin()
                ret_max = dat_slice[pennant.conf_x] - dat_slice[pennant.conf_x+1:pennant.conf_x + hp].min()
                ret_min = dat_slice[pennant.conf_x] - dat_slice[pennant.conf_x+1:pennant.conf_x + hp].max()
                ret_mean = (dat_slice[pennant.conf_x] - dat_slice[pennant.conf_x+1:pennant.conf_x + hp]).mean()
                bear_pennant_df.loc[i, 'return'] = ret_max
                bear_pennant_df.loc[i, 'return_min'] = ret_min
                bear_pennant_df.loc[i, 'return_mean'] = ret_mean
                bear_pennant_df.loc[i, 'id_max'] = id_max

        if len(bull_flag_df) > 0:
            bull_flag_count.append(len(bull_flag_df))
            bull_flag_avg.append(bull_flag_df['return'].mean())
            bull_flag_wr.append(len(bull_flag_df[bull_flag_df['return'] > 0]) / len(bull_flag_df))
            bull_flag_total_ret.append(bull_flag_df['return'].sum())
        else:
            bull_flag_count.append(0)
            bull_flag_avg.append(np.nan)
            bull_flag_wr.append(np.nan)
            bull_flag_total_ret.append(0)

        if len(bear_flag_df) > 0:
            bear_flag_count.append(len(bear_flag_df))
            bear_flag_avg.append(bear_flag_df['return'].mean())
            bear_flag_wr.append(len(bear_flag_df[bear_flag_df['return'] > 0]) / len(bear_flag_df))
            bear_flag_total_ret.append(bear_flag_df['return'].sum())
        else:
            bear_flag_count.append(0)
            bear_flag_avg.append(np.nan)
            bear_flag_wr.append(np.nan)
            bear_flag_total_ret.append(0)

        if len(bull_pennant_df) > 0:
            bull_pennant_count.append(len(bull_pennant_df))
            bull_pennant_avg.append(bull_pennant_df['return'].mean())
            bull_pennant_wr.append(len(bull_pennant_df[bull_pennant_df['return'] > 0]) / len(bull_pennant_df))
            bull_pennant_total_ret.append(bull_pennant_df['return'].sum())
        else:
            bull_pennant_count.append(0)
            bull_pennant_avg.append(np.nan)
            bull_pennant_wr.append(np.nan)
            bull_pennant_total_ret.append(0)

        if len(bear_pennant_df) > 0:
            bear_pennant_count.append(len(bear_pennant_df))
            bear_pennant_avg.append(bear_pennant_df['return'].mean())
            bear_pennant_wr.append(len(bear_pennant_df[bear_pennant_df['return'] > 0]) / len(bear_pennant_df))
            bear_pennant_total_ret.append(bear_pennant_df['return'].sum())
        else:
            bear_pennant_count.append(0)
            bear_pennant_avg.append(np.nan)
            bear_pennant_wr.append(np.nan)
            bear_pennant_total_ret.append(0)

        bear_pennant_df_full.append((bear_pennant_df, order))
        bull_pennant_df_full.append((bull_pennant_df, order))
        bear_flag_df_full.append((bear_flag_df, order))
        bull_flag_df_full.append((bull_flag_df, order))

    results_df = pd.DataFrame(index=orders)
    results_df['bull_flag_count'] = bull_flag_count
    results_df['bull_flag_avg'] = bull_flag_avg
    results_df['bull_flag_wr'] = bull_flag_wr
    results_df['bull_flag_total'] = bull_flag_total_ret

    results_df['bear_flag_count'] = bear_flag_count
    results_df['bear_flag_avg'] = bear_flag_avg
    results_df['bear_flag_wr'] = bear_flag_wr
    results_df['bear_flag_total'] = bear_flag_total_ret

    results_df['bull_pennant_count'] = bull_pennant_count
    results_df['bull_pennant_avg'] = bull_pennant_avg
    results_df['bull_pennant_wr'] = bull_pennant_wr
    results_df['bull_pennant_total'] = bull_pennant_total_ret

    results_df['bear_pennant_count'] = bear_pennant_count
    results_df['bear_pennant_avg'] = bear_pennant_avg
    results_df['bear_pennant_wr'] = bear_pennant_wr
    results_df['bear_pennant_total'] = bear_pennant_total_ret

    if print_results:
        # Plot bull flag results
        plt.style.use('dark_background')
        fig, ax = plt.subplots(2, 2)
        fig.suptitle("Bull Flag Performance", fontsize=20)
        results_df['bull_flag_count'].plot.bar(ax=ax[0, 0])
        results_df['bull_flag_avg'].plot.bar(ax=ax[0, 1], color='yellow')
        results_df['bull_flag_total'].plot.bar(ax=ax[1, 0], color='green')
        results_df['bull_flag_wr'].plot.bar(ax=ax[1, 1], color='orange')
        ax[0, 1].hlines(0.0, xmin=-1, xmax=len(orders), color='white')
        ax[1, 0].hlines(0.0, xmin=-1, xmax=len(orders), color='white')
        ax[1, 1].hlines(0.5, xmin=-1, xmax=len(orders), color='white')
        ax[0, 0].set_title('Number of Patterns Found')
        ax[0, 0].set_xlabel('Order Parameter')
        ax[0, 0].set_ylabel('Number of Patterns')
        ax[0, 1].set_title('Average Pattern Return')
        ax[0, 1].set_xlabel('Order Parameter')
        ax[0, 1].set_ylabel('Average Log Return')
        ax[1, 0].set_title('Sum of Returns')
        ax[1, 0].set_xlabel('Order Parameter')
        ax[1, 0].set_ylabel('Total Log Return')
        ax[1, 1].set_title('Win Rate')
        ax[1, 1].set_xlabel('Order Parameter')
        ax[1, 1].set_ylabel('Win Rate Percentage')

        plt.show()

        fig, ax = plt.subplots(2, 2)
        fig.suptitle("Bear Flag Performance", fontsize=20)
        results_df['bear_flag_count'].plot.bar(ax=ax[0, 0])
        results_df['bear_flag_avg'].plot.bar(ax=ax[0, 1], color='yellow')
        results_df['bear_flag_total'].plot.bar(ax=ax[1, 0], color='green')
        results_df['bear_flag_wr'].plot.bar(ax=ax[1, 1], color='orange')
        ax[0, 1].hlines(0.0, xmin=-1, xmax=len(orders), color='white')
        ax[1, 0].hlines(0.0, xmin=-1, xmax=len(orders), color='white')
        ax[1, 1].hlines(0.5, xmin=-1, xmax=len(orders), color='white')
        ax[0, 0].set_title('Number of Patterns Found')
        ax[0, 0].set_xlabel('Order Parameter')
        ax[0, 0].set_ylabel('Number of Patterns')
        ax[0, 1].set_title('Average Pattern Return')
        ax[0, 1].set_xlabel('Order Parameter')
        ax[0, 1].set_ylabel('Average Log Return')
        ax[1, 0].set_title('Sum of Returns')
        ax[1, 0].set_xlabel('Order Parameter')
        ax[1, 0].set_ylabel('Total Log Return')
        ax[1, 1].set_title('Win Rate')
        ax[1, 1].set_xlabel('Order Parameter')
        ax[1, 1].set_ylabel('Win Rate Percentage')
        plt.show()

        fig, ax = plt.subplots(2, 2)
        fig.suptitle("Bull Pennant Performance", fontsize=20)
        results_df['bull_pennant_count'].plot.bar(ax=ax[0, 0])
        results_df['bull_pennant_avg'].plot.bar(ax=ax[0, 1], color='yellow')
        results_df['bull_pennant_total'].plot.bar(ax=ax[1, 0], color='green')
        results_df['bull_pennant_wr'].plot.bar(ax=ax[1, 1], color='orange')
        ax[0, 1].hlines(0.0, xmin=-1, xmax=len(orders), color='white')
        ax[1, 0].hlines(0.0, xmin=-1, xmax=len(orders), color='white')
        ax[1, 1].hlines(0.5, xmin=-1, xmax=len(orders), color='white')
        ax[0, 0].set_title('Number of Patterns Found')
        ax[0, 0].set_xlabel('Order Parameter')
        ax[0, 0].set_ylabel('Number of Patterns')
        ax[0, 1].set_title('Average Pattern Return')
        ax[0, 1].set_xlabel('Order Parameter')
        ax[0, 1].set_ylabel('Average Log Return')
        ax[1, 0].set_title('Sum of Returns')
        ax[1, 0].set_xlabel('Order Parameter')
        ax[1, 0].set_ylabel('Total Log Return')
        ax[1, 1].set_title('Win Rate')
        ax[1, 1].set_xlabel('Order Parameter')
        ax[1, 1].set_ylabel('Win Rate Percentage')
        plt.show()

        fig, ax = plt.subplots(2, 2)
        fig.suptitle("Bear Pennant Performance", fontsize=20)
        results_df['bear_pennant_count'].plot.bar(ax=ax[0, 0])
        results_df['bear_pennant_avg'].plot.bar(ax=ax[0, 1], color='yellow')
        results_df['bear_pennant_total'].plot.bar(ax=ax[1, 0], color='green')
        results_df['bear_pennant_wr'].plot.bar(ax=ax[1, 1], color='orange')
        ax[0, 1].hlines(0.0, xmin=-1, xmax=len(orders), color='white')
        ax[1, 0].hlines(0.0, xmin=-1, xmax=len(orders), color='white')
        ax[1, 1].hlines(0.5, xmin=-1, xmax=len(orders), color='white')
        ax[0, 0].set_title('Number of Patterns Found')
        ax[0, 0].set_xlabel('Order Parameter')
        ax[0, 0].set_ylabel('Number of Patterns')
        ax[0, 1].set_title('Average Pattern Return')
        ax[0, 1].set_xlabel('Order Parameter')
        ax[0, 1].set_ylabel('Average Log Return')
        ax[1, 0].set_title('Sum of Returns')
        ax[1, 0].set_xlabel('Order Parameter')
        ax[1, 0].set_ylabel('Total Log Return')
        ax[1, 1].set_title('Win Rate')
        ax[1, 1].set_xlabel('Order Parameter')
        ax[1, 1].set_ylabel('Win Rate Percentage')
        plt.show()

    if save_results:
        concat_bear_flag_df_full = pd.concat(
            [df.assign(id=id)[['id'] + df.columns.tolist()] for df, id in bear_flag_df_full],
            ignore_index=True
        )

        concat_bull_flag_df_full = pd.concat(
            [df.assign(id=id)[['id'] + df.columns.tolist()] for df, id in bull_flag_df_full],
            ignore_index=True
        )

        concat_bear_pennant_df_full = pd.concat(
            [df.assign(id=id)[['id'] + df.columns.tolist()] for df, id in bear_pennant_df_full],
            ignore_index=True
        )

        concat_bull_pennant_df_full = pd.concat(
            [df.assign(id=id)[['id'] + df.columns.tolist()] for df, id in bull_pennant_df_full],
            ignore_index=True
        )

        results_df.to_csv('./Data/results/results_df.csv')
        concat_bear_flag_df_full.to_csv('./Data/results/bear_flag_df.csv')
        concat_bear_pennant_df_full.to_csv('./Data/results/bear_pennant_df.csv')
        concat_bull_pennant_df_full.to_csv('./Data/results/bull_pennant_df.csv')
        concat_bull_flag_df_full.to_csv('./Data/results/bull_flag_df.csv')

    return results_df, concat_bear_flag_df_full, concat_bull_flag_df_full, concat_bear_pennant_df_full, concat_bull_pennant_df_full
