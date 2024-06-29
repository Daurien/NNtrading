import numpy as np
import pandas as pd



#########################################################################################
############# ROLLING WINDOW USING FUTURE AND PAST DATA #################################
#########################################################################################

# identify pivot points with a rolling window of n1 on the left and n2 on the right
def pivotid(dataframe, l, n1, n2):

    if l-n1 < 0 or l+n2 >= len(dataframe):
        return 0

    window = dataframe.iloc[l-n1:l+n2+1]
    pivotlow = (dataframe.loc[l, 'Low'] <= window['Low']).all()
    pivothigh = (dataframe.loc[l, 'High'] >= window['High']).all()

    if pivotlow and pivothigh:
        return 3
    elif pivotlow:
        return 1
    elif pivothigh:
        return 2
    else:
        return 0

#########################################################################################
############################ DIRECTIONNAL CHANGE METHOD #################################
#########################################################################################
# see https://www.youtube.com/watch?v=X31hyMhB-3s for more information


def directional_change(close: np.array, high: np.array, low: np.array, sigma: float):

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
            elif close[i] < tmp_max - tmp_max * sigma:
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
            elif close[i] > tmp_min + tmp_min * sigma:
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


def get_extremes(ohlc: pd.DataFrame, sigma: float):
    tops, bottoms = directional_change(
        ohlc['close'], ohlc['high'], ohlc['low'], sigma)
    tops = pd.DataFrame(tops, columns=['conf_i', 'ext_i', 'ext_p'])
    bottoms = pd.DataFrame(bottoms, columns=['conf_i', 'ext_i', 'ext_p'])
    tops['type'] = 1
    bottoms['type'] = -1
    extremes = pd.concat([tops, bottoms])
    extremes = extremes.set_index('conf_i')
    extremes = extremes.sort_index()
    return extremes

#########################################################################################
######################### PERCUPTUALLY IMPORTANT POINTS #################################
#########################################################################################
# see https://www.youtube.com/watch?v=X31hyMhB-3s for more information

def find_pips(data: np.array, n_pips: int, dist_measure: int):
    # dist_measure
    # 1 = Euclidean Distance
    # 2 = Perpindicular Distance
    # 3 = Vertical Distance

    pips_x = [0, len(data) - 1]  # Index
    pips_y = [data[0], data[-1]] # Price

    for curr_point in range(2, n_pips):

        md = 0.0 # Max distance
        md_i = -1 # Max distance index
        insert_index = -1

        for k in range(0, curr_point - 1):

            # Left adjacent, right adjacent indices
            left_adj = k
            right_adj = k + 1

            time_diff = pips_x[right_adj] - pips_x[left_adj]
            price_diff = pips_y[right_adj] - pips_y[left_adj]
            slope = price_diff / time_diff
            intercept = pips_y[left_adj] - pips_x[left_adj] * slope;

            for i in range(pips_x[left_adj] + 1, pips_x[right_adj]):
                
                d = 0.0 # Distance
                if dist_measure == 1: # Euclidean distance
                    d =  ( (pips_x[left_adj] - i) ** 2 + (pips_y[left_adj] - data[i]) ** 2 ) ** 0.5
                    d += ( (pips_x[right_adj] - i) ** 2 + (pips_y[right_adj] - data[i]) ** 2 ) ** 0.5
                elif dist_measure == 2: # Perpindicular distance
                    d = abs( (slope * i + intercept) - data[i] ) / (slope ** 2 + 1) ** 0.5
                else: # Vertical distance    
                    d = abs( (slope * i + intercept) - data[i] )

                if d > md:
                    md = d
                    md_i = i
                    insert_index = right_adj

        pips_x.insert(insert_index, md_i)
        pips_y.insert(insert_index, data[md_i])

    return pips_x, pips_y

