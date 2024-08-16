import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import mplfinance as mpf
from flags_pennants import find_flags_pennants_pips, find_flags_pennants_trendline
from tqdm import tqdm

data = pd.read_csv("./Data/History/XAUUSD_M5_2023.csv")
data['Timestamp'] = pd.to_datetime(data['Date'].astype(str) + ' ' + data['Timestamp'], format='%Y%m%d %H:%M:%S')
data = data.drop(columns=['Date'])
data.rename(columns={'Timestamp': 'Date'}, inplace=True)
data.columns = data.columns.str.lower()
data['date'] = data['date'].astype('datetime64[s]')
data = data.set_index('date')


# data = pd.read_csv('./TechnicalAnalysisAutomation/BTCUSDT3600.csv')
# data['date'] = data['date'].astype('datetime64[s]')
# data = data.set_index('date')

# data = np.log(data)
dat_slice = data['close'].to_numpy()


orders = list(range(3, 49))
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
    bull_flags, bear_flags, bull_pennants, bear_pennants = find_flags_pennants_pips(dat_slice, order)
    # bull_flags, bear_flags, bull_pennants, bear_pennants  = find_flags_pennants_trendline(dat_slice, order)

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
        bull_flag_df.loc[i, 'slope'] = flag.resist_slope
        bull_flag_df.loc[i, 'support_slope'] = flag.support_slope
        bull_flag_df.loc[i, 'resist_slope'] = flag.resist_slope
        bull_flag_df.loc[i, 'support_intercept'] = flag.support_intercept
        bull_flag_df.loc[i, 'resist_intercept'] = flag.resist_intercept

        hp = int(flag.flag_width * hold_mult)
        if flag.conf_x + hp >= len(data):
            bull_flag_df.loc[i, 'return'] = np.nan
        else:
            # ret = dat_slice[flag.conf_x + hp] - dat_slice[flag.conf_x]
            ret = dat_slice[flag.conf_x+1:flag.conf_x + hp].max() - dat_slice[flag.conf_x]
            bull_flag_df.loc[i, 'return'] = ret

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
        if flag.conf_x + hp >= len(data):
            bear_flag_df.loc[i, 'return'] = np.nan
        else:
            ret = dat_slice[flag.conf_x] - dat_slice[flag.conf_x+1:flag.conf_x + hp].min()
            bear_flag_df.loc[i, 'return'] = ret

    for i, pennant in enumerate(bull_pennants):
        bull_pennant_df.loc[i, 'base_x'] = pennant.base_x
        bull_pennant_df.loc[i, 'pennant_width'] = pennant.flag_width
        bull_pennant_df.loc[i, 'pennant_height'] = pennant.flag_height
        bull_pennant_df.loc[i, 'pole_width'] = pennant.pole_width
        bull_pennant_df.loc[i, 'pole_height'] = pennant.pole_height
        bull_pennant_df.loc[i, 'support_slope'] = pennant.support_slope
        bull_pennant_df.loc[i, 'resist_slope'] = pennant.resist_slope
        bull_pennant_df.loc[i, 'support_intercept'] = pennant.support_intercept
        bull_pennant_df.loc[i, 'resist_intercept'] = pennant.resist_intercept

        hp = int(pennant.flag_width * hold_mult)
        if pennant.conf_x + hp >= len(data):
            bull_pennant_df.loc[i, 'return'] = np.nan
        else:
            ret = dat_slice[pennant.conf_x+1:pennant.conf_x + hp].max() - dat_slice[pennant.conf_x]
            if (order == 5):
                print(dat_slice[pennant.conf_x+1:pennant.conf_x + hp].max(), dat_slice[pennant.conf_x])
                print(ret)
            bull_pennant_df.loc[i, 'return'] = ret

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
        if pennant.conf_x + hp >= len(data):
            bear_pennant_df.loc[i, 'return'] = np.nan
        else:
            ret = dat_slice[pennant.conf_x] - dat_slice[pennant.conf_x+1:pennant.conf_x + hp].min()
            bear_pennant_df.loc[i, 'return'] = ret

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


def print_results(results_df):
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


print_results(results_df)


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
