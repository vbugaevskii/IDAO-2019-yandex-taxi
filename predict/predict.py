import datetime
import numpy as np
import pickle
import ast
import sys
import catboost
import pandas as pd
from collections import defaultdict

MODELS_FILE = 'models.pkl'
AERO_FILE = 'MSK-SVO.csv.agg'
OUTPUT_HEADER = 'datetime,target_{},target_{},target_{},target_{},target_{}'


def load_aero_schedule(filename):
    def minutes_gen():
        s_curr = datetime.datetime(year=2019, month=4, day=5)
        s_stop = datetime.datetime(year=2019, month=4, day=6)
        s_step = datetime.timedelta(minutes=1)

        while s_curr < s_stop:
            yield s_curr
            s_curr = s_curr + s_step
     
    index = [datetime.datetime.strftime(m, '%H:%M') for m in minutes_gen()]
    df_ts = pd.DataFrame(index, columns=['time'])
    
    df_aero = pd.read_csv(filename)
    df_aero = pd.merge(df_ts, df_aero, how='left', on='time')
    df_aero['flight'] = df_aero['flight'].fillna(0).astype(int)
    df_aero = df_aero.set_index(keys=['time'])
    return df_aero

def choose_flights_between(aero, dt_from, dt_till):
    dt_from = datetime.datetime.strftime(dt_from, '%H:%M')
    dt_till = datetime.datetime.strftime(dt_till, '%H:%M')
    
    if dt_from <= dt_till:
        return aero.loc[dt_from:dt_till, 'flight'].values
    else:
        return np.hstack((
            aero.loc[dt_from:, 'flight'].values,
            aero.loc[:dt_till, 'flight'].values
        ))
    
    
def count_std_sum(history, shift, duration):
    return np.std(np.sum(history.reshape(-1, shift)[:, -duration:], axis = 1))


HOUR_IN_MINUTES = 60
DAY_IN_MINUTES = 24 * HOUR_IN_MINUTES
WEEK_IN_MINUTES = 7 * DAY_IN_MINUTES

SHIFTS = [
    HOUR_IN_MINUTES // 4,
    HOUR_IN_MINUTES // 2,
    HOUR_IN_MINUTES,
    DAY_IN_MINUTES,
    DAY_IN_MINUTES * 2,
    WEEK_IN_MINUTES,
    WEEK_IN_MINUTES * 2
]

WINDOWS = [
    HOUR_IN_MINUTES // 4,
    HOUR_IN_MINUTES // 2,
    HOUR_IN_MINUTES,
    DAY_IN_MINUTES,
    DAY_IN_MINUTES * 2,
    WEEK_IN_MINUTES,
    WEEK_IN_MINUTES * 2
]

# should be modified for every track
set_name = 'set1'

target_positions = {
    'set1': [10, 30, 45, 60, 75],
    'set2': [5, 10, 15, 20, 25],
    'set3': [5, 7, 9, 11, 13]
}[set_name]


def compute_target(history, prefix, count_first=True):
    values = {}
    
    cumsum_num_orders = history[:2 * DAY_IN_MINUTES].cumsum()
    
    variants = target_positions
    if count_first:
        variants = [1] + variants
    
    for position in variants:
        orders_by_positions = np.where(cumsum_num_orders >= position)[0]
        if len(orders_by_positions):
            time = orders_by_positions[0] + 1
        else:
            time = MAX_TIME
        values['{}_{}'.format(prefix, position)] = time

    return values


def extra_features_history_1(history):
    return compute_target(history[::-1], prefix="order_last", count_first=True)

def extra_features_history_2(history):
    return compute_target(history[WEEK_IN_MINUTES:], prefix="order_middle_1week", count_first=False)

def extra_features_history_3(history):
    return compute_target(history, prefix="order_middle_2week", count_first=False)

def extra_features_history_4(history):
    return compute_target(history[-DAY_IN_MINUTES:], prefix="order_middle_1day", count_first=False)

def extra_features_history_6(history):
    return compute_target(history[-3 * DAY_IN_MINUTES:], prefix="order_middle_3day", count_first=False)


def max_empty_window(history, prefix):
    res = np.nonzero(history)[0]
    res = res[1:] - res[:-1]
    return {
        "{}_min".format(prefix) : res.min() if len(res) else np.nan,
        "{}_max".format(prefix) : res.max() if len(res) else np.nan,
        "{}_med".format(prefix) : np.median(res) if len(res) else np.nan,
        "{}_num".format(prefix) : res.shape[0],
    }

def max_empty_stats_1(history):
    return max_empty_window(history[-HOUR_IN_MINUTES:], "max_empty_stats_1hour")

def max_empty_stats_2(history):
    return max_empty_window(history[-2 * HOUR_IN_MINUTES:], "max_empty_stats_2hour")

def max_empty_stats_3(history):
    return max_empty_window(history[-HOUR_IN_MINUTES // 2:], "max_empty_stats_30min")

def max_empty_stats_4(history):
    return max_empty_window(history[-WEEK_IN_MINUTES-HOUR_IN_MINUTES:
                                    -WEEK_IN_MINUTES+HOUR_IN_MINUTES], "max_empty_stats_middle_2hour")


holidays = {
    '01-01': {'holiday', 'dayoff'},
    '01-02': {'dayoff'},
    '01-03': {'dayoff'},
    '01-04': {'dayoff'},
    '01-05': {'dayoff'},
    '01-06': {'dayoff'},
    '01-07': {'holiday', 'dayoff'},
    '02-23': {'holiday', 'dayoff'},
    '02-24': {'dayoff'},
    '03-08': {'holiday', 'dayoff'},
    '05-01': {'holiday', 'dayoff'},
    '05-08': {'dayoff'},
    '05-09': {'holiday', 'dayoff'},
    '11-06': {'dayoff'},
    '12-31': {'holiday', 'dayoff'}
}

holidays_good_arrive = {
    '01-01': {'holiday', 'dayoff'},
    '01-02': {'dayoff'},
    '02-22': {'holiday', 'dayoff'},
    '02-23': {'holiday', 'dayoff'},
    '03-07': {'holiday', 'dayoff'},
    '03-08': {'holiday', 'dayoff'},
    '05-01': {'holiday', 'dayoff'},
    '04-31': {'holiday', 'dayoff'},
    '05-02': {'holiday', 'dayoff'},
    '05-08': {'dayoff'},
    '05-09': {'holiday', 'dayoff'},
    '11-06': {'dayoff'},
    '12-31': {'holiday', 'dayoff'}
}


def extractor(dt, history, **params):    
    features = []
    
    """
    aero = params['aero']
    aero_c = choose_flights_between(
        aero,
        dt - datetime.timedelta(minutes=30),
        dt + datetime.timedelta(minutes=30)
    ).sum()
    features.append(aero_c)
    """

    features.append(dt.weekday())
    features.append(dt.hour)
    features.append(dt.minute)

    for shift in SHIFTS:
        for window in WINDOWS:
            if window >= shift:
                continue
            features.append(sum(history[-shift:-shift + window]))
    
    history = np.asarray(history)
    
    features_add = extra_features_history_1(history)
    for k in sorted(features_add.keys(), key=lambda x: int(x.split('_')[-1])):
        features.append(features_add[k])
    features_order_last = features_add
        
    features_add = extra_features_history_2(history)
    for k in sorted(features_add.keys(), key=lambda x: int(x.split('_')[-1])):
        features.append(features_add[k])
        
    features_add = extra_features_history_3(history)
    for k in sorted(features_add.keys(), key=lambda x: int(x.split('_')[-1])):
        features.append(features_add[k])
        
    features_add = extra_features_history_4(history)
    for k in sorted(features_add.keys(), key=lambda x: int(x.split('_')[-1])):
        features.append(features_add[k])
    
    # """
    features_add = extra_features_history_6(history)
    for k in sorted(features_add.keys(), key=lambda x: int(x.split('_')[-1])):
        features.append(features_add[k])
        
    features_add = max_empty_stats_1(history)
    for k, v in features_add.items():
        features.append(v)
        
    features_add = max_empty_stats_2(history)
    for k, v in features_add.items():
        features.append(v)
        
    features_add = max_empty_stats_3(history)
    for k, v in features_add.items():
        features.append(v)
        
    features_add = max_empty_stats_4(history)
    for k, v in features_add.items():
        features.append(v)
    # """

    features.append('{:%m-%d}'.format(dt) in holidays)
    features.append('{:%m-%d}'.format(dt) in holidays_good_arrive)
    
    for p in target_positions:
        features.append(count_std_sum(history, 60 * 24, features_order_last['order_last_'+str(p)]))
    
    return np.array(features)


if __name__ == '__main__':
    models = pickle.load(open(MODELS_FILE, 'rb'))
    # aero = load_aero_schedule(AERO_FILE)
    aero = None

    input_header = input()
    output_header = OUTPUT_HEADER.format(*sorted(list(models['models'].keys())))
    print(output_header)
    
    all_features = []
    all_queries = []
    while True:
        # read data, calculate features line by line for memory efficient
        try:
            raw_line = input()
        except EOFError:
            break
                    
        line = raw_line.split(',', 1)
        dt = datetime.datetime.strptime(line[0], '%Y-%m-%d %H:%M:%S')
        history = list(map(int, line[1][2:-2].split(', ')))
        features = extractor(dt, history, **{'aero': aero})
        
        all_features.append(features)
        all_queries.append(line[0])
    
    # predict all objects for time efficient
    predictions = []
    for position, model in models['models'].items():
        y_pred = np.mean([m.predict(all_features) for m in model], axis=0)
        predictions.append(y_pred)
    
    for i in range(len(predictions[0])):
        print(','.join([all_queries[i]] + list(map(lambda x: str(x[i]), predictions))))
