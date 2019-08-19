from sklearn_pandas import CategoricalImputer
import pandas as pd


# Функция которая добавляет новую колонку в датасет, обозначающую вероятность получения кредита в зависимости от column
def populate_with_mean(frame, column):
    frame['{}'.format(column)] = frame[column].map(
        frame.groupby(column).mean()['open_account_flg'].to_dict())


def sanitize_frame(_frame):
    frame = _frame.copy(deep=True)
    frame = format_and_fillna(frame)
    frame = one_hot_encode(frame)
    frame = handle_living_region(frame)
    frame = add_mean_columns(frame)
    frame = add_credit_history_ratio(frame)
    frame = normalize_frame(frame)
    frame = drop_columns(frame)

    return frame


def format_and_fillna(frame):
    if frame["credit_sum"].dtype == 'object':
        frame["credit_sum"] = frame["credit_sum"].str.replace(',', '.')

    frame["credit_sum"] = frame["credit_sum"].astype('float64')

    if frame["score_shk"].dtype == 'object':
        frame["score_shk"] = frame["score_shk"].str.replace(',', '.')

    frame["score_shk"] = frame["score_shk"].astype('float64')

    frame = fill_empty(frame)

    frame["age"] = frame["age"].astype('int')

    return frame


def add_credit_history_ratio(frame):
    active_creditor = frame[frame['credit_count'] > 0]['credit_count']
    frame['credit_history_ratio'] = (active_creditor - frame['overdue_credit_count']) / (active_creditor)

    # Людей с хорошей кредитной историей будем считать положительными(см график в visualise блокноте)
    frame['credit_history_ratio'].fillna(1, inplace=True)

    return frame


def normalize(series):
    return (series - series.mean()) / series.std()


def normalize_frame(frame):
    frame["credit_sum"] = normalize(frame["credit_sum"])
    frame["monthly_income"] = normalize(frame["monthly_income"])
    frame["tariff_id"] = normalize(frame["tariff_id"])
    frame["score_shk"] = normalize(frame["score_shk"])

    return frame


def fill_empty(frame):
    imputer = CategoricalImputer()

    return frame.apply(lambda x: imputer.fit_transform(x), axis=0)


def one_hot_encode(frame):
    frame = add_dummies(frame, 'job_position')
    frame = add_dummies(frame, 'marital_status')
    frame = add_dummies(frame, 'education')

    return frame


def add_dummies(frame, column):
    dfDummies = pd.get_dummies(frame[column], prefix='_')

    return frame.join(dfDummies)


# Заменяем соответствующий столбец на вероятности получения кредита для каждого категориального значения
def add_mean_columns(frame):
    populate_with_mean(frame, 'job_position')
    populate_with_mean(frame, 'education')
    populate_with_mean(frame, 'age')
    populate_with_mean(frame, 'tariff_id')
    populate_with_mean(frame, 'credit_count')
    populate_with_mean(frame, 'overdue_credit_count')
    populate_with_mean(frame, 'living_region')
    populate_with_mean(frame, 'marital_status')
    populate_with_mean(frame, 'credit_month')

    return frame


# Избавимся от gender потому что как мы выяснили из visualise блокнота, этот параметр не особо влияет на результат
# и от credit_count так как у него довольно сильная корреляция с credit_history_ratio
def drop_columns(frame):
    return frame.drop(columns=['gender', 'credit_count'])


# sanitized_regions.csv - отнормированные(с использованием yandex geocoder, см sanitize_regions.py)
# значения локаций с широтой и долготой каждого места.
# + в frame добавляется отдельный столбец индицирующий расстояние до точки с широтой и долготой 0.0, 0.0
def handle_living_region(frame):
    sanitized_regions = pd.read_csv('./sanitized_regions.csv', index_col='original')
    regions_to_dist_map = dict(zip(sanitized_regions.index, sanitized_regions['dist']))
    regions_to_sanitized_map = dict(zip(sanitized_regions.index, sanitized_regions['sanitized']))

    # Заменяем колонку living_region на расстояние до 0.0, 0.0
    frame['living_region_mean'] = frame['living_region'].map(regions_to_dist_map)
    frame['living_region'] = frame['living_region'].map(regions_to_sanitized_map)

    return frame
