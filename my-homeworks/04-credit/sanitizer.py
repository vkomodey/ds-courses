from sklearn_pandas import CategoricalImputer
import pandas as pd


def sanitize_frame(df):
    sanitized_frame = df.copy(deep=True)

    if sanitized_frame["credit_sum"].dtype == 'object':
        sanitized_frame["credit_sum"] = sanitized_frame["credit_sum"].str.replace(',', '.')

    sanitized_frame["credit_sum"] = sanitized_frame["credit_sum"].astype('float64')

    if sanitized_frame["score_shk"].dtype == 'object':
        sanitized_frame["score_shk"] = sanitized_frame["score_shk"].str.replace(',', '.')

    sanitized_frame["score_shk"] = sanitized_frame["score_shk"].astype('float64')

    imputer = CategoricalImputer()

    sanitized_frame = sanitized_frame.apply(lambda x: imputer.fit_transform(x), axis=0)

    sanitized_frame["age"] = sanitized_frame["age"].astype('int')

    return sanitized_frame
