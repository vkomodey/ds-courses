import pandas as pd

from sklearn import preprocessing


# TODO remove to_int
def fill_empty(series, to_int=False):
    # TODO change to median?
    mean = series.mean()
    if to_int:
        mean = round(mean)

    return series.fillna(mean, inplace=True)


def sanitize_credit_set(df):
    train_set = df.copy(deep=True)
    # gender. Видим, что пропусков нет. Значения пола равны 'M', 'F'. Перекодируем в числовые значения используя
    # LabelEncoder
    gender_encoder = preprocessing.LabelEncoder()
    train_set["gender"] = gender_encoder.fit_transform(train_set["gender"])

    # age.Всего три записи с незаполненными полями возраста. Заполним их средним по выборке значением
    train_set["age"].fillna(round(train_set["age"].mean()), inplace=True);

    # marital_status. В датасете видим незаполненные поля. Заполним данное поле самым встречаемым значением: `MAR` и
    # перекодируем используя LabelEncoder
    m_status = train_set["marital_status"]
    train_set["marital_status"].fillna(m_status.describe()["top"], inplace=True)

    marital_status_encoder = preprocessing.LabelEncoder()
    train_set["marital_status"] = marital_status_encoder.fit_transform(train_set["marital_status"])

    # job_position
    job_encoder = preprocessing.LabelEncoder()
    train_set["job_position"] = job_encoder.fit_transform(train_set["job_position"])

    # credit_sum Преобразуем строки к float и заполним NaN средним значением по выборке
    train_set["credit_sum"] = train_set["credit_sum"].str.replace(',', '.').astype('float')

    fill_empty(train_set["credit_sum"])

    # score_shk Возможен категориальный признак. Всего 31 значение

    train_set["score_shk"] = train_set["score_shk"].str.replace(',', '.').astype('float64')
    fill_empty(train_set["score_shk"])

    # education
    train_set["education"].fillna(train_set["education"].describe().top, inplace=True)
    education_encoder = preprocessing.LabelEncoder()
    train_set["education"] = education_encoder.fit_transform(train_set["education"])

    # living_region 192 пропущенных значения
    train_set["living_region"].fillna(train_set["living_region"].describe().top, inplace=True)
    location_encoder = preprocessing.LabelEncoder()
    train_set["living_region"] = education_encoder.fit_transform((train_set["living_region"]))

    # monthly_income Было бы неплохо взять среднюю зп из людей из той же специальности и локации. Ну или близко к
    # локации nan

    fill_empty(train_set["monthly_income"], True)

    # credit_count 9230 пропущенных значений То же самое что и с income. Надо взять всех похожих людей и засетать им
    # те же значения
    fill_empty(train_set["credit_count"], True)

    # Overdue Credit Count 9230 пропущенных значений
    # То же самое что и с income. Надо взять всех похожих людей и засетать им те же значения
    # Нельзя сетать среднее значение, так как overdue_credit_count может быть больше чем credit_count
    fill_empty(train_set["overdue_credit_count"], True)

    return train_set


train_set = pd.read_csv('./credit_train.csv', sep=';', index_col='client_id')

s_train_set = sanitize_credit_set(train_set)

s_train_set.to_csv('./credit_train_1.csv')
