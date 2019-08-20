import pandas as pd
import numpy as np
import requests as r


# Данные о проживании достаточно грязные - с ошибками и нерепрезентативные
# Хотелось бы по возможности представить каждую локацию одним числом + что бы для близких локаций числа тоже были близки

def construct_url(living_region):
    living_region = living_region.replace(' ', '%20')
    api_key = 'your key here. that will not work'

    return 'https://geocode-maps.yandex.ru/1.x/?apikey={}&geocode={}&format=json'.format(api_key, living_region)


living_regions = pd.read_csv('./unique_regions.csv', squeeze=True).values

empty_strings = ['' for i in range(len(living_regions))]
living_regions_frame = pd.DataFrame({
    'sanitized': empty_strings,
    'longitude': np.zeros(len(living_regions)),
    'latitude': np.zeros(len(living_regions))
}, index=living_regions)

# Яндекс отказался давать ответы на следующие 4 локации:
# горьковская область(Нижегородская Область)
# эвенкинский ао(Эвенкийский Район)
# 98(Санкт-Петербург)
# 74(Челябинская область)
for living_region in living_regions:
    url = construct_url(living_region)
    resp = r.get(url).json()

    found_results = int(
        resp['response']['GeoObjectCollection']['metaDataProperty']['GeocoderResponseMetaData']['found'])

    sanitized_value = resp['response']['GeoObjectCollection']['featureMember'][0]['GeoObject']['name']
    point = resp['response']['GeoObjectCollection']['featureMember'][0]['GeoObject']['Point']['pos'].split(' ')
    longitude = float(point[0])
    latitude = float(point[1])

    living_regions_frame.set_value(living_region, 'sanitized', sanitized_value)
    living_regions_frame.set_value(living_region, 'longitude', longitude)
    living_regions_frame.set_value(living_region, 'latitude', latitude)


# В колонке dist указывается новый признак - расстояние до фиксированной точки с широтой и долготой 0.0,
# 0.0 Это позволит в будущем единобразно интерпретировать признак living_region числом. Плюс, интуитивно хочется
# видеть близкие места с похожими коэффициентами(например, Москва и Московская область), из-за этого я отказался от
# обычного label encoder
living_regions_frame['dist'] = np.sqrt(living_regions_frame['longitude']**2 + living_regions_frame['latitude']**2)
living_regions_frame.to_csv('./sanitized_regions.csv')