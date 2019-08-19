import pandas as pd
import numpy as np
import requests as r


def construct_url(living_region):
    living_region = living_region.replace(' ', '%20')
    apiKey = '9c81adb9-0cfb-49a3-85b1-389698164430'

    return 'https://geocode-maps.yandex.ru/1.x/?apikey={}&geocode={}&format=json'.format(apiKey, living_region)


living_regions = pd.read_csv('./unique_regions.csv', squeeze=True).values

empty_strings = ['' for i in range(len(living_regions))]
living_regions_frame = pd.DataFrame({
    'sanitized': empty_strings,
    'longitude': np.zeros(len(living_regions)),
    'latitude': np.zeros(len(living_regions))
}, index=living_regions)

# Яндекс отказался давать ответы на следующие 4 локации:
# горьковской областью(нижегородская область)
# эвенкинским ао(эвенкийский район)
# 98(Санкт-Петербург)
# 74(Челябинская область)
for living_region in living_regions:
    url = construct_url(living_region)
    resp = r.get(url).json()

    found_results = int(
        resp['response']['GeoObjectCollection']['metaDataProperty']['GeocoderResponseMetaData']['found'])

    if found_results == 0:
        continue

    sanitized_value = resp['response']['GeoObjectCollection']['featureMember'][0]['GeoObject']['name']
    point = resp['response']['GeoObjectCollection']['featureMember'][0]['GeoObject']['Point']['pos'].split(' ')
    longitude = float(point[0])
    latitude = float(point[1])

    living_regions_frame.set_value(living_region, 'sanitized', sanitized_value)
    living_regions_frame.set_value(living_region, 'longitude', longitude)
    living_regions_frame.set_value(living_region, 'latitude', latitude)

living_regions_frame.to_csv('./sanitized_regions.csv')