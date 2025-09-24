import os
import requests
from langchain_core.tools import tool

from tools.utils import ADDRESSES
from tools.places_tools import get_address_long_lat

maps_key = os.getenv('GOOGLE_MAPS_KEY')

@tool
def get_current_weather(address_string: str = 'home_address') -> dict:
    """
    Get current weather conditions for a given address using Weather API.

    Args:
        maps_key (str): Google Maps API key.
        address_string (str): The address to look up, can be a full address, city or business name.

    Returns:    
        dict: Dictionary containing current weather conditions or error message
    """
    # Check if addresses are predefined names and get full addresses if they are
    if address_string in ADDRESSES:
        address_string = ADDRESSES[address_string]

    long_lat_dict = get_address_long_lat(address_string)

    if 'Error' in long_lat_dict:
        return {'Error':"Error: could not get response."}
    
    params = {'key':maps_key,
            'location.latitude':long_lat_dict['latitude'],
            'location.longitude':long_lat_dict['longitude']}
    response = requests.get("https://weather.googleapis.com/v1/currentConditions:lookup", params=params)
    if response.status_code == 200:
        response_json = response.json()

        keys_to_remove = ['timeZone', 'isDaytime', 'dewPoint', 'airPressure', 'heatIndex', 'windChill', 
                        'currentConditionsHistory', 'uvIndex']

        for key in keys_to_remove:
            response_json.pop(key, None)

        return response_json
    
    else:
        return {'Error':"Error: could not get response."}

@tool
def get_daily_weather(forecast_days: str, address_string: str = 'home_address') -> list:
    """
    Get daily weather forecast for a given address using Weather API.
    
    Args:
        forecast_days (str): Number of days to forecast ('0' (today) to '9').
        address_string (str): The address to look up, can be a full address, city or business name.
        
    Returns:
        list: List of dictionaries containing daily weather forecast or error message"""
    # Check if addresses are predefined names and get full addresses if they are
    if address_string in ADDRESSES:
        address_string = ADDRESSES[address_string]

    long_lat_dict = get_address_long_lat(address_string)

    if 'Error' in long_lat_dict:
        return {'Error':"Error: could not get response."}
    
    forecast_days = str(int(forecast_days) + 1)

    params = {'key':maps_key,
            'location.latitude':long_lat_dict['latitude'],
            'location.longitude':long_lat_dict['longitude'],
            'days':forecast_days,
            'pageSize':forecast_days}
    response = requests.get("https://weather.googleapis.com/v1/forecast/days:lookup", params=params)
    if response.status_code == 200:
        response_json = response.json()

        keys_to_remove = ['interval', 'nighttimeForecast', 'sunEvents', 'moonEvents', 'maxHeatIndex']

        for i in range(len(response_json['forecastDays'])):
            for key in keys_to_remove:
                response_json['forecastDays'][i].pop(key, None)

        response_json.pop('timeZone')
        return response_json['forecastDays']
    
    else:
        return ["Error: could not get response."]

@tool
def get_hourly_weather(address_string: str = 'home_address') -> list:
    """
    Get hourly weather forecast for a given address using Weather API.

    Args:
        address_string (str): The address to look up, can be a full address, city or business name.

    Returns:
        list: List of dictionaries containing hourly weather forecast or error message
    """
    # Check if addresses are predefined names and get full addresses if they are
    if address_string in ADDRESSES:
        address_string = ADDRESSES[address_string]

    long_lat_dict = get_address_long_lat(address_string)

    if 'Error' in long_lat_dict:
        return {'Error':"Error: could not get response."}
    
    params = {'key':maps_key,
            'location.latitude':long_lat_dict['latitude'],
            'location.longitude':long_lat_dict['longitude'],
            'hours':48,
            'pageSize':48}
    response = requests.get("https://weather.googleapis.com/v1/forecast/hours:lookup", params=params)

    if response.status_code == 200:
        response_json = response.json()
        final_response_list = []
        keys_to_keep = ['displayDateTime', 'weatherCondition', 'temperature', 'feelsLikeTemperature', 'precipitation', 'relativeHumidity']

        for forecast in response_json['forecastHours']:
            if forecast['isDaytime']:
                temp_dict = {}
                for key in forecast:
                    if key in keys_to_keep:
                        temp_dict[key] = forecast[key]
                final_response_list.append(temp_dict)

        return final_response_list
    
    else:
        return ["Error: could not get response."]
