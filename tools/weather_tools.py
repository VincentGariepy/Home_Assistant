import os
import requests

from places_tools import PlacesTools

class WeatherTools():
    home_address = os.getenv('GOOGLE_API_HOME_ADDRESS')

    @classmethod
    def get_current_weather(cls, maps_key: str, address_string: str = home_address) -> dict:
        long_lat_dict = PlacesTools.get_address_long_lat(address_string, maps_key)

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

    @classmethod
    def get_daily_weather(cls, maps_key: str, forecast_days: str, address_string: str = home_address) -> list:
        long_lat_dict = PlacesTools.get_address_long_lat(address_string, maps_key)

        if 'Error' in long_lat_dict:
            return {'Error':"Error: could not get response."}
        
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

    @classmethod
    def get_hourly_weather(cls, maps_key: str, address_string: str = home_address) -> list:
        long_lat_dict = PlacesTools.get_address_long_lat(address_string, maps_key)

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
