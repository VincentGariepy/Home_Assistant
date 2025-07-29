import json
import requests
import os
import math

class PlacesTools():
    home_address = os.getenv('GOOGLE_API_HOME_ADDRESS')

    @classmethod
    def get_adress_place_id(cls, address_string: str, maps_key: str) -> str:
        response = requests.get(
            "https://maps.googleapis.com/maps/api/geocode/json", 
            params={'address':address_string, 
            'key':maps_key}
            )
        
        if response.status_code == 200:
            return response.json()['results'][0]['place_id']
        else:
            return "Error: could not get response."

    @classmethod
    def get_address_long_lat(cls, address_string: str, maps_key: str) -> dict:
        response = requests.get(
            "https://maps.googleapis.com/maps/api/geocode/json", 
            params={'address':address_string, 
            'key':maps_key}
            )
        
        if response.status_code == 200:
            return response.json()['results'][0]['navigation_points'][0]['location']
        else:
            return {'Error':"Error: could not get response."}

    @classmethod
    def get_route(cls, destination_address: str, maps_key: str, origin_address: str = home_address) -> dict:
        origin_id = cls.get_adress_place_id(origin_address, maps_key)
        destination_id = cls.get_adress_place_id(destination_address, maps_key)
        
        url = 'https://routes.googleapis.com/directions/v2:computeRoutes'

        headers = {
            'Content-Type': 'application/json',
            'X-Goog-Api-Key': maps_key,
            'X-Goog-FieldMask': 'routes.duration,routes.staticDuration,routes.distanceMeters,routes.legs.steps.navigationInstruction'
        }

        payload = {
            "origin":{
                "placeId": origin_id
            },
            "destination":{
                "placeId": destination_id
            },
            "travelMode": "DRIVE",
            "routingPreference": "TRAFFIC_AWARE",
            "computeAlternativeRoutes": False,
            "routeModifiers": {
                "avoidTolls": False,
                "avoidHighways": False,
                "avoidFerries": False
            },
            "languageCode": "en-CA",
            "units": "METRIC"
        }

        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            response_json = response.json()
            return {'route_steps':response_json['routes'][0]['legs'][0]['steps'], 
                    'time_with_traffic':math.ceil(int(response_json['routes'][0]['duration'][:-1])/60),
                    'time_without_traffic':math.ceil(int(response_json['routes'][0]['staticDuration'][:-1])/60),
                    'distnce':math.ceil(int(response_json['routes'][0]['distanceMeters'])/1000)}
        
        else:
            return {'Error':"Error: could not get response."}
