import json
import requests
import os
import math

from langchain_core.tools import tool

from tools.utils import ADDRESSES


# Google Maps API key from environment variable
maps_key = os.getenv('GOOGLE_MAPS_KEY')

def get_address_place_id(address_string: str) -> str:
    """
    Get Google Places ID for a given address using Geocoding API.
    
    Args:
        address_string (str): The address to look up, can be a full address, city or business name.
        
    Returns:
        str: Google Places ID or error message if request fails
    """
    response = requests.get(
        "https://maps.googleapis.com/maps/api/geocode/json", 
        params={'address':address_string, 
        'key':maps_key}
        )

    if response.status_code == 200:
        return response.json()['results'][0]['place_id']
    else:
        return "Error: could not get response."

def get_address_long_lat(address_string: str) -> dict:
    """
    Get latitude and longitude coordinates for a given address using Geocoding API.
    
    Args:
        address_string (str): The address to look up, can be a full address, city or business name.
        
    Returns:
        dict: Dictionary containing location coordinates or error message
    """
    response = requests.get(
        "https://maps.googleapis.com/maps/api/geocode/json", 
        params={'address':address_string, 
        'key':maps_key}
        )
    
    print(response.json())

    if response.status_code == 200:
        response_json = response.json()['results'][0]

        if 'navigation_points' in response_json:
            return response_json['navigation_points'][0]['location']
        else:
            return {
                'latitude':response_json['geometry']['location']['lat'],
                'longitude':response_json['geometry']['location']['lng']
            }
    else:
        return {'Error':"Error: could not get response."}

@tool
def get_route(destination_address: str, origin_address: str) -> dict:
    """
    Calculate route details between two addresses using Google Routes API.
    
    Args:
        destination_address (str): Address, city, business or predefined location name for destination
        origin_address (str): Address, city, business or predefined location name for starting point
        
    Returns:
        dict: Route information including steps, travel times, and distance
    """
    # Check if addresses are predefined names and get full addresses if they are
    if destination_address in ADDRESSES:
        destination_address = ADDRESSES[destination_address]
    
    if origin_address in ADDRESSES:
        origin_address = ADDRESSES[origin_address]

    # Get Place IDs for both addresses
    origin_id = get_address_place_id(origin_address)
    destination_id = get_address_place_id(destination_address)
    
    url = 'https://routes.googleapis.com/directions/v2:computeRoutes'

    # Define request headers including API key and required fields
    headers = {
        'Content-Type': 'application/json',
        'X-Goog-Api-Key': maps_key,
        'X-Goog-FieldMask': 'routes.duration,routes.staticDuration,routes.distanceMeters,routes.legs.steps.navigationInstruction'
    }

    # Configure routing parameters
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
    # Make API request and process response
    response = requests.post(url, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        response_json = response.json()
        return {'route_steps':response_json['routes'][0]['legs'][0]['steps'], 
                'time_with_traffic':math.ceil(int(response_json['routes'][0]['duration'][:-1])/60),
                'time_without_traffic':math.ceil(int(response_json['routes'][0]['staticDuration'][:-1])/60),
                'distnce':math.ceil(int(response_json['routes'][0]['distanceMeters'])/1000)}
    
    else:
        return {'Error':"Error: could not get response."}