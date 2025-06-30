import os
import math


def flatten(nested_list):
    flat_list = []
    for element in nested_list:
        if isinstance(element, list) or isinstance(element, tuple):
            flat_list.extend(flatten(element))
        else:
            flat_list.append(element)
    return flat_list


def get_absolute_path():
    return os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def haversine(lon1, lat1, lon2, lat2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    return math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2


def normalized_position(lat, lon, bbox):
    normalized_lon = (lon - bbox[3]) / (bbox[2] - bbox[3])
    normalized_lat = (lat - bbox[1]) / (bbox[0] - bbox[1])
    return normalized_lon, normalized_lat


def haversine(lon1, lat1, lon2, lat2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return a

