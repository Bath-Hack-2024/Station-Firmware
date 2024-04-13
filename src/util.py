from datetime import datetime
import requests
import config as cfg

def station_log(msg):
    print(f"[{datetime.now()}] {msg}")

def upload_sensor_data(data: dict):

    try:
        response = requests.post(cfg.upload_url, json=data)
        
        if response.status_code != 200:
            return response.text
        
    except Exception as e:
        station_log(f"Error uploading data: {e}")
        return e

    return None
    
