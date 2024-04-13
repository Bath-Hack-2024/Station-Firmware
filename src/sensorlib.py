import board
import adafruit_si7021
import adafruit_bmp280
from time import sleep
import os
from typing import Tuple, List
from datetime import datetime
import config as cfg
import requests

def get_humidity_and_temperature(address=0x40):
    I2C = board.I2C()
    temp_humidity_sensor = adafruit_si7021.SI7021(I2C, address=address)

    return temp_humidity_sensor.relative_humidity, temp_humidity_sensor.temperature

def get_pressure_and_temperature(address=0x76):
    I2C = board.I2C()
    pressure_sensor = adafruit_bmp280.Adafruit_BMP280_I2C(I2C, address=address)

    return pressure_sensor.pressure, pressure_sensor.temperature

def take_picture(filepath, width=2592, height=1944, gain=15, shutter=3000):
    try:
        time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file = f"{filepath}/still_{time_stamp}.jpg"
        cli_command = f"rpicam-jpeg -o {file} --width {width} --height {width}"

        # run command and watch exit status
        exit_status = os.system(cli_command)

        if exit_status != 0:
            return None

        return file
    
    except Exception as e:
        return None

def upload_picture(file, img_upload_url):
    try:
        files = {'file': open(file, 'rb')}
        response = requests.post(img_upload_url, files=files)
    
        if response.status_code != 200:
            return None
    
        #parse response
        return response.json()["file_path"]
    
    except Exception as e:
        return None
    
    


