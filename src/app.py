from sensorlib import *
from util import *
import config as cfg
import time
from datetime import datetime


def run_weather_station():
    while True:
        si7021_humidity, si7021_temperature = get_humidity_and_temperature()
        bmp280_pressure, bmp280_temperature = get_pressure_and_temperature()

        temp_avg = (si7021_temperature + bmp280_temperature) / 2

        station_log(f"Hum: {si7021_humidity}% Temp1: {si7021_temperature}°C, Temp2: {bmp280_temperature}°C, Ps: {bmp280_pressure}hPa")

        data = {
            "station_id": cfg.station_id,
            "time": datetime.now().isoformat(),
            "temperature": temp_avg,
            "humidity": si7021_humidity,
            "barometric_pressure": bmp280_pressure,
            "lat": cfg.station_lat,
            "lon": cfg.station_lon,
            "elevation": cfg.station_elevation
        }

        error = upload_sensor_data(data)

        if error:
            station_log(f"Error uploading data: {error}")

        time.sleep(cfg.send_interval)

if __name__ == "__main__":
    run_weather_station()