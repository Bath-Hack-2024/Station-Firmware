import board
import adafruit_si7021
import adafruit_bmp280
from time import sleep
import os
from typing import Tuple, List

def get_humidity_and_temperature(address=0x40):
    I2C = board.I2C()
    temp_humidity_sensor = adafruit_si7021.SI7021(I2C, address=address)

    return temp_humidity_sensor.relative_humidity, temp_humidity_sensor.temperature

def get_pressure_and_temperature(address=0x76):
    I2C = board.I2C()
    pressure_sensor = adafruit_bmp280.Adafruit_BMP280_I2C(I2C, address=address)

    return pressure_sensor.pressure, pressure_sensor.temperature

