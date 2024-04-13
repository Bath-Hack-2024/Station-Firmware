import board
import adafruit_si7021
import adafruit_bmp280
from picamera import PiCamera
from time import sleep
import os

def getTemperatureAndHumidity():
    """
    Gets the temperature and humidity from the SI7021 sensor

    Returns:
        tuple: tuple of temperature, relative humidity
    """
    I2C = board.I2C()
    #Temperature and humidity here
    tempHumiditySensor = adafruit_si7021.SI7021(I2C)
    return tempHumiditySensor.temperature, tempHumiditySensor.relative_humidity

def getPressureAndTemperature():
    """
    Gets the pressure and temperature from the BMP280 sensor

    Returns:
        tuple: tuple of temperature and pressure
    """
    I2C = board.I2C()
    #pressure and temperature here
    pressureSensor = adafruit_bmp280.Adafruit_BMP280_I2C(I2C)
    return pressureSensor.temperature, pressureSensor.pressure


def get_next_image_number(directory: str):
    """
    Gets the next image number in a directory

    Args:
        directory (str): the directory to write the file to

    Raises:
        FileNotFoundError: If the file isn't found, raise an Error

    Returns:
        int: highest next available image number, or 1 if nothing exists yet
    """
    try:
        f = open(directory)
    except:
        raise FileNotFoundError("directory "+directory+" not found or inaccessible")
    
    # Get a list of files in the directory
    files = os.listdir(directory)

    # Filter out files that are not images
    image_files = [file for file in files if file.endswith(".jpg")]

    # If no image files exist, start from 1
    if not image_files:
        return 1

    # Get the highest image number
    highest_number = max([int(file.split(".")[0].split("image")[1]) for file in image_files])

    # Increment the highest number to get the next available number
    return highest_number + 1


def getImage(filepath: str):
    """
    takes an image from the camera and saves it to filepath

    Args:
        filepath (str): The file path to save the image to
    """
    
    #takes a photo and saves the image to the filepath as imageX.jpg, where X is the lowest free number in the file
    camera = PiCamera()
    camera.resolution = (2592, 1944)
    camera.start_preview()
    sleep(2)
    camera.capture(filepath+"/image"+str(get_next_image_number(filepath))+".jpg")