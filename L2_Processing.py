import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import ephem
from datetime import datetime
import math
import numpy as np

# Define the CNN architecture (same as the one used during training)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 8 * 8, 128)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = x.view(-1, 128 * 8 * 8)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
def hasClouds(image: Image):
    """
    Function that checks if an image contains clouds

    Args:
        image (Image): An image

    Returns:
        bool: true if a cloud present, false if not
    """
    
    if(type(image)!=Image):
        raise TypeError("image argument must be of type Image")
    
    # Load the saved model
    model = CNN()
    model.load_state_dict(torch.load('CloudWeights.pth'))
    model.eval()

    # Define transformations to be applied to the input image
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    # If the image has 4 channels (RGBA), convert it to RGB
    if image.mode == 'CMYK' or image.mode == 'RGBA':
        image = image.convert('RGB')

    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform the prediction
    with torch.no_grad():
        output = model(image_tensor)
        predicted = (output > 0.5).float().item()

    # Interpret the prediction
    return not predicted

def getCloudCoverPercentage(filepath: str):
    """
    function to get the percentage of an image with cloud coverage

    Args:
        filepath (str): the filepath of the image

    Returns:
        int: percentage of the image with cloud cover
    """
    if(type(filepath)!=str):
        raise TypeError("Filepath argument must be of type string")
    
    try:
        f = open(filepath)
    except:
        raise FileNotFoundError("File "+filepath+" not found or inaccessible")
    
    # Load the image
    image = Image.open(filepath)

    # Get the dimensions of the image
    width, height = image.size

    # Define the size of each subimage
    subimage_width = width // 5  # Divide width into 10 parts
    subimage_height = height // 5  # Divide height into 10 parts

    subimages = []

    # Iterate through each grid cell and extract the subimage
    for i in range(5):
        for j in range(5):
            left = j * subimage_width
            upper = i * subimage_height
            right = left + subimage_width
            lower = upper + subimage_height

            # Crop the subimage
            subimage = image.crop((left, upper, right, lower))

            # Append the subimage to the list
            subimages.append(subimage)
    
    cloudPercentage = 0
    for subimage in subimages:
        cloudPercentage+=4*hasClouds(subimage)
    
    return cloudPercentage


def getMoonPhase():
    """
        Gets the current phase of the moon
    """
    # Example: Specific date in UTC
    date = datetime.now()
    
    moon = ephem.Moon(date)
    moon_phase = moon.phase
    
    phases = {
    0: "New Moon",
    0.25: "Waxing Crescent",
    0.5: "First Quarter",
    0.75: "Waxing Gibbous",
    1: "Full Moon",
    1.25: "Waning Gibbous",
    1.5: "Last Quarter",
    1.75: "Waning Crescent"
}

    # Find the closest phase based on the value
    closest_phase = min(phases, key=lambda x: abs(x - moon_phase))

    return{phases[closest_phase]}

def isMoonVisible(latitude: float, longitude: float):
    if(type(latitude)!=float):
        raise TypeError("Latitude argument not of type float")
    if(type(longitude)!=float): 
        raise TypeError("Longitude argument not of type float")
    
    observer = ephem.Observer()
    observer.lat = str(latitude)  # Latitude in degrees
    observer.lon = str(longitude)  # Longitude in degrees

    # Compute the observer's local sidereal time (LST)
    observer.date = datetime.now()
    lst = observer.sidereal_time()

    # Moon
    moon = ephem.Moon()
    moon.compute(observer)

    # Calculate Moon's Local Hour Angle (LHA)
    ramoon = moon.ra  # Right Ascension of the Moon
    lha = lst - ramoon

    # Check if Moon is above the horizon
    decmoon = moon.dec  # Declination of the Moon
    lat = math.radians(latitude)
    dec = math.radians(decmoon)
    lha = math.radians(lha)
    altitude = math.asin(math.sin(dec) * math.sin(lat) + math.cos(dec) * math.cos(lat) * math.cos(lha))
    altitude_deg = math.degrees(altitude)

    # Check if Moonrise and Moonset
    moon_rise = observer.previous_rising(moon)
    moon_set = observer.next_setting(moon)

    # Check if Moon is above horizon
    return altitude_deg > 0


def getCurrentScore(cloudCoverPercentage : float, fogCoverPercentage : float, moonPhase: str, moonVisible: bool, dewPointSpread: float, wobble=None):
    """
    gets the current score for a location based on the conditions that impact the quality of astral photography

    Args:
        cloudCoverPercentage (float): cloud cover percentage
        fogCoverPercentage (float): fog cover percentage
        moonPhase (str): the current phase of the moon
        moonVisible (bool): is the onn visible
        dewPointSpread (float): the current dew point spread level
        wobble (float) : default = None. Atmospheric wobble

    Raises:
        TypeError: for cloud cover
        TypeError: for fog cover
        TypeError: tests moon phase
        TypeError: moon visibility 
        TypeError: dew point spread
        TypeError: Wobble error 

    Returns:
        int: score 
    """
    if not str(cloudCoverPercentage).isnumeric():
        raise TypeError("Cloud cover percentage should be numeric")
    if not str(fogCoverPercentage).isnumeric():
        raise TypeError("Fog cover percentage should be numeric")
    if type(moonPhase)!=str:
        raise TypeError("moon phase should be a string")
    if type(moonVisible)!=bool:
        raise TypeError("Moon visibility should be of type bool")
    if not str(dewPointSpread).isnumeric():
        raise TypeError("Dew point spread should be numeric")
    
    score = 100.0
    #deals with the moon. If it isn't visible, no impact, otherwise score decreases with the brightness (phase)
    if not moonVisible:
        pass
    else:
        if moonPhase=="New Moon":
            score *= 0.8
        elif moonPhase=="Waxing Crescent" or moonPhase=="Waning Crescent":
            score *= 0.65
        elif moonPhase=="First Quarter" or moonPhase=="Last Quarter":
            score *= 0.5
        elif moonPhase=="Waxing Gibbous" or moonPhase=="Waning Gibbous":
            score *= 0.35
        elif moonPhase=="Full Moon":
            score *= 0.2

    #Accounts for cloud cover percentage - exponentially lower the higher the cloud cover
    score *= (10**(-0.01*cloudCoverPercentage))

    #Accounts for fog cover percentage - exponentially lower but not quite as quick as cloud cover with more fog
    score *= (10**(-0.005*fogCoverPercentage))

    #Accounts for Dew Point spread. Higher absolute values of dew point spread are better
    score *= (10**((0.00006*dewPointSpread)-0.006))
    
    #Wobble
    if wobble is not None:
        score *= np.min([1, 1.05-(0.01*wobble)])
    
    return score


    
print(getCurrentScore(5, 12, "New Moon", True, 40, 5))
print(isMoonVisible(51.3782, -2.3264))
print(getMoonPhase())
    
    