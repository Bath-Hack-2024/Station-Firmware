from sensorlib import get_humidity_and_temperature, get_pressure_and_temperature

p, t2 = get_pressure_and_temperature()
h, t = get_humidity_and_temperature()

print(f"Pressure: {p} hPa")
print(f"Temperature 2: {t2} C")
print(f"Humidity: {h} %")
print(f"Temperature: {t} C")
