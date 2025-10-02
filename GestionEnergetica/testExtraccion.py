import requests

SERVER = "https://shelly-193-eu.shelly.cloud"
AUTH_KEY = "MzQ4MzQ2dWlk81D0463E4A86968A9B2D1D3ABBE13B2629208C4C288D9DDF8AE9C45E112BA1ED7AD6BDC5BD880E7F"

DEVICE_ID = "2cbcbba6337c"

url = f"{SERVER}/device/status"
params = {"id": DEVICE_ID, "auth_key": AUTH_KEY}

r = requests.get(url, params=params)
print(r.status_code, r.json())
