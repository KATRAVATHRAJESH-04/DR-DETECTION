import requests
import time

url = "http://localhost:8000/health"
print(f"Checking {url}...")
try:
    r = requests.get(url, timeout=5)
    print(f"Status: {r.status_code}")
    print(f"Response: {r.json()}")
except Exception as e:
    print(f"Error: {e}")
