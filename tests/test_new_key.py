import google.generativeai as genai
import os

key = "AIzaSyCRcZCxe8ZFXWk6Gp1rODIQWPLw67mkn3I"
genai.configure(api_key=key)

try:
    model = genai.GenerativeModel("gemini-2.0-flash-lite")
    response = model.generate_content("Hello! Can you hear me?")
    print("SUCCESS: The key works!")
    print(response.text)
except Exception as e:
    print("ERROR:")
    print(e)
