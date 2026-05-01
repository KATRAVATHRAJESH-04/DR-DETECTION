import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.llm_engine import get_gemini_chat
import google.generativeai as genai
import os

key = os.environ.get("GEMINI_API_KEY", "AIzaSyDoTwlAAumem7cc33Js7ylxkV62anfzXFI")

res1 = get_gemini_chat('What is this?', 1, 0.9, key, [])
print("--- Response 1 ---")
print(res1)

res2 = get_gemini_chat('Is it contagious?', 1, 0.9, key, [{'role': 'user', 'text': 'What is this?'}, {'role': 'ai', 'text': res1}])
print("--- Response 2 ---")
print(res2)

