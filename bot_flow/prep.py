import os 
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

API_KEY=os.getenv('GEMINI_API_KEY')

MODEL=os.getenv('MODEL_GEMINI_2_FLASH')

genai.configure(api_key=API_KEY)