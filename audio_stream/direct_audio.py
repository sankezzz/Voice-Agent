#in this we will give directly the audio output to the gemini 
from google.genai import types
from google.genai import Client

client = Client()  

with open('path/to/small-sample.mp3', 'rb') as f:
    audio_bytes = f.read()

response = client.models.generate_content(
  model='gemini-2.5-flash',
  contents=[
    'Describe this audio clip',
    types.Part.from_bytes(
      data=audio_bytes,
      mime_type='audio/mp3',
    )
  ]
)

print(response.text)