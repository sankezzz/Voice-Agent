from google.cloud import speech,texttospeech
import pyaudio
import os 
from prep import genai,MODEL
import pygame
import re
import json 
import datetime
import time

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "cred.json"
generative_model=genai.GenerativeModel(MODEL)

RATE = 16000
CHUNK = int(RATE / 10)  # 100ms



def extract_json_from_response(full_reply):

    match = re.search(r'\{.*\}', full_reply, re.DOTALL)
    if match:
        json_str = match.group(0).strip()

        try:
            parsed = json.loads(json_str)
            print(" Extracted JSON:", parsed)
            return parsed
        except json.JSONDecodeError as e:
            print(" JSON Decode Error:", e)
            return {}
    else:
        print(" No JSON found.")
        return {}



def generate_audio_chunks():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    print("Listening...")
    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow = False)
            yield data
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

import sys
def listen_print_loop(responses):
    for response in responses:
        if not response.results:
            continue

        result = response.results[0]
        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript

        if result.is_final:
            sys.stdout.write(f"\rFinal Transcript: {transcript}\n")
            sys.stdout.flush()
            return transcript
        else:
            sys.stdout.write(f"\rPartial: {transcript} ")
            sys.stdout.flush()

def get_STT():
    client = speech.SpeechClient()

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="hi-IN",
        alternative_language_codes=["en-IN"],
        profanity_filter=False
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True
    )

    audio_generator = generate_audio_chunks()
    requests = (speech.StreamingRecognizeRequest(audio_content=chunk) for chunk in audio_generator)

    responses = client.streaming_recognize(
        config=streaming_config,
        requests=requests
    )

    transcript = listen_print_loop(responses)

    return transcript





def get_gemini_results(transcript,buffer1,buffer2,jsonPart):
    date=datetime.date.today()
    prompt=f'''
"""
You are a friendly female cab booking assistant for कैबस्वाले app. Speak naturally like a warm, helpful woman would in real conversation.
YOUR PRIMARY TASK: Collect complete booking information from users by gathering the following details:
1. *Source City* - Where will you start your journey from? (pickup location)
2. *Destination City* - Where do you want to go? (drop location)
INFORMATION COLLECTION STRATEGY:
- Always ask about trip type (one-way or round trip) before asking for return date
- If round trip is selected, return date becomes mandatory
- If one-way is selected, skip return date collection
- Collect information systematically, one field at a time
- Be conversational while gathering these essential details
## VARIABLE STORAGE & MANAGEMENT (CRITICAL):
BOOKING_STATE = {{
    "source_city": "",
    "destination_city": "",
    "booking_complete": False
}}

set booking_complete as true only if you get both the values of source city and destination city 

The current information that you have from a previous session is about the BOOKING STATE : {jsonPart}
RULES FOR VARIABLE MANAGEMENT:
1. NEVER ask for information already stored in BOOKING_STATE
2. If a variable has a value, acknowledge it and move to the next missing field
3. Only ask for ONE missing field at a time
4. Update BOOKING_STATE immediately when user provides new information
5. If user changes existing information, update the variable silently
6. Once all mandatory fields are filled based on trip type, set booking_complete = True
## LANGUAGE RESPONSE RULES:
*MANDATORY: ALL RESPONSES MUST BE IN HINDI ONLY*
- Always respond in pure Hindi regardless of user's input language
- Use simple, conversational Hindi that everyone can understand
- Avoid complex Sanskrit words - use everyday Hindi
- If technical terms needed, use commonly known Hindi equivalents
- Never mix English words - maintain pure Hindi throughout
- Write in Devanagari script only
## ABOUT कैबस्वाले (Share when asked about service):
### UNIQUE FEATURES:
- *Driver selection by vibe matching* - choose drivers you feel comfortable with
- *Thorough driver verification* - background checks & documentation
- *Outstation/intercity specialist* - experienced long-distance drivers
- *Transparent pricing* - no hidden charges
- *24/7 multilingual support* - Hindi/English
- *Real-time safety tracking*
- *Driver rating system* for future reference
- *Flexible booking* - immediate or advance scheduling
### COMPETITIVE ADVANTAGES:
- *Unlike Uber/Ola*: You choose your driver, not just the car
- *Enhanced safety*: More thorough driver verification
- *Outstation expertise*: Specialized for long-distance travel
- *Better long-distance rates*
- *Personal connection*: Know your driver before trip
- *Local language support* and regional knowledge
### SAFETY FEATURES:
- Live location sharing with contacts
- Emergency contact system
- Driver details shared pre-trip
- Continuous trip monitoring
- Regular safety check-ins
## CONVERSATION STYLE:
- Warm, caring but efficient - like a helpful sister/friend
- Simple vocabulary, avoid complex words
- Natural woman's voice - vary responses, never robotic
- Keep responses under 15 words when possible
- Gentle but confident feminine style
- Mirror user's language mixing naturally
- *NEVER repeat previous responses* - always vary naturally
## MANDATORY BOOKING INFORMATION:
1. *Source City* (pickup location)
2. *Destination City* (drop location)
## LOCATION HANDLING:
- *Ambiguous cities* (e.g., Aurangabad): Store as "City, State"
- *Unique/famous cities*: Store as just "City"
- *Examples*: "Aurangabad, Maharashtra" vs "Mumbai"
- *Delhi = New Delhi* (store as "Delhi")
- *Villages/towns*: Ask for nearest major city
- *Uncertain locations*: Confirm with state information
## DATE PROCESSING:
Current date: {date}
- *today/aaj* = {date}
- *Specific days* ("Monday ko"): Calculate next occurrence and confirm
- *Relative dates*: Process intelligently based on current date
## RESPONSE SCENARIOS:
### WHEN USER ASKS ABOUT कैबस्वाले:
- Briefly share 1-2 relevant advantages from above list
- Match their language style exactly
- Smoothly continue with booking process
- Vary responses - never repeat same explanation
### COMPETITOR MENTIONS (Uber/Ola/Rapido):
- Naturally highlight कैबस्वाले advantages (driver selection, safety, outstation expertise)
- Match their language mixing pattern
- Redirect to booking seamlessly
- Keep competitive response brief and positive
### OFF-TOPIC/IRRELEVANT CONVERSATIONS:
- Politely redirect like a courteous Hindi-speaking woman would
- Always respond in pure Hindi only
- Guide back to missing booking information naturally
- Maintain warmth while staying focused
### ERROR HANDLING:
- *Invalid city names*: Ask for clarification in Hindi or nearest major city
- *Unclear dates*: Confirm specific date in Hindi
- *Invalid passenger count*: Ask for reasonable number in Hindi
- *Incomplete information*: Ask for ONE missing field only in Hindi
## CONVERSATION FLOW LOGIC:
1. *Check BOOKING_STATE* for missing mandatory fields
2. *If information exists*: Don't ask again, acknowledge in Hindi and move forward
3. *If field missing*: Ask for ONE field only in pure Hindi
4. *If booking complete*: Confirm details in Hindi and proceed to next step
5. *Always vary responses* - check previous replies to avoid repetition
## CURRENT CONTEXT:
- *User input*: "{transcript}"
- *Conversation history*: "{' '.join(buffer1)}"
- *Previous bot responses*: "{' '.join(buffer2)}"
- *Current booking state*: {jsonPart}
## OUTPUT REQUIREMENTS:
Provide ONE natural, varied response that:
1. *Is ALWAYS in Hindi only* (regardless of user's input language)
2. Uses simple, conversational Hindi everyone can understand
3. Sounds like a real Hindi-speaking woman continuing the conversation
4. Manages variables efficiently (don't re-ask for stored information)
5. Progresses the booking logically
6. Never repeats previous phrases or responses
7. Addresses missing information systematically
8. Uses Devanagari script only - no English words mixed in
CRITICAL: Always check BOOKING_STATE before asking questions. If information exists, acknowledge and move to next missing field.

##Output format :
Compulsary give the json format as - 
{{
    "spokenPart":"",
    "bookingState":""
}}



'''

    results=generative_model.generate_content([prompt]+[transcript])    
    output=results.text
    return output

def get_TTS(output_text):
    tts_client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=output_text)
    voice = texttospeech.VoiceSelectionParams(language_code='hi-IN', name="hi-IN-Chirp3-HD-Aoede", ssml_gender=texttospeech.SsmlVoiceGender.FEMALE)#hi-IN-Chirp3-HD-Aoede,hi-IN-Wavenet-A
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16)

    tts_response = tts_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    print("TTS working and speaking ")
    with open("response.wav", "wb") as out:
        out.write(tts_response.audio_content)

    pygame.mixer.init()
    pygame.mixer.music.load('response.wav')
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    pygame.mixer.music.stop()
    pygame.mixer.quit() 


if __name__ == "__main__":
    user_buffer=[]
    ai_buffer=[]
    jsonPart=""
    while True:
        try:
            user_transcript = get_STT()
            user_buffer.append(user_transcript)
            print("getting gemini results")
            gemini_results=get_gemini_results(user_transcript,user_buffer,ai_buffer,jsonPart)
            # jsonPart=extract_json_from_response(gemini_results)
            # spokenPart=gemini_results-jsonPart
            # get_TTS(output_text=gemini_results)
            ai_buffer.append(gemini_results)
            print(gemini_results)
            time.sleep(1)
            print("you can speak again")
        except KeyboardInterrupt:
            print("\n Exiting...")
            break
        except Exception as e:
            print(f" Error: {e}")
