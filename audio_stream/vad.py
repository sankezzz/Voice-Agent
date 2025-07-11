import webrtcvad
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
CHUNK_DURATION_MS = 30  
CHUNK = int(RATE * CHUNK_DURATION_MS / 1000)
SILENCE_TIMEOUT = 5
MAX_PAUSE_DURATION = 3

def generate_audio_chunks_vad(vad, silence_timeout=SILENCE_TIMEOUT):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print(" Listening with VAD...")

    speech_started = False
    last_speech_time = time.time()
    session_start_time = time.time()

    try:
        while True:
            audio_chunk = stream.read(CHUNK, exception_on_overflow=False)
            is_speech = vad.is_speech(audio_chunk, RATE)

            current_time = time.time()

            if is_speech:
                if not speech_started:
                    print(" Speech started")
                    speech_started = True
                last_speech_time = current_time
                yield audio_chunk

            else:
                if speech_started and (current_time - last_speech_time) < MAX_PAUSE_DURATION:
                    yield audio_chunk  
                else:
                    if speech_started and (current_time - last_speech_time) >= silence_timeout:
                        print(" Silence timeout reached after speech.")
                        break

                    if not speech_started and (current_time - session_start_time) >= silence_timeout:
                        print(" No speech detected at all. Timing out.")
                        break

                    yield audio_chunk  

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

import sys

def listen_print_loop(responses):
    final_transcript = ""
    last_result_time = time.time()

    for response in responses:
        if not response.results:
            continue

        result = response.results[0]
        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript.strip()
        current_time = time.time()

        if result.is_final:
            # Clear partial, show final on new line
            sys.stdout.write("\r" + " " * 80 + "\r")  # Clear current line
            sys.stdout.write(f"🎤 Final Transcript: {transcript}\n")
            sys.stdout.flush()
            return transcript
        else:
            # Overwrite the same line with updated partial transcript
            sys.stdout.write(f"\r🗣️  Listening: {transcript}")
            sys.stdout.flush()
        

    return final_transcript.strip()

def get_STT():
    client = speech.SpeechClient()

    vad = webrtcvad.Vad()
    vad.set_mode(2)  

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="hi-IN",
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
    )

    audio_generator = generate_audio_chunks_vad(vad, silence_timeout=SILENCE_TIMEOUT)
    requests = (speech.StreamingRecognizeRequest(audio_content=chunk) for chunk in audio_generator)

    responses = client.streaming_recognize(
        config=streaming_config,
        requests=requests
    )

    transcript = listen_print_loop(responses)
    return transcript if transcript else ""



def get_gemini_results(transcript,buffer1,buffer2):
    date=datetime.date.today()
    prompt=f'''
You are a friendly female cab booking assistant for कैबस्वाले app. Speak naturally like a warm, helpful woman would in real conversation.
CRITICAL: Only provide YOUR response. Do NOT repeat, echo, or include the user's message in your output. Give only your own reply.
LANGUAGE RESPONSE RULE:
- If user speaks completely in English: Respond completely in English (simple, conversational English)
- If user speaks completely in Hindi/regional languages: Respond in Hinglish (Hindi mixed with simple English words)
- If user mixes both languages (code-switching): Match their exact style - respond to English parts in English and Hindi parts in Hindi/Hinglish
- Follow the user's language pattern naturally - be flexible and adaptive
ABOUT कैबस्वाले (use this info when users ask about service):
UNIQUE FEATURES:
- Driver selection based on vibe matching - passengers can choose drivers they feel comfortable with
- All drivers are thoroughly verified with background checks and documentation
- Specialized in outstation/intercity travel with experienced long-distance drivers
- Transparent pricing with no hidden charges
- 24/7 customer support in Hindi/English
- Real-time tracking and safety features
- Option to rate and review drivers for future reference
- Flexible booking - can book immediately or schedule in advance
ADVANTAGES OVER COMPETITORS:
- Unlike Uber/Ola/Rapido: We let you choose your driver, not just the car
- Driver verification is more thorough - safety first approach
- Focused on outstation travel expertise, not just city rides
- Better rates for long-distance travel
- Personal touch - know your driver before the trip
- Regional language support and local knowledge
SAFETY FEATURES:
- Live location sharing with family/friends
- Emergency contact system
- Driver photo and details shared before trip
- Trip monitoring and check-ins
CONVERSATION STYLE:
- Be warm, caring but efficient - like a helpful sister/friend
- Use simple words, avoid difficult vocabulary
- Don't repeat what user said - acknowledge with "okay", "great", "badhiya", "achha" and move to next question
- Talk like a real woman, not a robot - vary your responses naturally
- Keep responses under 15 words
- Use feminine conversational style - gentle but confident
- Mirror the user's language mixing style naturally
MANDATORY BOOKING INFO (collect all except return date):
1. Source City (pickup city only)
2. Destination City (drop city only)
3. Number of Passengers
4. Journey Date (understand: tomorrow, today, next Monday, kal, aaj, monday ko etc.)
5. Return Date (optional - if user doesn't want, that's fine)
LOCATION STORAGE FORMAT:
- If city exists in multiple states (like Aurangabad in Bihar/Maharashtra): Store as "City, State"
- If city is unique or well-known: Store as just "City"
- Examples: "Aurangabad, Maharashtra" vs "Mumbai" or "Delhi"
INDIAN CITIES KNOWLEDGE:
- Delhi = New Delhi (same city, store as "Delhi")
- If ambiguous cities, ask state and store as "City, State"
- If village/town mentioned, ask nearest famous city
- If unsure about city name, confirm with state
DATE UNDERSTANDING:
- Current date = {date}
- Tomorrow/kal = tomorrow, today/aaj = today, day after tomorrow/parso = day after tomorrow
- "Monday"/"Monday ko" = next upcoming Monday (confirm date)
HANDLE QUESTIONS ABOUT कैबस्वाले:
When users ask about service, features, safety, or compare with competitors, briefly share relevant कैबस्वाले advantages from the info above, then continue with booking. Always vary your responses and match their language style.
HANDLE COMPETITOR MENTIONS:
If user mentions Uber, Ola, Rapido, briefly highlight कैबस्वाले advantages in a natural way, then redirect to booking. Vary your responses and match their language mixing pattern.
HANDLE IRRELEVANT TALK:
When user asks personal questions, inappropriate comments, or non-booking topics, gently redirect like a polite woman would. Based on what's missing, naturally guide them back in the same language style they used.
HANDLE DIFFICULT SITUATIONS:
Respond naturally and vary your language while matching their communication style:
- Adapt to their language preference in real-time
- If they switch languages mid-conversation, switch with them
- Maintain natural flow regardless of language mixing
IMPORTANT:
1. Be flexible with language - don't force consistency, follow the user's natural speaking pattern
2. Never repeat the exact same response twice - always vary naturally
3. Think like a real woman having a conversation, adapting to how the user naturally speaks
4. OUTPUT ONLY YOUR RESPONSE - do not include or repeat the user's message
CURRENT CONTEXT:
User said: "{transcript}"
Previous conversation: "{' '.join(buffer1)}"
Your previous replies: "{' '.join(buffer2)}"
Provide ONLY your response as the कैबस्वाले assistant. Do not repeat the user's message. Give only your own natural reply to continue the booking conversation.

'''
    results=generative_model.generate_content([prompt]+[transcript])    
    output=results.text
    return output

def get_TTS(output_text):
    tts_client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=output_text)
    voice = texttospeech.VoiceSelectionParams(language_code='hi-IN', name="hi-IN-Wavenet-A", ssml_gender=texttospeech.SsmlVoiceGender.FEMALE)#hi-IN-Chirp3-HD-Aoede
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
    user_buffer = []
    ai_buffer = []
    get_TTS("namaste आपका caabswale में स्वागत hai.")
    while True:
        try:
            user_transcript = get_STT()

            if user_transcript:
                print(f"\n You said: {user_transcript}\n")
            else:
                print("\n Silence detected. Sending empty prompt to Gemini...\n")

            user_buffer.append(user_transcript)

            print(" Getting Gemini results...")
            gemini_results = get_gemini_results(user_transcript, user_buffer, ai_buffer)
            print(gemini_results)

            ai_buffer.append(gemini_results)

            get_TTS(output_text=gemini_results)
            time.sleep(1)
            print(" You can speak again\n")

        except KeyboardInterrupt:
            print("\n Exiting...")
            break
        except Exception as e:
            print(f" Error: {e}")

