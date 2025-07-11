import asyncio
import pyaudio
import time
import numpy as np
from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents
import pygame 
import datetime
from prep import genai, MODEL
import os 
from google.cloud import speech,texttospeech


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "cred.json"
generative_model=genai.GenerativeModel(MODEL)
DEEPGRAM_API_KEY = ""

RATE = 16000
CHUNK = 1600
CHANNELS = 1
SILENCE_TIMEOUT = 3.1
VOLUME_THRESHOLD = 500
INITIAL_GRACE_PERIOD = 5.0

class HindiStreamingBot:
    def __init__(self, api_key):
        self.deepgram = DeepgramClient(api_key)
        self.connection = None
        self.audio_stream = None
        self.is_streaming = False
        self.keepalive_task = None
        self.final_transcripts = []
        self.current_interim = ""
        self.last_speech_time = None
        self.start_time = None
        self.speech_started = False

    async def setup_connection(self):
        options = LiveOptions(
            model="nova-2",
            language="hi",
            encoding="linear16",
            sample_rate=RATE,
            channels=CHANNELS,
            interim_results=True,
            smart_format=True,
            punctuate=True,
            vad_events=True,
            endpointing=300
        )

        self.connection = self.deepgram.listen.asyncwebsocket.v("1")
        self.connection.on(LiveTranscriptionEvents.Transcript, self.on_transcript)
        self.connection.on(LiveTranscriptionEvents.Error, self.on_error)

        return await self.connection.start(options)

    async def on_transcript(self, connection, result, **kwargs):
        if not result.channel or not result.channel.alternatives:
            return

        transcript = result.channel.alternatives[0].transcript.strip()
        if not transcript:
            return

        current_time = time.time()
        self.last_speech_time = current_time
        self.speech_started = True

        if result.is_final:
            if self.current_interim:
                print("\r" + " " * len(self.current_interim) + "\r", end="", flush=True)
            print(transcript)
            self.final_transcripts.append(transcript)
            self.current_interim = ""
        else:
            display = f"{transcript}"
            if self.current_interim:
                print("\r" + " " * len(self.current_interim) + "\r", end="", flush=True)
            print(display, end="", flush=True)
            self.current_interim = display

    async def on_error(self, connection, error, **kwargs):
        print("Error:", error)

    async def send_keepalive(self):
        while self.is_streaming:
            await asyncio.sleep(8)
            if self.connection:
                try:
                    # ✅ Just call `.ping()` or skip entirely if not needed
                    await self.connection.keep_alive()
                except Exception as e:
                    print("Keepalive error:", e)
                    break


    async def stream_audio(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=CHANNELS, rate=RATE,
                        input=True, frames_per_buffer=CHUNK)

        self.audio_stream = stream
        self.start_time = time.time()

        try:
            while self.is_streaming:
                data = stream.read(CHUNK, exception_on_overflow=False)
                if self.connection:
                    await self.connection.send(data)

                volume = np.abs(np.frombuffer(data, dtype=np.int16)).mean()
                now = time.time()
                grace_period = (now - self.start_time) < INITIAL_GRACE_PERIOD

                if volume > VOLUME_THRESHOLD and not self.speech_started and not grace_period:
                    self.speech_started = True
                    self.last_speech_time = now

                if self.speech_started and self.last_speech_time:
                    if now - self.last_speech_time > SILENCE_TIMEOUT:
                        self.is_streaming = False
                        break

                await asyncio.sleep(0.01)
        finally:
            stream.close()
            p.terminate()

    async def start(self):
        print("Listening for Hindi speech...")
        if not await self.setup_connection():
            print("Failed to connect to Deepgram.")
            return ""

        self.is_streaming = True
        self.keepalive_task = asyncio.create_task(self.send_keepalive())

        try:
            await self.stream_audio()
        finally:
            await self.cleanup()
            return " ".join(self.final_transcripts)

    async def cleanup(self):
        self.is_streaming = False

        if self.keepalive_task:
            self.keepalive_task.cancel()
            try:
                await self.keepalive_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print("Keepalive task error:", e)

        if self.audio_stream:
            self.audio_stream.close()

        if self.connection:
            try:
                await self.connection.finish()  # Only this, no send()
            except Exception as e:
                print("Connection finish error:", e)


def run_bot():
    try:
        return asyncio.run(HindiStreamingBot(DEEPGRAM_API_KEY).start())
    except KeyboardInterrupt:
        print("\nTerminated by user.")
        return ""
    except Exception as e:
        print(f"Runtime error: {e}")
        return ""
    

def get_gemini_results(transcript,buffer1,buffer2):
    date=datetime.date.today()
    prompt=f'''
You are a friendly female cab booking assistant for कैबस्वाले app. Speak naturally like a warm, helpful woman would in real conversation.
LANGUAGE RESPONSE RULE:
- If user speaks completely in English: Respond completely in English (simple, conversational English)
- If user speaks completely in Hindi/regional languages: Respond in Hindi (Hindi mixed with simple English words)
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
- Unlike other platforms: We let you choose your driver, not just the car
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
- Don't repeat what user said
- Talk like a real woman, not a robot - vary your responses naturally
- Keep responses under 15 words
- Use feminine conversational style - gentle but confident
- Mirror the user's language mixing style naturally
MANDATORY BOOKING INFO (collect all except return date):
1. Source City (pickup city only)
2. Destination City (drop city only)
3. Number of Passengers (no of people traveling)
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
- Todays date is {date}
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
CURRENT CONTEXT:
User said: "{transcript}"
Previous conversation: "{' '.join(buffer1)}"
Your previous replies: "{' '.join(buffer2)}"
Give ONE natural, varied response that mirrors the user's language mixing style and sounds like a real woman continuing the booking conversation. Check your previous replies to ensure you don't repeat the same phrases.
'''

    results=generative_model.generate_content([prompt]+[transcript])    
    output=results.text
    return output

import threading


stop_tts_flag = threading.Event()
tts_thread = None

def get_TTS(output_text):
    def play_audio():
        tts_client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=output_text)
        voice = texttospeech.VoiceSelectionParams(
            language_code='hi-IN',
            name="hi-IN-Wavenet-A",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16)

        tts_response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        with open("response.wav", "wb") as out:
            out.write(tts_response.audio_content)

        pygame.mixer.init()
        pygame.mixer.music.load("response.wav")
        pygame.mixer.music.play()
        print("Speaking...")

        while pygame.mixer.music.get_busy():
            if stop_tts_flag.is_set():
                print("Interrupted by user.")
                pygame.mixer.music.stop()
                break
            pygame.time.Clock().tick(10)

        pygame.mixer.quit()

    stop_tts_flag.clear()
    t = threading.Thread(target=play_audio)
    t.start()
    return t



if __name__ == "__main__":
    user_buffer = []
    ai_buffer = []
    while True:
        try:
            user_transcript = run_bot()

            if tts_thread and tts_thread.is_alive():
                #  If user speaks during TTS, interrupt it
                stop_tts_flag = True
                tts_thread.join()

            user_buffer.append(user_transcript)
            print(" Getting Gemini results...")
            gemini_results = get_gemini_results(user_transcript, user_buffer, ai_buffer)
            print(gemini_results)
            ai_buffer.append(gemini_results)

            tts_thread = get_TTS(gemini_results)

        except KeyboardInterrupt:
            print("\n Exiting...")
            break
        except Exception as e:
            print(f" Error: {e}")

