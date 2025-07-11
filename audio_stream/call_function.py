import requests
import tempfile
import wave
import pyaudio
from playsound import playsound

# Constants
RECORD_SECONDS = 2
SAMPLE_RATE = 16000
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
URL = "https://voice-bot-7cijur72sa-uc.a.run.app"

def record_audio(file_path):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("🎙️ Recording for 5 seconds...")

    frames = []
    for _ in range(0, int(SAMPLE_RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("✅ Recording complete")

    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(frames))


def send_audio_and_play_response(audio_path):
    with open(audio_path, 'rb') as audio_file:
        files = {'audio': ('audio.wav', audio_file, 'audio/wav')}
        print("📤 Sending audio to bot...")
        response = requests.post(URL, files=files)

    if response.status_code == 200:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_mp3:
            temp_mp3.write(response.content)
            mp3_path = temp_mp3.name
        print("🔊 Playing bot response...")
        playsound(mp3_path)
    else:
        print("❌ Error:", response.status_code)
        print(response.text)


if __name__ == "__main__":
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
        wav_path = temp_wav.name

    record_audio(wav_path)
    send_audio_and_play_response(wav_path)
