import streamlit as st
import numpy as np
import sounddevice as sd
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
import tempfile
import google.generativeai as genai
import os
from gtts import gTTS
import io
from scipy.io import wavfile
import librosa
import soundfile as sf

# Set page title
st.set_page_config(page_title="Voice Query Assistant")

def record_audio(duration, sample_rate=16000):
    st.write("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    st.write("Recording finished")
    return audio

def preprocess_audio(audio):
    # Normalize audio
    audio = audio / np.max(np.abs(audio))
    
    # Convert to AudioSegment
    audio_segment = AudioSegment(
        audio.tobytes(), 
        frame_rate=16000, 
        sample_width=audio.dtype.itemsize, 
        channels=1
    )
    
    # Apply VAD
    chunks = split_on_silence(audio_segment, min_silence_len=500, silence_thresh=-40)
    return chunks

def speech_recognition(audio_chunks):
    recognizer = sr.Recognizer()
    text = ""
    
    for chunk in audio_chunks:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            chunk.export(temp_audio.name, format="wav")
            with sr.AudioFile(temp_audio.name) as source:
                audio = recognizer.record(source)
                try:
                    text += recognizer.recognize_google(audio) + " "
                except sr.UnknownValueError:
                    st.warning("Could not understand a portion of the audio")
                except sr.RequestError as e:
                    st.error(f"Could not request results; {e}")
            os.unlink(temp_audio.name)
    
    return text.strip()

def voice_to_text_pipeline(audio):
    preprocessed_audio = preprocess_audio(audio)
    raw_text = speech_recognition(preprocessed_audio)
    return raw_text

def text_to_text_generation(text):
    os.environ["GOOGLE_API_KEY"]="GOOGLE_API_KEY"
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel('gemini-pro')
    prompt = f"""
    You are an AI assistant responding to voice commands. Provide meaningful answers that are no longer than 2 sentences. 
    The user's question is as follows: {text}
    Generate a brief response considering the context and ensure clarity. 
    If you're unsure or don't know the answer, respond with: 
    "I'm not able to provide an answer to that question. Could you please provide more clarity or ask a different question?"
    """

    response = model.generate_content(prompt, stream=True)
    ai_answer = ''.join(chunk.text for chunk in response)
    return ai_answer

def audio_generation(ai_answer, pitch, speed, voice):
    ai_answer = ai_answer.replace('\n', ' ')
    ai_answer = ai_answer.replace('*', '')
    tts = gTTS(text=ai_answer, lang="en", slow=False)
    audio_file = io.BytesIO()
    tts.write_to_fp(audio_file)
    audio_file.seek(0)
    
    # Load the audio file
    y, sr = librosa.load(audio_file, sr=16000)
    
    # Adjust pitch
    y_pitched = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch)
    
    # Adjust speed
    y_speed = librosa.effects.time_stretch(y_pitched, rate=speed)
    
    # Save the modified audio
    output_file = io.BytesIO()
    sf.write(output_file, y_speed, sr, format='wav')
    output_file.seek(0)
    
    return output_file

# Streamlit app
st.title("Voice Query Assistant")

st.write("Click the button below to start recording your voice:")

duration = st.slider("Recording duration (seconds)", 1, 10, 5)

# Tunable parameters
pitch = st.slider("Pitch adjustment", -5.0, 5.0, 0.0, 0.1)
speed = st.slider("Speed adjustment", 0.5, 2.0, 1.0, 0.1)
voice = st.radio("Voice type", ("Male", "Female"))

if st.button("Start Recording"):
    audio = record_audio(duration)
    
    st.write("Processing audio...")
    result = voice_to_text_pipeline(audio)
    
    st.write("Transcription:")
    st.write(result)
    
    answer = text_to_text_generation(result)
    st.write("AI Response:")
    st.write(answer)

    st.write("Generated Audio:")
    audio_output = audio_generation(answer, pitch, speed, voice)
    st.audio(audio_output, format='audio/wav')

st.write("Note: This app uses Google's speech recognition and generative AI services. Make sure you have an active internet connection.")
