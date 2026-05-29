from modules.whisper_stt import transcribe_audio

audio_path = "temp_recording.wav"

text = transcribe_audio(audio_path)

print("\nTRANSCRIBED TEXT:\n")
print(text)