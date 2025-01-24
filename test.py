import pyaudio

pa = pyaudio.PyAudio()
stream = pa.open(format=pyaudio.paInt16, channels=1, rate=48000, input=True, frames_per_buffer=960)
print("Listening...")
for _ in range(10):
    data = stream.read(960)
    print(f"Read {len(data)} bytes from microphone.")
stream.close()
pa.terminate()