from faster_whisper import BatchedInferencePipeline, WhisperModel

model = WhisperModel("large-v3", device="cuda", compute_type="int8_float16")

print("Transcribing audio file...")
segments, info = model.transcribe(
    "/media/works/audio_v2/9640675627.mp3",
)

probs = []
text = ""
for segment in segments:
    probs.append(segment.avg_logprob)
    text += segment.text.strip() + "\n"

probs = sum(probs) / len(probs)

print(text.strip())
print("Average logprob: %.2f" % (probs))
