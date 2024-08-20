from src.pipelines.transcript.transcription import TranscriptModel

AUDIO_PATH = """ 
/media/works/audio_v2/1736967003.mp3
""".strip()

model = TranscriptModel()
lyrics = model.transcript(AUDIO_PATH)
