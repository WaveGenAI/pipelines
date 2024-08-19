from src.pipelines.transcript.transcription import TranscriptModel

AUDIO_PATH = """ 
/home/jourdelune/Téléchargements/test8d.mp3
""".strip()

model = TranscriptModel()
lyrics = model.transcript(AUDIO_PATH)
