from lyric_whisper import LyricGen

lyric_gen = LyricGen()
out = lyric_gen.generate_lyrics("COVEX - Good Side (ft. Delaney Jane).mp3")

print(out)
