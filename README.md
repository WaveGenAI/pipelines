# Audio Pipelines for Creating a Music Dataset

This repository provides a set of audio processing pipelines designed to facilitate the creation of a music dataset. The tools included allow for downloading audio files from URLs, transcribing audio into text, generating prompts, and splitting audio files into smaller chunks. Follow the setup instructions and use the provided commands to manage and preprocess your audio data effectively.

## Setup

Before you begin, you need to install the required dependencies. Run the following command to install them:

```bash
python3 -m pip install -r requirements.txt
```

This command installs all the necessary Python packages listed in the `requirements.txt` file. Make sure you have Python 3 and `pip` installed on your system.

## Download Audio from URL

To download audio files from a list of URLs, use the following command:

```bash
python3 -m scripts.dl --input FILE --directory DIR
```

- **`--input FILE`**: Specifies the file containing the list of audio URLs to be downloaded. Each URL should be on a separate line.
- **`--directory DIR`**: Specifies the directory where the downloaded audio files will be saved.

This command reads the URLs from the specified input file and downloads each audio file into the provided directory.

## Transcribe

To convert the downloaded audio files into text transcriptions, use the following command:

```bash
python3 -m scripts.transcribe --directory DIR
```

- **`--directory DIR`**: Specifies the directory containing the audio files you wish to transcribe.

This command processes each audio file in the specified directory, using a speech recognition system to generate text transcriptions. The transcriptions are saved alongside the original audio files.

## Prompt Creation

To create prompts based on the transcriptions, use the following command:

```bash
python3 -m scripts.preprocess --directory DIR
```

- **`--directory DIR`**: Specifies the directory containing the transcription files.

This command processes the transcription files to generate prompts, which could be used for training or evaluation purposes in machine learning models, for example. The prompts are based on the content of the transcriptions.

## Split Audio into Chunks

To split audio files into smaller, more manageable chunks, use the following command:

```bash
python3 -m scripts.split --chunk-size 30 --directory DIR --output DIR
```

- **`--chunk-size 30`**: Defines the duration (in seconds) of each audio chunk. Here, audio will be split into 30-second segments. You can adjust the chunk size as needed.
- **`--directory DIR`**: Specifies the directory containing the audio files you wish to split.
- **`--output DIR`**: Specifies the directory where the resulting audio chunks will be saved.

This command splits each audio file in the input directory into smaller chunks of the specified size and saves them to the output directory. This is useful for handling large audio files, making them easier to process in subsequent steps.
