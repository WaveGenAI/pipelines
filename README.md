# Audio Pipelines for Creating a Music Dataset

This repository provides a set of audio processing pipelines designed to facilitate the creation of a music dataset. The tools included allow for downloading audio files from URLs, transcribing audio into text, generating prompts, and splitting audio files into smaller chunks. Follow the setup instructions and use the provided commands to manage and preprocess your audio data effectively.

## Setup

Before you begin, you need to install the required dependencies. Run the following command to install them:

```bash
python3 -m pip install -r requirements.txt
```

This command installs all the necessary Python packages listed in the `requirements.txt` file. Make sure you have Python 3 and `pip` installed on your system.

## Run the pipelines

Start the proxy:

```bash
docker run -d --rm -it -p 3128:3128 -p 4444:4444 -e "TOR_INSTANCES=40" zhaowde/rotating-tor-http-proxy
```

To run the pipelines

```bash
python3 main.py  --huggingface SRC_DS --output_dataset TGT_DS --use_cache --cache_dir DIR --download
```
