# Audio Pipelines for Creating a Music Dataset
This repository provides a set of audio processing pipelines designed to facilitate the creation of a music dataset. The tools included help in downloading audio files, creating prompts, and splitting audio into manageable segments.

## Setup

Before you begin, you need to install the required dependencies. Run the following command to install them:

```bash
python3 -m pip install -r requirements.txt
```

This command installs all the necessary Python packages listed in the `requirements.txt` file. Make sure you have Python 3 and `pip` installed on your system.

## Run the pipelines

Start the proxy:

```bash
sudo docker run -d --rm -it -p 3128:3128 -p 4444:4444 -e "TOR_INSTANCES=120" jourdelune/rotating-tor-http-proxy
```

## Pipeline Steps

### 1. Downloader

The downloader module is responsible for fetching audio files from various sources. It ensures that the files are downloaded and stored in the appropriate directory for further processing.

### 2. Prompt Creation

The prompt creation module generates prompts based on the downloaded audio files. These prompts can be used for various purposes, such as training machine learning models or creating metadata for the audio files.

### 3. Split Audio

The split audio module takes the downloaded audio files and splits them into smaller, manageable segments. This is useful for processing large audio files and making them easier to handle in subsequent steps.
