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
sudo docker run -d --rm -it -p 3128:3128 -p 4444:4444 -e "TOR_INSTANCES=40" jourdelune/rotating-tor-http-proxy
```

## Pipeline Steps

### 1. Downloader

The downloader module is responsible for fetching audio files from various sources. It ensures that the files are downloaded and stored in the appropriate directory for further processing.

```
python3 -m scripts.downloader --input_dataset WaveGenAI/youtube-cc-by-music --cache_dir PATH --max_files 50000 --shuffle
```

### 2. Split Audio

The split audio module takes the downloaded audio files and splits them into smaller, manageable segments. This is useful for processing large audio files and making them easier to handle in subsequent steps.

```
python3 -m scripts.split_data --input_dir DIR --output_dir DIR  --remove-original --chunk-size 30
```


### 3. Prompt Creation

The prompt creation module generates prompts based on the descriptions of the audio.

```
python3 -m scripts.prompt_creator --input_dataset HUGGING_FACE_DS --use_cache --cache_dir DIR
```

### 4. Push to huggingface

Push the dataset to huggingface for further processing.

```
python3 -m scripts.push_to_huggingface --input_dir DIR --output_dataset NAME
```

### 5. Codec Conversion

The codec conversion module converts audio files to DAC format, then it could be used to train a transformer model.

```
python3 -m scripts.codec_generator --input_dataset NAME/DIR --output_dataset NAME --streaming
```