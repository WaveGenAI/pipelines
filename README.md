# pipelines
Processing audio pipelines

## Setup

```
python3 -m pip install -r requirements.txt
```

## Download audio url

```
cd src/pipelines
python3 -m scripts.dl.py
```

## Process

```
cd src/pipelines
python3 -m scripts.process.py
```

## Export to huggingface

```
cd src/pipelines
python3 -m scripts.huggingface.py
```
