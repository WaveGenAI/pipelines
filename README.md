# pipelines
Processing audio pipelines

# Setup

```
python3 -m pip install -r requirements.txt
```

# Inference

```
python3 main.py --input FILE --output DIR
```

### Faster-whisper issue

```
export LD_LIBRARY_PATH=`python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; import torch; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__) + ":" + os.path.dirname(torch.__file__) +"/lib")'`
```