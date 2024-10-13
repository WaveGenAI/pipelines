import dac
from datasets import load_dataset, Dataset
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_dataset", type=str, required=True)
parser.add_argument("--output_dataset", type=str, required=True)
args = parser.parse_args()

# Download a model
model_path = dac.utils.download(model_type="44khz")
model = dac.DAC.load(model_path)

model.to("cuda")
dataset = load_dataset(args.input_dataset, streaming=True)


@torch.no_grad()
def dataset_generator():
    """
    This function will be called by the Dataset.from_generator function.
    """

    for split in dataset:
        for data in dataset[split]:
            audio = torch.Tensor(data["audio"]["array"])
            # move to [C, T] to [C, 1, T]
            audio = audio.unsqueeze(1).to(model.device)
            x = model.preprocess(audio, data["audio"]["sampling_rate"])
            _, codes, _, _, _ = model.encode(x)

            # get the prompt
            prompt = data["prompt"]

            yield {
                "codes": codes.cpu().numpy(),
                "prompt": prompt,
            }


codec_dataset = Dataset.from_generator(dataset_generator)
codec_dataset.push_to_hub(args.output_dataset)
