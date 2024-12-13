import copy
from datasets import load_dataset, Dataset
import itertools
import torch
import base64
from io import BytesIO
from PIL import Image
import re
import json
import pandas as pd

# done
# check system prompt token seq or user prompt token seq is in the current token list
def check_header(targets,seq):
    for i in range(len(seq)-3):
        if seq[i:i+3] in targets:
            return True
    return False

# done
def replace_target(target,seq):
    for i in range(len(seq)-3):
        if seq[i:i+3] == target:
            seq[i],seq[i+1],seq[i+2] = -100,-100,-100
    return seq

# done
def tokenize_dialogs(dialogs, images, processor):
    text_prompt = processor.apply_chat_template(dialogs)
    batch = processor(images=images, text=text_prompt,padding = True, return_tensors="pt")
    label_list = []
    for i in range(len(batch["input_ids"])):
        dialog_tokens = batch["input_ids"][i].tolist()
        labels = copy.copy(dialog_tokens)
        eot_indices = [i for i,n in enumerate(labels) if n == 128009]
        last_idx = 0
        # system prompt header "<|start_header_id|>system<|end_header_id|>" has been tokenized to [128006, 9125, 128007]
        # user prompt header "<|start_header_id|>user<|end_header_id|>" has been tokenized to [128006, 882, 128007]
        prompt_header_seqs = [[128006, 9125, 128007],[128006, 882, 128007]]
        for n, idx in enumerate(eot_indices):
            current_seq = labels[last_idx:idx+1]
            if check_header(prompt_header_seqs,current_seq):
                # found prompt header, indicating that this seq should be masked
                labels[last_idx:idx+1] = [-100] * (idx-last_idx+1)
            else:
                last_idx = idx+1
            #  Mask all the assistant header prompt <|start_header_id|>assistant<|end_header_id|>, which has been tokenized to [128006, 78191, 128007]
        assistant_header_seq = [128006, 78191, 128007]
        labels = replace_target(assistant_header_seq,labels)
        # Mask the padding token and image token 128256 
        for i in range(len(labels)):
            if labels[i] == processor.tokenizer.pad_token_id or labels[i] == 128256: #  128256 is image token index
                labels[i] = -100
        label_list.append(labels)
    batch["labels"] = torch.tensor(label_list)
    return batch


def get_custom_dataset(dataset_config, processor, split, split_ratio=0.99):
    # load_dataset will return DatasetDict that contains all the data in the train set
    dataset_dict = load_dataset("csv", data_files=dataset_config.data_path, header=0)
    dataset = dataset_dict['train']
    # df = pd.read_csv(dataset_config.data_path, header=0)
    # dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=1-split_ratio, shuffle=True, seed=42)[split]
    return dataset

# done
class LUAD_LUSC:
    def __init__(self, processor):
        self.processor = processor
        self.processor.tokenizer.padding_side = "right" # during training, one always uses padding on the right
    def __call__(self, samples):
        # deal with many images one text
        dialogs, images = [],[]
        img_text_content = {"type": "image"}

        for sample in samples:
            image_list, sample_list = sample["images"], sample["texts"]
            image_list = re.sub(r"[\[\]']", "", image_list).split(',')
            user_content = []
            for idx, image in enumerate(image_list):
                try:
                    user_content.append(img_text_content)
                    # turn base64 img into pil item
                    image = base64.b64decode(image)
                    image = BytesIO(image)
                    image = Image.open(image).convert("RGB")

                    image_list[idx] = image
                except:
                    print(image)

            # system content
            sample_list = re.sub("'", '"', sample_list)
            sample_list = json.loads(sample_list)
            user_content.append({"type": "text", "text": sample_list["user"]})
            assistant_content = [{"type": "text", "text": sample_list["assistant"]}]
            dialog = [ 
                {"role":"user","content": user_content},
                {"role":"assistant","content": assistant_content}
            ]
            dialogs.append(dialog)
            images.append(image_list)

        return tokenize_dialogs(dialogs, images, self.processor)


# done
def get_data_collator(processor):
    return LUAD_LUSC(processor)

