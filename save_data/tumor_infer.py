import argparse
import os
import sys

import json
import PIL
import torch
from accelerate import Accelerator
from PIL import Image as PIL_Image
from transformers import MllamaForConditionalGeneration, MllamaProcessor
from peft import PeftModel, PeftConfig
import re 
import csv
from io import BytesIO
import base64

# attribute -> LUAD: 0, LUSC: 1

csv.field_size_limit(sys.maxsize)

accelerator = Accelerator()

device = accelerator.device

# Constants
DEFAULT_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"


def load_model_and_processor(peft_model_path: str):
    """
    Load the model and processor based on the 11B or 90B model.
    """
    # peft
    config = PeftConfig.from_pretrained(peft_model_path)

    model = MllamaForConditionalGeneration.from_pretrained(
        config.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        device_map=device,
    )
    processor = MllamaProcessor.from_pretrained(config.base_model_name_or_path, use_safetensors=True)
    
    model = PeftModel.from_pretrained(model, peft_model_path, config=config)
    model, processor = accelerator.prepare(model, processor)
    return model, processor


def process_image(base64_str: str) -> PIL_Image.Image:
        
    image_list = re.sub(r"[\[\]']", "", base64_str).split(',')
    for idx, image in enumerate(image_list):
        image = base64.b64decode(image)
        image = BytesIO(image)
        image = PIL_Image.open(image).convert("RGB")
        image_list[idx] = image


    return image_list

def process_conversation(json_str: str, num_image: int):

    user_contents = []
    for _ in range(num_image):
        user_contents.append({"type": "image"})

    text = re.sub("'", '"', json_str)
    text = json.loads(text)
    user_contents.append({"type": "text", "text": text["user"]})
    conversation = [
                    {
                        "role": "user",
                        "content": user_contents,
                    }
     ]

    # assistant
    true_y = 1
    if re.search('LUAD', text["assistant"]) is not None:
        true_y = 0

    return conversation, true_y
    
def evaluate_data(
        model, processor, data_file_path, temperature: float, top_p: float, output_csv: str
):

    # get the data from csv
    with open(data_file_path, 'r') as csvfile:
        
        rows = csv.reader(csvfile)
        
        # skip the header
        next(rows, None)

        ans = []
        i = 0
        for row in rows:
            #print(i)
            #i += 1
            images = process_image(row[0])
            conversation, true_y = process_conversation(row[1], len(images))
            prompt = processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            inputs = processor(images, prompt, return_tensors="pt").to(device)
            output = model.generate(
                **inputs, temperature=temperature, top_p=top_p, max_new_tokens=512
            )

            # take reponse and determine pre_y
            pre_y = 1
            llm_ans = 'Wrong'
            if re.search('LUAD', processor.decode(output[0])[len(prompt) :]) is not None:
                pre_y = 0
                llm_ans = 'LUAD'
            elif re.search('LUSC', processor.decode(output[0])[len(prompt) :]) is not None:
                llm_ans = 'LUSC'
            ans.append([true_y, pre_y, llm_ans])
            
        with open(output_csv, 'w', newline='') as csvfile:

            writer = csv.writer(csvfile)
            writer.writerow(['true', 'pre', 'response'])

            writer.writerows(ans)



def main(
        data_file_path: str, temperature: float, top_p: float,  peft_model_path: str, output_csv: str
):
    """
    Call all the functions.
    """

    print(output_csv)

    model, processor = load_model_and_processor(peft_model_path)
    evaluate_data(model, processor, data_file_path, temperature, top_p, output_csv)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate text from an image and prompt using the 3.2 MM Llama model."
    )
    parser.add_argument("--data_file_path", type=str, help="Path to the data file")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation (default: 0.7)",
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9, help="Top p for generation (default: 0.9)"
    )
    parser.add_argument(
        "--peft_model_path",
        type=str,
        help="Path to peft",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        help="Path to output csv",
    )

    args = parser.parse_args()
    main(
        args.data_file_path, args.temperature, args.top_p, args.peft_model_path, args.output_csv
    )
