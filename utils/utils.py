import pickle
import json
import re
import random
import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm
import torch
import argparse
from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "llava-hf/llava-interleave-qwen-7b-hf"
IMG_NUM = 8

def load_pickle(addr):
    with open(addr, 'rb') as f:
        return pickle.load(f)
    
def dump_pickle(data, addr):
    with open(addr, 'wb') as f:
        pickle.dump(data, f)

def load_json(addr):
    with open(addr, 'r') as f:
        return json.load(f)
    
def dump_json(data, addr):
    with open(addr, 'w') as f:
        json.dump(data, f, indent=4)

def load_txt(fp, mode):
    with open(fp, 'r') as f:
        lines = f.readlines()
    if mode == 'mm':
        return [line.strip().split('\t') for line in lines]
    else:
        return [line.strip() for line in lines]
    
def normalize_img(image):
    if len(image.split()) < 3:
        image = image.convert("RGB")
    return image

def find_included_response(sentence, options):
    options = list(options)
    pattern = r'^\s*(' + '|'.join(re.escape(opt) for opt in options) + r')\b|\b(' + '|'.join(re.escape(opt) for opt in options) + r')\s*:\s*'
    
    match = re.search(pattern, sentence)
    if match:
        return match.group(1) if match.group(1) else match.group(2)
    return ""

def get_valid_response(task, response):
    valid_response_map = {
        "answerability_prediction": ['yes', 'no', 'Yes', 'No', 'YES', 'NO'],
        "binary-question_answering": ['yes', 'no', 'Yes', 'No', 'YES', 'NO', 'cannot answer', 'Cannot Answer', 'Cannot answer'],
        "sequential_recommendation": [chr(65+i) for i in range(20)],
        "click-through_predction": ['yes', 'no', 'Yes', 'No', 'YES', 'NO'],
        "multiclass_product_classification": [chr(65+i) for i in range(4)],
        "production_substitute_identification": ['yes', 'no', 'Yes', 'No', 'YES', 'NO'],
        "sentiment_analysis": [chr(65+i) for i in range(5)],
        "product_relation_prediction": [chr(65+i) for i in range(3)],
        "verfication": [chr(65+i) for i in range(4)],
    }
    options = valid_response_map[task]
    response = response.replace('\n', ' ').split('assistant')[-1]
    
    return find_included_response(response, options)

def vrf_img_num(entry):
    if entry["task"] == 'product_relation_prediction':
        num_images = 2
    elif entry["task"] in ['sequential_recommendation', 'click-through_predction']:
        num_images = len(json.loads(entry['input'])['purchase history'])
    else:
        num_images = 1
    
    return num_images