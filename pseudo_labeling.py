from utils.utils import *
from utils.template import *

def get_labeling_chat(entry, mode='text'):
    num_images = vrf_img_num(entry)

    content = [{"type": "text", "text": entry["instruction"]+entry['input']}]
    if mode != 'text':
        content += [{"type": "image"}] * num_images

    chat = [
          {"role": "system",
          "content": [{"type": "text", "text": "You are a helpful shopping assistant."}]},
        {
          "role": "user",
          "content": content
        }]
    
    return chat
    
def process_text(entry):
    chat = get_labeling_chat(entry, 'text')
    prompt = processor.apply_chat_template(chat, add_generation_prompt=True)
    inputs = processor(images=None, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.bfloat16)
    try:
        outputs = model.generate(**inputs, max_new_tokens=16)
    except:
        return ''
    response = processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    response = get_valid_response(entry["task"], response)

    return response
    
def process_mm(entry):
    response_list = []
    chat = get_labeling_chat(entry, 'mm')
    prompt = processor.apply_chat_template(chat, add_generation_prompt=True)
    input
    for img_id in range(IMG_NUM):
        images = [normalize_img(Image.open(requests.get(item_img[img_id], stream=True).raw)) for item_img in entry['images']]
        inputs = processor(images=images, text=prompt, return_tensors="pt").to(device=device, dtype=torch.bfloat16)
        try:
            outputs = model.generate(**inputs, max_new_tokens=16)
        except:
            response_list.append('')
            continue
        response = processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        response = get_valid_response(entry["task"], response)
        response_list.append(response)
    return response_list

def cls_pseudo_label(text_res, mm_res):
    if not text_res and mm_res:
        return 'A: helpful'
    if not text_res and not mm_res:
        return 'B: ineffective'
    if text_res and mm_res:
        return 'C: redundant'
    if text_res and not mm_res:
        return 'D: misleading'
    
def balance_vrf_data(vrf_data):
    df = pd.DataFrame(vrf_data)
    df['output'] = [row['conversations'][-1]['value'] for idx, row in df.iterrows()]
    dict_cnt = df['output'].value_counts().to_dict()
    min_num = min(list(dict_cnt.values()))

    new_data = []
    cnt = {chr(op+65):0 for op in range(4)}
    for idx, row in df.iterrows():
        if cnt[row['conversations'][-1]['value'][0]] < min_num:
            new_data.append({
                'system_prompt': row['system_prompt'],
                'image': row['image'],
                'conversations': row['conversations']
            })
            cnt[row['conversations'][-1]['value'][0]] += 1

    return new_data

def convert_vrf_ft_data(spl):
    '''Process data for training the verifier'''

    data = pd.DataFrame(load_json(f'data/dataset.json'))
    data = data[data['split'] == spl].iloc[:len(data) // 2]
    data = data[data['is_vc'] == True]

    vrf_data = []

    for _, entry in data.iterrows():
        entry = entry.to_dict()
        text_response = process_text(entry)
        if len(text_response) == 0:
            continue

        mm_response = process_mm(entry)
        text_correct = text_response.lower() == entry['output'].lower()
        for img_id in range(IMG_NUM):
            if len(mm_response[img_id]) == 0:
                continue
            mm_correct = mm_response[img_id].lower() == entry['output'].lower()
            pseudo_label = cls_pseudo_label(text_correct, mm_correct)
            images = [item_img[img_id] for item_img in entry['images']]
            content = verifier_instr(entry)
            
            vrf_data.append({
                'system_prompt': "You are a helpful shopping assistant.",
                'image': images,
                'conversations': [
                    {
                        'from': 'human',
                        'value': '{}{}'.format(''.join(['<image>' for _ in images]), content)
                    },
                    {
                        'from': 'gpt',
                        'value': pseudo_label
                    }
                ]
            })

    vrf_data = balance_vrf_data(vrf_data)
    dump_json(vrf_data, f'data/verification/{spl}.json')

    return vrf_data

if __name__ == "__main__":

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id, dtype=torch.bfloat16, device=device)
    model.vision_tower.to(dtype=torch.bfloat16, device=device)
    
    vrf_train_data = convert_vrf_ft_data('train')
    vrf_val_data = convert_vrf_ft_data('val')
