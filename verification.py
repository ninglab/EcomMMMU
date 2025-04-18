from utils.utils import *
from utils.template import *

def get_vrf_chat(entry):
    num_images = vrf_img_num(entry)
    chat = [{
          "role": "system",
          "content": [{"type": "text", "text": "You are a helpful shopping assistant."}]},
        {
          "role": "user",
          "content": [
            {"type": "text", "text": verifier_instr(entry)}] + \
            [{"type": "image"}] * num_images
        }] 

    return chat

def verify_entry(model, processor, spl):

    data = pd.DataFrame(load_json(f'data/dataset.json'))
    data = data[data['split'] == spl]
    data = data.iloc[len(data) // 2:] if spl != 'test' else data

    downstrem_data = []
            
    for _, entry in data.iterrows():
        entry = entry.to_dict()
        prompt = processor.apply_chat_template(get_vrf_chat(entry), add_generation_prompt=True)

        cand_img_ids = []
        for img_id in range(IMG_NUM):
            images = [normalize_img(Image.open(requests.get(item_img[img_id], stream=True).raw)) for item_img in entry['images']]
            inputs = processor(images=images, text=prompt, return_tensors="pt").to(device=device, dtype=torch.bfloat16)
            try:
                outputs = model.generate(**inputs, max_new_tokens=16)
                response = processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                label = get_valid_response("verfication", response)
            except:
                label = ''

            if label == 'A':
                cand_img_ids.append(img_id)

        if len(cand_img_ids) > 0:
            selected_idx = random.choice(cand_img_ids)
            img_list = [item_img[selected_idx] for item_img in entry['images']]

        task_chat = {'system_prompt': "You are a helpful shopping assistant.",}
        if len(img_list) > 0:
            task_chat['image'] = img_list
        task_chat['conversations'] = [
                {
                    'from': 'human',
                    'value': '{}{}'.format(''.join(['<image>' for _ in img_list]), entry['instruction'] + entry['input'])
                },
            ]
        if spl != 'test':
            task_chat['conversations'].append(
                {
                    'from': 'gpt',
                    'value': entry['output']
                }
            )
        else:
            task_chat['task'] = entry['task']

        downstrem_data.append(task_chat)

    dump_json(downstrem_data, f'data/downstrem/{spl}.json')

if __name__ == "__main__":
    model = LlavaForConditionalGeneration.from_pretrained(
        "lmms-finetune/ckp_merged/verification_llava",
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)
    processor = LlavaProcessor.from_pretrained(model_id, dtype=torch.bfloat16, device=device)
    model.vision_tower.to(dtype=torch.bfloat16, device=device)

    verify_entry(model, processor, 'train')
    verify_entry(model, processor, 'val')
    verify_entry(model, processor, 'test')