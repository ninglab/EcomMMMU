from utils.utils import *
from utils.template import *

def get_downstram_chat(entry):
    conv = entry["conversations"][0]["value"]
    num_images = len([m.start() for m in re.finditer("<image>", conv)])
    conv = conv.replace("<image>", "").strip()
    chat = [{
          "role": "system",
          "content": [{"type": "text", "text": "You are a helpful shopping assistant."}]},
        {
          "role": "user",
          "content": [
            {"type": "text", "text": conv}] + \
            [{"type": "image"}] * num_images
        }]

    return chat

def inf(model, processor, task):
    data = load_json('data/downstrem/test.json')
    results = []

    for entry in tqdm(data):
        if entry['task'] != task:
            continue

        chat = get_downstram_chat(entry)
        prompt = processor.apply_chat_template(chat, add_generation_prompt=True)

        if 'image' in entry:
            images = [normalize_img(Image.open(requests.get(img, stream=True).raw)) for img in entry['image']]
        else:
            images = None

        inputs = processor(images=images, text=prompt, return_tensors="pt").to(device=device, dtype=torch.bfloat16)
        outputs = model.generate(**inputs, max_new_tokens=16)
        response = processor.batch_decode(outputs, skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
        response = get_valid_response(entry["task"], response)

        results.append(response)
    
    dump_json(results, f'results/{task}.json')

if __name__ == "__main__":
    model = LlavaForConditionalGeneration.from_pretrained(
        'lmms-finetune/ckp_merged/downstream_llava',
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)
    model.vision_tower.to(dtype=torch.bfloat16, device=device)
    processor = AutoProcessor.from_pretrained(model_id, device=device, dtype=torch.bfloat16, vision_feature_select_strategy='default')

    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--task')
    args = parser.parse_args()
    inf(model, processor, args.task)