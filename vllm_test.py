import torch
import os
import json
import argparse
import gc
from eval_utils import load_models
from utils import *
from tqdm import tqdm
import time
import math
import cv2
# ppocr
def parse_args():
    parser = argparse.ArgumentParser(description="Flowchart Inference")
    parser.add_argument(
        "--image_dir",
        default="/data1/gyl/HZBank/ocrres",
        type=str,
    )
    parser.add_argument(
        "--engine",
        "-e",
        choices=[
            "openflamingo",
            "otter-llama",
            "llava16-7b",
            "qwen-vl",
            "qwen-vl-max",
            "qwen-vl-chat",
            "internlm-x2",
            "internvl2",
            "emu2-chat",
            "idefics-9b-instruct",
            "deepseek-vl-7b-chat",
            "deepseek-vl-7b-chat-v2",  # modified prompting template
            "step-1v",
        ],
        default=["internvl2"],
        nargs="+",
    )
    parser.add_argument(
        "--max-new-tokens", default=512, type=int, help="Max new tokens for generation."
    )
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    
    parser.add_argument("--prompt", default='ocr', type=str, help="the prompt type with ocr or not")
    
    return parser.parse_args()

def get_prompt(
    args,
    engine,
):
    if "ocr" in args.prompt:
        prompt = "这是一个流程图或者架构图，其中在感兴趣的文本节点周围绘制了红框。\
        请你用红框中的文本内容，结合这个图像的结构，生成一段文本描述这个图，反应图中各个节点和边的关系。"
    else:
        prompt = "请你结合图像中的结构和文本，生成一段针对这个图的描述，要求准确反应图中各个节点和边的关系，语言简洁明了。"
    return prompt
    
def inference_images(
    args,
    engine,
    model,
    tokenizer,
    processor,
    out_path,
):
    results = dict()
    image_names = os.listdir(args.image_dir)
    final_inputs = None
    for image_name in tqdm(image_names):
        # print("the image name is :{}".format(image_name))
        image_path = os.path.join(args.image_dir, image_name)
        prompt = get_prompt(args, engine)
        if "qwen-vl" in engine:
            inputs = [{"text": f"你是一个乐于助人的助手。{prompt}"}]
            inputs.append({"image":f"{image_path}"})
            inputs.append({"text":"User: " + "请你为我生成描述。" + "\nAssistant: "})
            final_inputs = inputs
            total_inputs = tokenizer.from_list_format(inputs)
            inputs = tokenizer(total_inputs, return_tensors="pt")
            inputs = inputs.to(model.device)
            with torch.no_grad():
                pred = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=1,
                )
            input_token_len = inputs["input_ids"].shape[1]
            predicted_answer = tokenizer.decode(
                pred[:, input_token_len:].cpu()[0], skip_special_tokens=True
            )
        elif "internlm-x" in engine:
            image = Image.open(image_path).convert("RGB")
            query_images = []
            query_images.append(model.vis_processor(image))
            # input_text = f"{prompt}"
            input_text = "<ImageHere>"
            input_text += f"{prompt}。\nAnswer:"
            final_inputs = input_text
            query_images = torch.stack(query_images).to(torch.bfloat16).cuda()
            predicted_answer, history = model.chat(
                tokenizer,
                query=input_text,
                image=query_images,
                history=[],
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
            )
        elif "internvl" in engine:
            # set the max number of tiles in `max_num`
            pixel_values = internvl_load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
            generation_config = dict(max_new_tokens=512, do_sample=True)
            # single-image single-round conversation (单图单轮对话)
            question = f'<image>\n{prompt}'
            predicted_answer = model.chat(tokenizer, pixel_values, question, generation_config)
            final_inputs = question
            # print(f'User: {question}\nAssistant: {response}')
        elif "deepseek" in engine:
            from deepseek_vl.utils.io import load_pil_images
            input_text = f"{prompt}"
            conversation = [
                {
                    "role": "User",
                    "content": "<image_placeholder>我在图中感兴趣的文本周围绘制了红框。请你用红框中的文本内容，结合这个图像的结构，生成一段文本描述这个图，反应图中各个节点和边的关系。",
                    "images": [image_path],
                },
                {"role": "Assistant", "content":""}
            ]
            final_inputs = conversation
            pil_images = load_pil_images(conversation)
            prepare_inputs = processor(
                conversations=conversation, images=pil_images, force_batchify=True
            ).to(model.device)

            with torch.no_grad():
                inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
                outputs = model.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=512,
                    do_sample=False,
                    use_cache=True,
                )

                predicted_answer = tokenizer.decode(
                    outputs[0].cpu().tolist(), skip_special_tokens=True
                )

        results[image_name] = {
            "prompt" : final_inputs,
            "prediction" : predicted_answer
        }
            
    return results


if __name__ == "__main__":
    args = parse_args()
    for engine in args.engine:
        print("Loaded model: {}\n".format(engine))
        set_random_seed(args.seed)
        out_path = f"results/{engine}_{args.prompt}.json"
        print("Start evaluating. Output is to be saved to:{}".format(out_path))
        model, tokenizer, processor = load_models.load_i2t_model(engine, args)
        results_dict = inference_images(
            args,
            engine,
            model,
            tokenizer,
            processor,
            out_path,
        )
        with open(out_path, "w", encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=4)
        print(f"Finished evaluating. Output saved to:{out_path}")
        del model, tokenizer, processor
        torch.cuda.empty_cache()
        gc.collect()

