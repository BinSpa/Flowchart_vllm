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
            "emu2-chat",
            "idefics-9b-instruct",
            "deepseek-vl-7b-chat",
            "deepseek-vl-7b-chat-v2",  # modified prompting template
            "step-1v",
        ],
        default=["qwen-vl-chat"],
        nargs="+",
    )
    parser.add_argument(
        "--max-new-tokens", default=512, type=int, help="Max new tokens for generation."
    )
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    
    return parser.parse_args()

def get_prompt(
    args,
    engine,
):
    prompt = "我将提供一个流程图或者架构图，其中在感兴趣的文本节点周围绘制了红框。请你用红框中的文本内容，结合这个图像的结构，生成一段文本描述这个图，反应图中各个节点和边的关系。"
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
        prompt = get_prompt(engine, args)
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
            query_image = model.vis_process(image)
            input_text = f"{prompt}"
            input_text = "<ImageHere>"
            input_text += f"请你为我生成描述。\nAnswer:"
            final_inputs = input_text
            query_image = torch.stack(query_image).to(torch.bfloat16).cuda()
            predicted_answer, history = model.chat(
                tokenizer,
                query=input_text,
                image=query_image,
                history=[],
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
            )
        elif "deepseek" in engine:
            from deepseek_vl.utils.io import load_pil_images
            input_text = f"{prompt}"
            conversation = [
                {
                    "role": "User",
                    "content": prompt,
                }
            ]
            conversation += [
                {
                    "role": "User",
                    "content": f"",
                    "images": [f""],
                },
                {"role": "Assistant", "content":""}
            ]
            final_inputs = conversation
            pil_images = load_pil_images(conversation)
            prepare_inputs = processor(
                conversation=conversation, images=pil_images, force_batchify=True
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
        out_path = f"results/{engine}.json"
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
            json.dump(results_dict, f, indent=4)
        print(f"Finished evaluating. Output saved to:{out_path}")
        del model, tokenizer, processor
        torch.cuda.empty_cache()
        gc.collect()

