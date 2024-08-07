import torch
import os
import json
import argparse
import gc
from eval_utils import load_models
from utils import *
from tqdm import tqdm
import time
# ppocr
from paddleocr import PaddleOCR, draw_ocr

def parse_args():
    parser = argparse.ArgumentParser(description="Flowchart Inference")
    parser.add_argument(
        "--image_dir",
        default="None",
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
        "--max-new-tokens", default=15, type=int, help="Max new tokens for generation."
    )
    return parser.parse_args()

def get_prompt(
    args,
    engine,
):
    prompt = ""
    return prompt

def PaddleOCR(image_path):
    def mergeboxes(boxes, texts):
        n = len(boxes)
        for i in range(n):
            for j in range(n):
                centric_i, centric_j = (), ()

    # ocr模型识别文本框
    ocr = PaddleOCR(use_angle_cls=True, lang="ch")
    result = ocr.ocr(image_path, cls=True)
    # 转换坐标为四元组(left, up, right, down)
    # 记录text
    boxes = []
    texts = []
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            texts.append(line[1][0])
            left, up, right, down = line[0][0], line[0][1], line[2][0], line[2][1]
            boxes.append((left,up,right,down))
    # 合并距离过近的text
                
        

def inference_images(
    args,
    engine,
    model,
    tokenizer,
    processor,
    out_path,
):
    image_names = os.listdir(args.image_dir)
    for image_name in image_names:
        image_path = os.path.join(args.image_dir, image_name)
    pass

if __name__ == "__main__":
    args = parse_args()
    for engine in args.engine:
        model, tokenizer, processor = load_models.load_i2t_model(engine, args)
        print("Loaded model: {}\n".format(engine))
        set_random_seed(args.seed)
        out_path = f"results/{engine}.json"
        print("Start evaluating. Output is to be saved to:{}".format(out_path))
        results_dict = inference_images(
            args,
            engine,
            model,
            tokenizer,
            processor,
            out_path,
        )
        with open(out_path, "w") as f:
            json.dump(results_dict, f, indent=4)
        print(f"Finished evaluating. Output saved to:{out_path}")
        del model, tokenizer, processor
        torch.cuda.empty_cache()
        gc.collect()

