from eval_utils import load_models

engines = ["qwen-vl", "internlm-x2", "deepseek-vl-7b-chat"]

for engine in engines:
    print("download {}".format(engine))
    load_models.load_i2t_model(engine, args=None)