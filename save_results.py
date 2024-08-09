import os
import json
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

def create_image_with_text(top_text, image_path, bottom_text, output_path):
    # 加载图像
    image = Image.open(image_path)
    width, height = image.size

    # 设置字体（根据需要调整字体大小和类型）
    try:
        # 尝试加载一个特定的字体文件
        font = ImageFont.truetype("arial.ttf", 36)
    except IOError:
        # 如果无法加载特定字体，则使用默认字体
        font = ImageFont.load_default()

    # 为顶部和底部文本计算高度
    draw = ImageDraw.Draw(image)
    top_text_width, top_text_height = draw.textsize(top_text, font=font)
    bottom_text_width, bottom_text_height = draw.textsize(bottom_text, font=font)

    # 创建新图像的大小（高度为原始图像高度加上文本区域的高度）
    total_height = height + top_text_height + bottom_text_height + 20  # 20为文本与图像间的间距
    new_image = Image.new("RGB", (width, total_height), "white")

    # 在新图像上添加顶部文本
    draw = ImageDraw.Draw(new_image)
    top_text_position = ((width - top_text_width) // 2, 0)
    draw.text(top_text_position, top_text, font=font, fill="black")

    # 将原始图像粘贴到新图像的中间部分
    image_position = (0, top_text_height + 10)
    new_image.paste(image, image_position)

    # 在新图像上添加底部文本
    bottom_text_position = ((width - bottom_text_width) // 2, top_text_height + height + 10)
    draw.text(bottom_text_position, bottom_text, font=font, fill="black")

    # 保存新图像
    new_image.save(output_path)

if __name__ == "__main__":
    image_dir = "/home/rsr/gyl/HZBank/ocrres"
    json_path = "results/qwen-vl-chat.json"
    output_path = "visualization/qwen-vl-chat/"
    engine = "qwen-vl-chat"
    with open(json_path, 'r', encoding='utf-8') as f:
        result = json.load(f)
    for key, value in tqdm(result.items()):
        image_path = os.path.join(image_dir, key)
        top_text = value["prompt"]
        bot_text = value["prediction"]
        create_image_with_text(top_text, image_path, bot_text, os.path.join(output_path, "{}_{}.png".format(key.split('.')[0], engine)))

