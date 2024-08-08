from paddleocr import PaddleOCR, draw_ocr
import os
import cv2
import math
from tqdm import tqdm

def OCR_Detect(image_path, ocr):
    def distance(point1, point2):
        # left,up,right,down,计算两个中心点坐标
        # 点坐标用(x,y)表示
        centric1 = ((point1[0]+point1[2])/2, (point1[1]+point1[3])/2) 
        centric2 = ((point2[0]+point2[2])/2, (point2[1]+point2[3])/2)
        dist = math.sqrt((centric2[0]-centric1[0])**2 + (centric2[1]-centric2[1])**2)
        return dist

    def mergeboxes(boxes, texts):
        i = 0
        merged = True
        while merged:
            merged = False
            i = 0
            while i < len(boxes):
                j = i+1
                while j < len(boxes):
                    dist = distance(boxes[i], boxes[j])
                    # print("the distance between points {}".format(dist))
                    if distance(boxes[i], boxes[j]) <= 3:
                        # 合并
                        left = min(boxes[i][0], boxes[j][0])
                        up = min(boxes[i][1], boxes[j][1])
                        right = max(boxes[i][2], boxes[j][2])
                        down = max(boxes[i][3], boxes[j][3])
                        boxes[i] = (left, up, right, down)
                        texts[i] = texts[i] + texts[j]
                        del boxes[j]
                        del texts[j]
                        merged = True
                        break
                    j += 1
                if merged:
                    break
                i += 1
        return boxes, texts

    # ocr模型识别文本框
    result = ocr.ocr(image_path, cls=True)
    # 转换坐标为四元组(left, up, right, down)
    # 记录text
    boxes = []
    texts = []
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            texts.append(line[1][0])
            left, up, right, down = line[0][0][0], line[0][0][1], line[0][2][0], line[0][2][1]
            boxes.append((left,up,right,down))
    # 合并距离过近的text
    # boxes, texts = mergeboxes(boxes, texts)
    return boxes, texts

if __name__ == "__main__":
    ocr_dir = "/home/rsr/gyl/HZBank/ocrres"
    img_dir = "/home/rsr/gyl/HZBank/new_archs"
    image_names = os.listdir(img_dir)
    ocr = PaddleOCR(use_angle_cls=True, lang="ch")
    for image_name in tqdm(image_names):
        image_path = os.path.join(img_dir, image_name)
        boxes, texts = OCR_Detect(image_path, ocr)
        image = cv2.imread(image_path)
        for (x1,y1,x2,y2) in boxes:
            cv2.rectangle(image, (int(x1),int(y1)), (int(x2),int(y2)), (0,0,255), 2)
        save_path = os.path.join(ocr_dir, image_name)
        cv2.imwrite(save_path, image)