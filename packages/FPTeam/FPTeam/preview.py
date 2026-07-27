from PIL import Image, ImageDraw
from sahi.utils.file import load_json
import os

def coco_from_path(path): #X0Y0WH
    'Draw BBoxes from Coco image path dataset.'
    img = Image.open(path).convert('RGBA')
    name = os.path.basename(path)
    print(name)
    coco_path = os.path.join(os.path.dirname(path), "_annotations.coco.json")
    coco_dict = load_json(coco_path)
    item = [item for item in coco_dict['images'] if item['file_name']==name]
    boxes = [bb['bbox'] for bb in coco_dict['annotations'] if bb['image_id']==item[0]['id']]
    for bb in boxes:
        xyxy = [bb[0], bb[1], bb[0]+bb[2], bb[1]+bb[3]]
        ImageDraw.Draw(img).rectangle(xyxy, width=5)    
    img.show()

def sahi_from_resultOBJ(res):
    'Draw BBoxes from sahi result object.'
    img = res.image
    labels = res.object_prediction_list
    for l in labels:
        xyxy = l.bbox.to_xyxy()
        cat = str(l.category.id)
        if cat=='0':
            out = (0,255,0)
        elif cat=='1':
            out = (255,0,0)
        else:
            out = (255,255,255)
        ImageDraw.Draw(img).rectangle(xyxy, width=4, outline=out)
    img.show()

def yolo_from_path(path): #CxCyWH
    'Draw BBoxes from yolo image path dataset.'
    img = Image.open(path).convert('RGBA')
    img_width = img.width
    img_height = img.height
    label_path = path.replace('/images/','/labels/').replace('.jpg','.txt').replace('.png','.txt')
    label = open(label_path,'r')
    for l in label:
        data = l.split()
        cat = data[0]
        Cx = float(data[1])*img_width
        Cy = float(data[2])*img_height
        w = float(data[3])*img_width
        h = float(data[4])*img_height
        xyxy = [int(Cx - w/2),int(Cy - h/2),int(Cx + w/2),int(Cy + h/2)]
        if cat=='0':
            out = (0,255,0)
        elif cat=='1':
            out = (255,0,0)
        else:
            out = (255,255,255)
        ImageDraw.Draw(img).rectangle(xyxy, width=4,  outline=out)    
    img.show()
        