from sahi.slicing import slice_coco
import os
import shutil
import tqdm
from sahi.utils.file import load_json
import numpy as np

def slice_coco_DS(input_path, output_path, slice_h=800, slice_w=800, overlap_h=0.2, overlap_w=0.2):
    annotation_path = os.path.join(input_path,'_annotations.coco.json')
    coco_dict, coco_path = slice_coco(
        coco_annotation_file_path=annotation_path,
        image_dir=input_path,
        output_coco_annotation_file_name="_annotations.coco.json",
        ignore_negative_samples=False,
        output_dir=output_path,
        slice_height=slice_h,
        slice_width=slice_w,
        overlap_height_ratio=overlap_h,
        overlap_width_ratio=overlap_w,
        min_area_ratio=0.1,
        verbose=False
    )

def bbox_coco_to_yolo(A, img_width, img_height):    
    'convert from "X0 Y0 W H" to "Cx Cy W D"'
    W = A[2] 
    H = A[3] 
    Cx = (A[0] + (W / 2)) 
    Cy = (A[1] + (H / 2)) 
    return [(Cx / img_width), (Cy / img_height),(W / img_width),(H / img_height)]

def convert_coco_to_yolo(path, destination):
    'Convert coco to yoolo dataset. Include Origin and destination folders'
    label_dir = os.path.join(destination,'labels')
    image_dir = os.path.join(destination,'images')
    if os.path.exists(destination):
        print('directory exist')
        return
    os.mkdir(destination)
    os.mkdir(label_dir)
    os.mkdir(image_dir)
    coco_path = os.path.join(path, "_annotations.coco.json")
    coco_dict = load_json(coco_path)
    for item in tqdm.tqdm(coco_dict['images']):
        
        lista = []
        image_name = item['file_name']
        label_name = image_name.replace('.jpg','.txt').replace('.png','.txt')
        image_origin_path = os.path.join(path,image_name)
        image_destin_path = os.path.join(image_dir,image_name)
        shutil.copy(image_origin_path,image_destin_path)
        label_path = os.path.join(label_dir,label_name)
        width = item['width']
        height = item['height']
        id = item['id']
        bboxes = [bb for bb in coco_dict['annotations'] if bb['image_id']==id]
        for bb in bboxes:
            cat = bb['category_id']
            bbox = bb['bbox']
            bbox = bbox_coco_to_yolo(bbox, width, height)
            bbox.insert(0,str(cat-1))
            lista.append(bbox)
        np.savetxt(label_path,np.array(lista), fmt='%s')