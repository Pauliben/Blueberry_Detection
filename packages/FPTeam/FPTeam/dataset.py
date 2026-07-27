from sahi.slicing import slice_coco
import os
import shutil
import tqdm
from sahi.utils.file import load_json
import numpy as np
import json
def slice_coco_DS(input_path, output_path, slice_h=800, slice_w=800, overlap_h=0.2, overlap_w=0.2):
    'Wrapper for slicing coco dataset using sahi framework'
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
        
        
def filter_annotations_by_area(dataset_dir, area_threshold=1.0):
    # Define the input file path
    json_file = f'{dataset_dir}/train/_annotations.coco.json'

    # Load the COCO annotations from the input file
    with open(json_file, 'r') as file:
        coco_data = json.load(file)

    # Filter out annotations with areas below the threshold
    filtered_annotations = [
        annotation for annotation in coco_data['annotations'] if annotation['area'] >= area_threshold
    ]
    prev_count = len(coco_data['annotations'])

    # Update the COCO data with the filtered annotations
    coco_data['annotations'] = filtered_annotations

    # Save the updated annotations to a new file
    with open(json_file, 'w') as file:
        json.dump(coco_data, file)

    # Print summary of filtering
    print(f"Number of annotations before filtering: {prev_count}")
    print(f"Number of annotations after filtering: {len(filtered_annotations)}")
    print(f"Filtered annotations is overwritten to {json_file}")