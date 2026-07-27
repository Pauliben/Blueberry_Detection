import numpy as np



def bbox_coco_to_yolo(A, img_width, img_height):    
    'convert from "X0 Y0 W H" to "Cx Cy W D"'
    W = A[2] 
    H = A[3] 
    Cx = (A[0] + (W / 2)) 
    Cy = (A[1] + (H / 2)) 
    return [(Cx / img_width), (Cy / img_height),(W / img_width),(H / img_height)]

def save_sahi_result(res, dest_txt_path):
    'save sahi results Bboxes in yolo format'
    bboxes = res.object_prediction_list
    arr = []
    img_h = res.image_height
    img_w = res.image_width
    for box in bboxes:
        xywh = box.bbox.to_xywh()
        x = xywh[0]
        y = xywh[1]
        w = xywh[2]
        h = xywh[3]
        
        cx = (x + (x+w))/2
        cy = (y + (y+h))/2
      
        cx = cx / img_w
        cy = cy / img_h
        w = w / img_w
        h = h / img_h
      
        cx = format(cx, '.6f')
        cy = format(cy, '.6f')
        w = format(w, '.6f')
        h = format(h, '.6f')
        
        arr.append([box.category.id,cx,cy,w,h])
    np.savetxt(dest_txt_path,np.array(arr), fmt='%s')