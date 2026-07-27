
import tqdm, os
import cv2

def slice_video(video_path, destination, rate):
    'slice videos for training data. USE: Input video path, destination folder, frame rate '
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = 0
    basename = os.path.basename(video_path).split('.')[0]
    os.mkdir(destination)
    with tqdm.tqdm(total=total_frames, desc="Saving Frames") as pbar:
        while cap.isOpened():
            filename = f"{basename}_{frame_number}.png"
            file_path = os.path.join(destination, filename)
            success, frame = cap.read()
            if success:
                if (frame_number%rate)==0:
                    cv2.imwrite(file_path, frame)
            pbar.update(1)
            frame_number +=1
            if (frame_number>=total_frames):
                break
    cap.release()