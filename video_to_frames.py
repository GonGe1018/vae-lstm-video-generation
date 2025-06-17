import cv2
import os

video_dir = './walking'
frame_save_dir = './walking_frames'
os.makedirs(frame_save_dir, exist_ok=True)

resize_dim = (64, 64)  # 원하는 해상도
frame_skip = 2  # 2이면 25fps→12.5fps (매 2프레임마다 1장 저장)

for video_name in os.listdir(video_dir):
    if video_name.endswith('.avi'):
        video_path = os.path.join(video_dir, video_name)
        cap = cv2.VideoCapture(video_path)
        count = 0
        saved_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if count % frame_skip == 0:
                frame = cv2.resize(frame, resize_dim)
                # 흑백 변환 원하면 아래 주석 해제
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                save_path = os.path.join(
                    frame_save_dir,
                    f"{os.path.splitext(video_name)[0]}_frame{saved_count:03d}.png"
                )
                cv2.imwrite(save_path, frame)
                saved_count += 1
            count += 1
        cap.release()
