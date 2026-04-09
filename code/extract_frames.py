import cv2
import os

# --- ตั้งค่าตรงนี้ ---
video_name = 'videos/movie_16.mp4' # เปลี่ยนเป็นชื่อไฟล์วิดีโอของคุณทิพย์
output_name = 'dataset_raw5' # ชื่อโฟลเดอร์ที่จะเก็บรูป
every_x_frames = 24 # ยิ่งเลขน้อย ยิ่งได้รูปเยอะ (เช่น ทุกๆ 15 เฟรม)
# ------------------

cap = cv2.VideoCapture(video_name)
if not os.path.exists(output_name):
    os.makedirs(output_name)

count = 0
saved = 0

print("กำลังเริ่มตัดรูป... กรุณารอสักครู่")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # ถ้าครบรอบเฟรมที่ตั้งไว้ ให้เซฟรูป
    if count % every_x_frames == 0:
        file_path = f"{output_name}/frame1_{saved}.jpg"
        cv2.imwrite(file_path, frame)
        saved += 1
    
    count += 1

cap.release()
print(f"เสร็จเรียบร้อย! ได้รูปทั้งหมด {saved} รูป อยู่ในโฟลเดอร์ {output_name}")