import os
import shutil
from deepface import DeepFace

# --- ตั้งค่าพาธ ---
input_folder = 'dataset_raw5'        # โฟลเดอร์รูปต้นฉบับ
output_base_folder = 'classified_faces5'

# คลาสเป้าหมาย
target_classes = {
    'happy': 'happy',
    'sad': 'sad',
    'angry': 'angry',
    'neutral': 'normal'
}

# สร้างโฟลเดอร์สำหรับ 4 คลาส + 1 โฟลเดอร์สำหรับ "ไม่ใช่หน้าคน"
all_folders = ['happy', 'sad', 'angry', 'normal', 'unknown']
for folder in all_folders:
    os.makedirs(os.path.join(output_base_folder, folder), exist_ok=True)

# ตัวนับเลข
counters = {f: 0 for f in all_folders}

print("--- เริ่มต้นการคัดแยก (รวมโหมดตรวจจับใบหน้า) ---")

for filename in os.listdir(input_folder):
    if filename.lower().endswith(".jpg"):
        img_path = os.path.join(input_folder, filename)
        
        try:
            # ใช้ enforce_detection=True เพื่อให้มันฟ้องถ้าหาหน้าไม่เจอ
            results = DeepFace.analyze(img_path, actions=['emotion'], enforce_detection=True, silent=True)
            
            # ถ้าผ่านบรรทัดบนมาได้ แสดงว่า "เจอใบหน้า"
            raw_emotion = results[0]['dominant_emotion']
            
            # ตรวจสอบคลาสอารมณ์
            if raw_emotion in target_classes:
                label = target_classes[raw_emotion]
            else:
                # กรณีเป็นอารมณ์อื่นที่ไม่ได้สั่งไว้ (เช่น surprised, fear) ให้ลง unknown หรือสร้างเพิ่มก็ได้
                label = 'unknown'
        
        except ValueError:
            # DeepFace จะส่ง ValueError ออกมาถ้าหาใบหน้าไม่เจอเลย (Face could not be detected)
            label = 'unknown'
            print(f"[-] ไม่พบใบหน้าในรูป: {filename} -> ย้ายไป unknown")
        
        except Exception as e:
            label = 'unknown'
            print(f"[!] เกิดข้อผิดพลาดกับ {filename}: {e}")

        # จัดการบันทึกไฟล์
        counters[label] += 1
        new_name = f"{label}{counters[label]:05d}.jpg"
        destination = os.path.join(output_base_folder, label, new_name)
        
        shutil.copy(img_path, destination)
        if label != 'unknown':
            print(f"[+] สำเร็จ: {filename} -> {label}/{new_name}")

print("\n" + "="*30)
print("สรุปการทำงาน:")
for cls, count in counters.items():
    print(f"- {cls}: {count} รูป")