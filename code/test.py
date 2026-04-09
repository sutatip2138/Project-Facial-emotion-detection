import cv2
import json
import base64
import time
import paho.mqtt.client as mqtt
from ultralytics import YOLO
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

MQTT_BROKER = "127.0.0.1"
MQTT_PORT = 1883
TOPIC_CAMERA = "camera"
TOPIC_EMOTION = "Emotion"
TOPIC_SELECT = "camera_select"

# ==================== Google Sheets Setup ====================
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("service_account.json", scope)
client_gs = gspread.authorize(creds)
sheet = client_gs.open("Emotion_Detection_Data").sheet1

# เพิ่ม Header ถ้ายังไม่มี
if sheet.row_values(1) == []:
    sheet.append_row(["Timestamp", "Emotion", "Emoji", "Happy Count", "Sad Count", "Angry Count", "Neutral Count"])
    print("สร้าง Header ใน Google Sheets แล้ว")
# =============================================================

current_source = "esp32"
cap = None
need_switch = False
ready = False

def open_camera():
    global cap, current_source
    if cap is not None:
        cap.release()
        cap = None
    time.sleep(2)
    if current_source == "webcam":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("เปิด Webcam")
    else:
        cap = cv2.VideoCapture("http://192.168.137.125:81/stream")
        print("เปิด ESP32-CAM")

def save_to_sheet(label, emoji, counts):
    """บันทึก raw data ลง Google Sheets"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [
            timestamp,
            label,
            emoji,
            counts["😊"],
            counts["😢"],
            counts["😡"],
            counts["😐"]
        ]
        sheet.append_row(row)
        print(f"บันทึกลง Google Sheets: {row}")
    except Exception as e:
        print(f"บันทึก Google Sheets error: {e}")

def on_message(client, userdata, msg):
    global current_source, need_switch, ready
    if not ready:
        print("ยังไม่พร้อม ละเว้น")
        return
    if msg.topic == TOPIC_SELECT:
        new_source = msg.payload.decode().strip()
        print(f"รับค่า: '{new_source}' | ปัจจุบัน: '{current_source}'")
        if new_source != current_source:
            current_source = new_source
            need_switch = True
            print(f"กำลังสลับไป {current_source}...")
        else:
            print("กล้องเดิมอยู่แล้ว ไม่สลับ")

mqtt_client = mqtt.Client()
mqtt_client.on_message = on_message
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.subscribe(TOPIC_SELECT)
mqtt_client.loop_start()

model = YOLO(r"C:\Users\Admin\Desktop\pro ject\runs\detect\train24\weights\best.pt")

emoji_map = {
    "happy": "😊",
    "sad": "😢",
    "angry": "😡",
    "neutral": "😐"
}

emotion_counts = {
    "😊": 0,
    "😢": 0,
    "😡": 0,
    "😐": 0
}

last_detected = ""
open_camera()

print("รอ 5 วินาที...")
time.sleep(5)
ready = True
print("พร้อมรับค่าแล้ว!")

frame_count = 0
last_send = time.time()

print("Emotion System Start")

while True:
    if need_switch:
        need_switch = False
        open_camera()
        continue

    if cap is None or not cap.isOpened():
        time.sleep(0.1)
        continue

    ret, frame = cap.read()
    if not ret:
        time.sleep(0.3)
        continue

    frame_count += 1
    display = frame.copy()

    if frame_count % 3 == 0:
        results = model(frame, verbose=False)
        for r in results:
            display = r.plot()
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf > 0.6:
                    class_id = int(box.cls[0])
                    label = model.names[class_id].lower()
                    emoji = emoji_map.get(label, label)
                    if emoji != last_detected:
                        last_detected = emoji
                        emotion_counts[emoji] += 1

                        payload = {
                            "emoji": emoji,
                            "happy_count": emotion_counts["😊"],
                            "sad_count": emotion_counts["😢"],
                            "angry_count": emotion_counts["😡"],
                            "neutral_count": emotion_counts["😐"]
                        }
                        mqtt_client.publish(TOPIC_EMOTION, json.dumps(payload))
                        print("Detected:", emoji, emotion_counts)

                        # ==================== บันทึกลง Google Sheets ====================
                        save_to_sheet(label, emoji, emotion_counts)
                        # ================================================================

    if time.time() - last_send > 0.1:
        try:
            small = cv2.resize(display, (480, 360))
            _, buffer = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 60])
            jpg = base64.b64encode(buffer).decode()
            mqtt_client.publish(TOPIC_CAMERA, json.dumps({"image": jpg}))
            last_send = time.time()
        except Exception as e:
            print("ส่งภาพ error:", e)

if cap is not None:
    cap.release()
cv2.destroyAllWindows()
mqtt_client.loop_stop()
mqtt_client.disconnect()