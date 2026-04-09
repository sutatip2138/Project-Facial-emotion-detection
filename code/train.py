from ultralytics import YOLO

def train_model():
    model = YOLO('yolov8n.pt')
    model.train(data=(r'C:\Users\Admin\Desktop\pro ject\emo2.2.v7i.yolov11\data.yaml'),
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
    )

if __name__ == '__main__':
    train_model()