from ultralytics import YOLO
model = YOLO("yolo26n.pt")
model.train(data="dataset/data.yaml", epochs=100, imgsz=640)


# from ultralytics import YOLO
# model = YOLO("/home/vikbot/Documents/label-inspection/runs/detect/train-2/weights/last.pt")
# model.train(resume=True)
