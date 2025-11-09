from ultralytics import YOLO


model = YOLO("../models/v0-1.pt")


def predict(img):

    results = model.predict(source=img, save=False, conf=0.3, verbose=False)
    return results
