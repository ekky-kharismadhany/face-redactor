from ultralytics import YOLO
import cv2
import base64
from PIL import Image
from io import BytesIO
import numpy as np

class Model:
    def __init__(self, model_path: str):
        print(model_path)
        self.model_weight = YOLO(model_path, task="detect")

    def convert_base64_to_img(self, img_base64: str) -> cv2.UMat:
        img_decoded = base64.b64decode(img_base64)
        img = Image.open(BytesIO(img_decoded))
        return cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)

    def detect_faces(self, image: cv2.UMat) -> tuple[str, int]:
        person_boxes = []
        person_count = 0
        results = self.model_weight(image)
        for result in results:
            for box in result.boxes:
                if result.names[0] == "face":
                    person_count += 1
                    person_boxes.append(box)
        for box in person_boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            roi = image[int(y1):int(y2), int(x1):int(x2)]
            redacted_face = (255, 255, 255)
            roi[:, :] = redacted_face
            image[int(y1):int(y2), int(x1):int(x2)] = roi
        _, buffer = cv2.imencode(".jpg", image)
        jpg_as_txt = base64.b64encode(buffer).decode("utf-8")
        return jpg_as_txt, person_count