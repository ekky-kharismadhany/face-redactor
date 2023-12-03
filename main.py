from flask import Flask, jsonify, request
from model import Model

app = Flask(__name__)
model = Model("/home/kharismadhany/Documents/work/iot-face-blur/yolov8n_openvino_model/")

@app.route("/")
def index():
    response = {
        "status": "ok"
    }
    return jsonify(response)

@app.route("/v1/image", methods=["POST"])
def blur_image():
    if not request.is_json :
        return jsonify({
            "error": "bad request"
        })
    image = ""
    try:
        image = request.get_json()["image"]
    except KeyError:
        response = jsonify({
            "error": "image key does not exist"
        })
        response.status_code = 400
        return response
    image = model.convert_base64_to_img(image)
    img_result, person_count = model.detect_faces(image)
    return jsonify({
        "face_count": person_count,
        "img_result": img_result
    })

if __name__ == "__main__":
    app.run()