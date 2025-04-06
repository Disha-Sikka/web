from flask import Flask, request, jsonify
import cv2
import numpy as np
import face_recognition
import base64

app = Flask(__name__)

def decode_base64_image(data):
    try:
        image_data = base64.b64decode(data)
        np_arr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print("Decoding error:", e)
        return None

def compare_faces(voter_img, live_img):
    try:
        voter_encodings = face_recognition.face_encodings(voter_img)
        live_encodings = face_recognition.face_encodings(live_img)

        if not voter_encodings or not live_encodings:
            return False, "No face detected in one or both images"

        result = face_recognition.compare_faces([voter_encodings[0]], live_encodings[0])[0]
        return result, None if result else "Face does not match"
    except Exception as e:
        return False, str(e)

@app.route("/verify", methods=["POST"])
def verify():
    data = request.get_json()
    if not data or "voter_id" not in data or "live_face" not in data:
        return jsonify({"status": "error", "reason": "Missing images"}), 400

    voter_img = decode_base64_image(data["voter_id"])
    live_img = decode_base64_image(data["live_face"])

    if voter_img is None or live_img is None:
        return jsonify({"status": "error", "reason": "Invalid image format"}), 400

    match, reason = compare_faces(voter_img, live_img)
    if match:
        return jsonify({"status": "success"})
    else:
        return jsonify({"status": "failure", "reason": reason}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
