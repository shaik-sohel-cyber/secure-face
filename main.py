import cv2
import os
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
import argparse

# Initialize detector and FaceNet embedder
detector = MTCNN()
embedder = FaceNet()

# 1️⃣ Load known faces and compute embeddings
def load_known_faces(known_faces_dir):
    known_embeddings = []
    known_names = []

    for person in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, person)

        if not os.path.isdir(person_dir):
            continue

        for image_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, image_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(img_rgb)

            if faces:
                x, y, w, h = faces[0]['box']
                x, y = abs(x), abs(y)
                face = img_rgb[y:y+h, x:x+w]

                face = cv2.resize(face, (160, 160))
                face = np.expand_dims(face, axis=0)

                embedding = embedder.embeddings(face)[0]
                known_embeddings.append(embedding)
                known_names.append(person)
                print(f"[INFO] Loaded {image_name} for {person}")

    return known_embeddings, known_names

# 2️⃣ Cosine similarity function
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 3️⃣ Process video - blur unknown faces
def process_video(input_path, output_path, known_embeddings, known_names, threshold=0.50):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        output_path, fourcc, 
        int(cap.get(cv2.CAP_PROP_FPS)),
        (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    )

    print("[INFO] Processing video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb_frame)

        for face in faces:
            x, y, w, h = face['box']
            x, y = abs(x), abs(y)
            face_rgb = rgb_frame[y:y+h, x:x+w]

            if face_rgb.size == 0:
                continue

            face_resized = cv2.resize(face_rgb, (160, 160))
            face_resized = np.expand_dims(face_resized, axis=0)

            emb = embedder.embeddings(face_resized)[0]

            best_match = "Unknown"
            best_score = 0

            for stored_emb, name in zip(known_embeddings, known_names):
                score = cosine_similarity(emb, stored_emb)
                if score > best_score:
                    best_score = score
                    best_match = name

            if best_score < threshold:
                blurred = cv2.GaussianBlur(frame[y:y+h, x:x+w], (51, 51), 30)
                frame[y:y+h, x:x+w] = blurred
                cv2.putText(frame, "Unknown - Blurred", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.putText(frame, f"Allowed: {best_match}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)
        cv2.imshow("Output", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("[INFO] Saved output to:", output_path)

# 4️⃣ Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blur unknown people using FaceNet + MTCNN")
    parser.add_argument("-i", "--input", required=True, help="Input video path")
    parser.add_argument("-o", "--output", default="output.mp4", help="Output video path")
    parser.add_argument("-fg", "--face_gallery", required=True, help="Folder with known faces")
    parser.add_argument("-t", "--threshold", type=float, default=0.50, help="Matching threshold")
    args = parser.parse_args()

    known_embeddings, known_names = load_known_faces(args.face_gallery)

    if len(known_embeddings) == 0:
        print("[ERROR] No known faces found.")
        exit()

    process_video(args.input, args.output, known_embeddings, known_names, args.threshold)
