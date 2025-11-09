# 🎭 Selective Face Blur 

This program uses:
✅ **MTCNN** → detects faces in each video frame
✅ **FaceNet (Keras-FaceNet)** → extracts 512D unique face embeddings
✅ **Cosine similarity** → matches each video face to known faces
✅ **Flask + OpenCV** → creates a downloadable blurred output video

---

## ✅ Features

✔ Add a **video in project root(.mp4/.avi/.mov)**
✔ Upload one or more **face images of people to keep visible** inside the known_faces folder
✔ Every *other* face in the video is automatically **blurred**
✔ Output video can be **generated** inside the project folder

---

## 🛠 1. Setup Instructions (VS Code / Windows)

### **Step 1: Clone repo & open in VS Code**

```powershell
git clone https://github.com/shaik-sohel-cyber/secure-face.git
cd <file directory>
```
### **Step 2: Install requirements**
```powershell
pip install -r requirements.txt
```
### **Step 3: Activate Virtual Environment**
```powershell
 .\openvino_secure_env\Scripts\activate
 ```

## ▶ 2. Run the App

```powershell
python main.py -i input.mp4 -o output.mp4 -fg known_faces
```


## 📂 3. Project Structure

```
.
├── main.py                 # Flask backend (Version A – MTCNN + FaceNet)     
├── known_faces/
│   ├──person1 /            # Uploaded video files
│   └── person2/             # Uploaded known face images
├── output.mp4               # Final blurred video
├── my_test_video.mp4        # input video
└── README.md
```



## 🎛 . Important Parameters (configurable in the web form)

| Parameter            | Description                                  |
| -------------------- | -------------------------------------------- |
| **Threshold**        | Controls match strictness. (0.4–0.6 is best) |
| **Blur Type**        | Gaussian blur (default) or pixel blur        |
| **Video Resolution** | 720p / 1080p supported                       |
| **Faces per Person** | 1–3 face images gives better accuracy        |

---


