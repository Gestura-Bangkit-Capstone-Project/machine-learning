from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import cv2
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image as mp_Image
from mediapipe import ImageFormat   as mp_ImageFormat
import tensorflow as tf
import numpy as np
from pydantic import BaseModel



# Inisialisasi object FastAPI
app = FastAPI()

#Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Deklarasi Class untuk mengolah input gambar
class input_pipeline:
    def __init__(self, task = os.path.join(os.getcwd(), 'hand_landmarker.task'), model = os.path.join(os.getcwd(), 'model8.keras')):
        #Inisialisasi Model Mediapipe
        base_options = python.BaseOptions(model_asset_path= task)
        options = vision.HandLandmarkerOptions(base_options=base_options,
                                               num_hands=2)
        detector = vision.HandLandmarker.create_from_options(options)
        #Inisialisasi Model machinelearning
        model = tf.keras.models.load_model(model)

        self.detector = detector
        self.model = model

    def __call__(self, frame):
        self.extracted_landmarks = self.end2endProcess(frame)
        return self.extracted_landmarks
    
    def predict_landmarks(self, frame):
        image = mp_Image(image_format = mp_ImageFormat.SRGB, data = frame)

        return self.detector.detect(image)

    
    def adjust_landmarks(self, hand_landmarks, selisih = 0, zeros = False):
        if type(selisih) == int:
            adjusted_landmarks = [np.zeros(3)]
        else :
            adjusted_landmarks = [hand_landmarks[0]]
            
        for i in range(1, len(hand_landmarks)):
            if i % 4 == 0:
                adjusted_landmarks.append([
                    hand_landmarks[i][0] - hand_landmarks[0][0],
                    hand_landmarks[i][1] - hand_landmarks[0][1],
                    hand_landmarks[i][2] - hand_landmarks[0][2]
                ])
            else:
                adjusted_landmarks.append([
                    hand_landmarks[i][0] - hand_landmarks[i - 1][0],
                    hand_landmarks[i][1] - hand_landmarks[i - 1][1],
                    hand_landmarks[i][2] - hand_landmarks[i - 1][2]
                ])

        adjusted_landmarks = np.array(adjusted_landmarks)
        if not zeros:
            adjusted_landmarks = adjusted_landmarks - selisih
            
        return adjusted_landmarks
    
    def extract_keypoints(self, landmark_result):
        lh = np.zeros((21,3))
        rh = np.zeros((21,3))
        
        if landmark_result.handedness is not None:
        #Ekstrak koordinat landmarks
            for index, handedness in enumerate(landmark_result.handedness):
                if handedness[0].display_name == 'Right':
                    rh = [[landmark.x, landmark.y, landmark.z]for landmark in landmark_result.hand_landmarks[index]]
                if handedness[0].display_name == 'Left':
                    lh = [[landmark.x, landmark.y, landmark.z]for landmark in landmark_result.hand_landmarks[index]]

        #Gabungkan Landmarks
        self.lh = self.adjust_landmarks(lh, selisih = np.array(rh[0]), zeros = (np.all(lh == 0)))
        self.rh = self.adjust_landmarks(rh)
        self.concatenated_landmarks = np.concatenate([
            self.lh.flatten(),
            self.rh.flatten()
        ])
        
        #Gabungkan landmarks menjadi satu list
        return self.concatenated_landmarks


    def end2endProcess(self, frame):
        mediapipe_result = self.predict_landmarks(frame) 
        extracted_keypoints = self.extract_keypoints(mediapipe_result)

        return extracted_keypoints

    def predict_sign(self, treshold = 0.95):
        expanded_landmarks = np.expand_dims(self.extracted_landmarks, axis = 0)
        self.prediksi = self.model.predict(expanded_landmarks)
        predicted_class = np.where(self.prediksi[0] >= treshold )[0] if np.where(self.prediksi[0] >= treshold )[0].size > 0 else [26]
        
        return predicted_class, self.prediksi

# Inisialisasi Object input_pipeline
TRANSLATOR = input_pipeline()



@app.get('/')
async def test():
    return JSONResponse(
    content={
        "message": "Berhasil terhubung, silahkan akses endpoint dibawah ini",
        "endpoints": [
            {
                "path": "/sign-to-text",
                "method": "POST",
                "input": {
                    "video": {
                        "type": "file",
                        "format": "mp4",
                        "fps" : 15,  
                        "description": "Video file containing sign language gestures.",
                        "required": True
                    }
                },
                "output": {
                    "message": "type: string",
                    "filename": "type: string",
                    "hasil_prediksi" : "type: string"
                },   
                "description": "Transcribes sign language video into text.",
                "example_request": {
                    "url": "/sign-to-text",
                    "method": "POST",
                    "headers": {
                        "Content-Type": "multipart/form-data"
                    },
                    "body": {
                        "video": ("data: ...", "filename: my_sign_video.mp4", "format: video/mp4")
                    }
                },
                "example_response": {
                    "message": "Video berhasil diunggah!",
                    "filename": "testVideo2.mp4",
                    "hasil_prediksi": "DDDDDAAABBBBE"
                }
            },
            {
                "path": "/text-to-sign",
                "method": "POST",
                "input": {
                    "text": {
                        "type": "string",
                        "max-length" : 20,
                        "description": "Text to be translated into sign language.",
                        "example": "I love coding"
                    }
                },
                "output": {
                    "video": {
                        "type": "file",
                        "format": "mp4",
                        "description": "Video file containing the sign language translation."
                    }
                },
                "description": "Translates text into sign language video.",
                "example_request": {
                    "url": "/text-to-sign",
                    "method": "POST",
                    "headers": {
                        "Content-Type": "application/json"
                    },
                    "body": {
                        "text": "I love coding"
                    }
                },
                "example_response": "Video Example"
            }
        ]
    }
)

# Direktori sementara
TMP_STORAGE_PATH = os.path.join(os.getcwd(), 'tmp')
os.makedirs(TMP_STORAGE_PATH, exist_ok=True)

# Helper function untuk bantuk prediksi
def append_if_same(alphabet_list, target_list):
    if len(set(alphabet_list)) == 1:
        target_list.append(alphabet_list[0])

# Endpoint untuk menerjemahkan bahasa isyarat
alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '']
@app.post("/sign-to-text")
async def upload_video(file: UploadFile = File(...)):
    # Validasi tipe file
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File harus berupa video!")

    # Tentukan nama file tujuan
    file_path = os.path.join(TMP_STORAGE_PATH, file.filename)

    # Simpan file video
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)        

        # Hitung durasi dan FPS video menggunakan OpenCV
        video_capture = cv2.VideoCapture(str(file_path))
        if not video_capture.isOpened():
            raise HTTPException(status_code=500, detail="Gagal membaca video untuk analisis!")
        
        # Prediksi
        luaran_translator = []
        temp_list = []
        while True:
            ret, frame = video_capture.read()

            if not ret:
                break

            TRANSLATOR(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            hasil_predict, array_hasil_predict = TRANSLATOR.predict_sign()
            temp_list.append(alphabet[hasil_predict[0]])
            if len(temp_list) > 4:
                append_if_same(temp_list, luaran_translator)
                temp_list = []
                
        video_capture.release()

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Gagal menghitung durasi video: {str(e)}")
    finally:
         # Hapus file video setelah diproses
        if os.path.exists(file_path):
            os.remove(file_path)

    return JSONResponse(
        content={
            "message": "Video berhasil diunggah!",
            "filename": file.filename,
            "hasil_prediksi" : ''.join(luaran_translator)
        }
    )







# Deklarasi Path ke Asset PNG
asset_path = os.path.join(os.getcwd(), "asset_jari")

# Deklarasi class struktur request body
class body_uploaded_text(BaseModel):
    text : str

# Fungsi untuk membuat video dari gambar PNG dengan durasi yang ditentukan
def create_video_from_images(image_paths : list, outputPath):
    # Tentukan Ukuran frame
    height, width = (626,626)    

    # Buat objek VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Untuk format mp4
    out = cv2.VideoWriter(outputPath, fourcc, 2, (width, height))

    for image_path in image_paths:
        frame = cv2.imread(image_path)
        out.write(frame)  # Menambahkan frame ke video

    out.release()  # Menyelesaikan penulisan video

def delete_video(file_path : str):
    os.remove(file_path)

@app.post("/text-to-sign")
async def upload_text(request : body_uploaded_text, background_task : BackgroundTasks):
    try:

        if not request.text or (len(request.text) > 20) :
            raise HTTPException(status_code=400, detail="text tidak boleh kosong atau lebih dari 20 character")
    
        char_array = list(request.text)
        path_array = [os.path.join(asset_path, f"Gesture-{png.upper()}.png") for png in char_array]

        # Membuat video dari gambar-gambar dengan durasi yang diinginkan
        video_path = os.path.join(TMP_STORAGE_PATH, "text-to-sign.mp4")
        create_video_from_images(path_array, video_path)

        # Membuat variable yang akan dikembalikan ke user
        response = FileResponse(video_path, media_type="video/mp4", filename="output_video.mp4")

        # Menambahkan background task untuk menghapus video setelah mengirim response
        background_task.add_task(delete_video, video_path)

        return response

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"gagal menerjemahkan text: {e}")
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=3000)


# Jalankan server dengan Uvicorn
# uvicorn nama_file:app --reload
