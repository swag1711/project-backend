import os
import uuid
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from model_loader.load_models import image_model
from preprocess.preprocess_image import new_prediction
 
app = FastAPI()
 
 
origins = ["*"]
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)
 
 
 
# Create the 'uploads' directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)
 
@app.get("/")
def home():
    return {"message": "Welcome to the home route!"}
 
 
 
 
# Set a threshold for classification (adjust as needed)
threshold = 0.5

@ app.post("/preprocess-image/")
async def preprocess_image(file: UploadFile = File(...)):
    try:
        IMG_SIZE = 128
        
        # Read the uploaded image file
        contents = await file.read()
        
        # Convert the file contents to numpy array for preprocessing
        nparr = np.frombuffer(contents, np.uint8)
        
        # Decode the numpy array into an image using cv2
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        # Preprocess the image
        img_array = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        new_array = img_array.reshape(1, IMG_SIZE, IMG_SIZE, 1)
        predict_model = image_model(new_array)
        print("predict",predict_model)
        predict =np.argmax(predict_model)
        #print(predict)
        if predict==0:
            print("YOU ARE IN NORMAL CONDITION NO NEED TO WORRY ABOUT")
            return {"message":"YOU ARE IN NORMAL CONDITION NO NEED TO WORRY ABOUT","probability":predict.tolist()}
    
        elif predict== 1:
            print("Benign the cells are not yet cancerous, but they have the potential to become malignant consult the doctor")
            return {"message":"Benign the cells are not yet cancerous, but they have the potential to become malignant consult the doctor","probability":predict.tolist()}
        else:
            print("Malignant the tumors are cancerous. The cells can grow and spread to other parts of the body.")
            return {"message":"Malignant the tumors are cancerous. The cells can grow and spread to other parts of the body.","probability":predict.tolist()}


        # Return the preprocessed image
        # return {"message": "Image preprocessed successfully", "processed_image": new_array.tolist()}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}
 
 
# This route predicts using the audio file uploaded to the server. The audio file is processed and the prediction is returned as a JSON response.
@app.post("/cancer-prediction-local")
async def cancerPredictionLocal(files: List[UploadFile] = File(...)):
    prediction_results = []
    for file in files:
        print(file)
             # Convert the file contents to numpy array for preprocessing
        nparr = np.frombuffer(file, np.uint8)
        print("arr",nparr)
        # Decode the numpy array into an image using cv2
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        print("img",img)
        unique_id = str(uuid.uuid4())
        # Get the file extension from the original filename
        file_extension = os.path.splitext(file.filename)[1]
        # Construct the new file path with the unique filename and original extension
        file_path = os.path.join("uploads", unique_id + file_extension)
        # Save the uploaded file to the directory
        with open(file_path, "wb") as f:
            f.write(await file.read())

        data = new_prediction(img)
        print("image",data)
        predict = image_model.predict(data)
        print("predict",predict)
        done ='DONE'
    return {"result": done}
 
