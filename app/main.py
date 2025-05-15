# app/main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from app.model_utils import load_all_models, preprocess_dicom, predict_all_models
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


models_dict = load_all_models()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith(".dcm"):
        return JSONResponse(
            content={"error": "Only DICOM (.dcm) files are supported"}, status_code=400
        )

    try:
        contents = await file.read()
        print("Received file, size:", len(contents))  #Debug print
        image_tensor = preprocess_dicom(contents)
        print("Preprocessing successful")  #Debug print
        predictions = predict_all_models(image_tensor, models_dict)
        print("Prediction successful")  # Debug print
        return predictions

    except Exception as e:
        print("Exception caught:", str(e))  # Print full error in terminal
        return JSONResponse(content={"error": str(e)}, status_code=500)

