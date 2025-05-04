# app/main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from app.model_utils import load_all_models, preprocess_dicom, predict_all_models

app = FastAPI()

models_dict = load_all_models()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith(".dcm"):
        return JSONResponse(
            content={"error": "Only DICOM (.dcm) files are supported"}, status_code=400
        )

    try:
        contents = await file.read()
        image_tensor = preprocess_dicom(contents)
        predictions = predict_all_models(image_tensor, models_dict)
        return predictions

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
