from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib

app = FastAPI(title="ML Model Inference")

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)

templates = Jinja2Templates(directory="template")

model = joblib.load(
    r"C:\Users\krishankant\Downloads\my_model.pkl"
)


class PredictionRequest(BaseModel):
    text: str


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.post("/predict")
async def predict(data: PredictionRequest):

    prediction = model.predict([data.text])

    # Convert numpy value to normal Python value
    prediction_value = prediction[0]
   

    return {
               "prediction": str(prediction_value) 
    }
