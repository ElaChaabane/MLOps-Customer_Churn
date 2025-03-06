from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Create folders for templates and static files if they don't exist
if not os.path.exists('templates'):
    os.makedirs('templates')
if not os.path.exists('static'):
    os.makedirs('static')

# Load the trained model
model = joblib.load("churn_model.joblib")

# Initialize FastAPI app
app = FastAPI(title="Customer Churn Prediction API")

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your domain(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up templates
templates = Jinja2Templates(directory="templates")

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define input data model with 20 fields
class InputData(BaseModel):
    account_length: int
    number_vmail_messages: int
    total_day_minutes: float
    total_day_calls: int
    total_eve_minutes: float
    total_eve_calls: int
    total_night_minutes: float
    total_night_calls: int
    total_intl_minutes: float
    total_intl_calls: int
    customer_service_calls: int
    international_plan: int
    voice_mail_plan: int
    state_0: int
    state_1: int
    state_2: int
    state_3: int
    state_4: int
    state_5: int
    state_6: int

# Home page route
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# API prediction endpoint
@app.post("/api/predict")
def predict_api(data: InputData):
    # Create a feature list in the order used during training:
    features = [
        data.account_length,
        data.number_vmail_messages,
        data.total_day_minutes,
        data.total_day_calls,
        data.total_eve_minutes,
        data.total_eve_calls,
        data.total_night_minutes,
        data.total_night_calls,
        data.total_intl_minutes,
        data.total_intl_calls,
        data.customer_service_calls,
        data.international_plan,
        data.voice_mail_plan,
        data.state_0,
        data.state_1,
        data.state_2,
        data.state_3,
        data.state_4,
        data.state_5,
        data.state_6,
    ]
    X = np.array(features).reshape(1, -1)
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]  # Probability of churn
    result = "Customer Will Churn" if prediction == 1 else "Customer Will Not Churn"
    return {
        "prediction": result,
        "probability": round(float(probability), 3) * 100,
        "is_churn": bool(prediction)
    }

# Web form prediction endpoint
@app.post("/predict")
async def predict_form(request: Request):
    try:
        form_data = await request.form()
        
        # Get state value and create state encoding
        state_value = int(form_data.get("state"))
        state_encoding = [0] * 7
        state_encoding[state_value] = 1
        
        # Create input data object
        input_data = InputData(
            account_length=int(form_data.get("account_length")),
            number_vmail_messages=int(form_data.get("number_vmail_messages")),
            total_day_minutes=float(form_data.get("total_day_minutes")),
            total_day_calls=int(form_data.get("total_day_calls")),
            total_eve_minutes=float(form_data.get("total_eve_minutes")),
            total_eve_calls=int(form_data.get("total_eve_calls")),
            total_night_minutes=float(form_data.get("total_night_minutes")),
            total_night_calls=int(form_data.get("total_night_calls")),
            total_intl_minutes=float(form_data.get("total_intl_minutes")),
            total_intl_calls=int(form_data.get("total_intl_calls")),
            customer_service_calls=int(form_data.get("customer_service_calls")),
            international_plan=1 if form_data.get("international_plan") == "on" else 0,
            voice_mail_plan=1 if form_data.get("voice_mail_plan") == "on" else 0,
            state_0=state_encoding[0],
            state_1=state_encoding[1],
            state_2=state_encoding[2],
            state_3=state_encoding[3],
            state_4=state_encoding[4],
            state_5=state_encoding[5],
            state_6=state_encoding[6]
        )
        
        # Make prediction
        prediction_result = predict_api(input_data)
        
        # Return result
        return templates.TemplateResponse(
            "index.html", 
            {
                "request": request, 
                "prediction": prediction_result["prediction"],
                "probability": prediction_result["probability"],
                "is_churn": prediction_result["is_churn"],
                "form_data": dict(form_data)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))