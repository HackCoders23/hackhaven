from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import traceback

# Load model and artifacts
model = joblib.load("cyber_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("label_encoder.pkl")
# import joblib
top_features = joblib.load("top_features.pkl")
print(top_features)

# Sanitize feature names for Pydantic (replace spaces and slashes)
top_features_pydantic = [f.replace(" ", "_").replace("/", "_") for f in top_features]
# Mapping: sanitized_name → original_name
pyd_to_original = {f.replace(" ", "_").replace("/", "_"): f for f in top_features}
app = FastAPI(title="Cyber Attack Detection and Prevention API")
class TrafficFeatures(BaseModel):
    Bwd_Packets_s: float
    Max_Packet_Length: float
    Avg_Bwd_Segment_Size: float
    Average_Packet_Size: float
    Packet_Length_Std: float
    Packet_Length_Mean: float
    Bwd_Packet_Length_Max: float
    Destination_Port: float
    Bwd_Packet_Length_Std: float
    Packet_Length_Variance: float
    Init_Win_bytes_forward: float
    Total_Length_of_Bwd_Packets: float
    Subflow_Bwd_Bytes: float
    Fwd_Packet_Length_Mean: float
    Avg_Fwd_Segment_Size: float
    Min_Packet_Length: float
    Flow_Bytes_s: float
    Flow_IAT_Max: float
    Bwd_Packet_Length_Mean: float
    Flow_Packets_s: float
    print(top_features_pydantic)


# Create FastAPI app


# # Define Pydantic model for sanitized inputs
# class TrafficFeatures(BaseModel):
#     Flow_Duration: float
#     Total_Fwd_Packets: float
#     Total_Backward_Packets: float
#     Flow_Bytes_s: float
#     Flow_Packets_s: float
#     Fwd_Packet_Length_Max: float
#     Bwd_Packet_Length_Max: float
#     Flow_IAT_Mean: float
#     Fwd_IAT_Total: float
#     Bwd_IAT_Total: float
#     Min_Packet_Length: float
#     Packet_Length_Mean: float
#     Packet_Length_Std: float
#     Packet_Length_Variance: float
#     FIN_Flag_Count: float
#     SYN_Flag_Count: float
#     ACK_Flag_Count: float
#     Average_Packet_Size: float
#     Init_Win_bytes_forward: float
#     Init_Win_bytes_backward: float

@app.get("/")
def home():
    return {"message": "Cyber Attack Detection and Prevention API is running 🚀"}

@app.post("/predict")
def predict_attack(features: TrafficFeatures):
    try:
        # Get feature vector in correct order (match training data)
        feature_vector = [getattr(features, pyd_name) for pyd_name in pyd_to_original]

        # Scale input and make prediction
        input_scaled = scaler.transform([feature_vector])
        prediction = model.predict(input_scaled)
        label = encoder.inverse_transform(prediction)[0]

        # Decide what action to take
        if label != "BENIGN":
            action = take_preventive_action(label)
        else:
            action = "No action needed. Traffic is normal."

        return {
            "prediction": label,
            "action_taken": action
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Define simple response action logic
def take_preventive_action(attack_type):
    if "DDoS" in attack_type:
        return "Block IP address temporarily. Alert network admin."
    elif "PortScan" in attack_type:
        return "Flag the source IP. Monitor closely for further activity."
    elif "BruteForce" in attack_type:
        return "Lock user account. Force password reset."
    else:
        return "Alert security team. Log and monitor suspicious activity."
 