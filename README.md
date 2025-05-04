
# 🔐 AI-Powered Cyber Attack Detection and Prevention System

This project leverages machine learning and FastAPI to build a *real-time cyberattack detection and prevention system*. It detects malicious network traffic and automatically recommends preventive actions based on the type of attack.

---

## 🚀 Features

- Trained on the [CICIDS 2017 dataset](https://www.kaggle.com/datasets/sampadab17/cicids2017)
- Machine Learning model using *Random Forest Classifier*
- Feature selection based on importance ranking
- Real-time predictions via *FastAPI*
- Automated *prevention actions* based on detected threats

---

## 📁 Repository Structure

.
├── main.py # FastAPI app for real-time predictions
├── test.py # Data processing, model training, and feature selection
├── scaler.pkl # Scaler for feature normalization
├── label_encoder.pkl # Encoder for converting labels into numeric format
├── top_features.pkl # Top 20 features selected for training
├── cyber_model.pkl # Final trained Random Forest model
├── requirements.txt # Python dependencies
└── README.md # Project documentation



## 📊 Dataset

The model is trained on the Wednesday-workingHours.pcap_ISCX.csv file from the *CICIDS 2017* dataset.

> 🔗 [Download Dataset from Kaggle](https://www.kaggle.com/datasets/sampadab17/cicids2017)



## ⚙ Getting Started

### 1. Clone the repository

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
2. Install dependencies
pip install -r requirements.txt

If requirements.txt is missing, install manually:

pip install pandas numpy scikit-learn fastapi uvicorn joblib

3. Train the model (Optional)
If you wish to retrain the model from scratch:
python test.py

4. Run the FastAPI server
uvicorn main:app --reload

Open in your browser:

Swagger Docs: http://127.0.0.1:8000/docs

Main Endpoint: http://127.0.0.1:8000

🧠 How It Works
test.py

Loads the dataset, cleans it, and balances classes
Trains a baseline Random Forest model
Selects top 20 important features
Retrains model with selected features and saves it

main.py
Loads trained model and artifacts
Accepts network traffic feature input via POST request

Returns predicted attack type and recommended prevention action

🧪 API Usage
POST /predict
Request Body (JSON):
Keys must match the top 20 features used in training.
{
"Bwd_Packets_s": 0.04,
"Max_Packet_Length": 1450.0,
"Avg_Bwd_Segment_Size": 280.5,
"Average_Packet_Size": 340.0,
"Packet_Length_Std": 180.0,
"Packet_Length_Mean": 295.0,
"Bwd_Packet_Length_Max": 1350.0,
"Destination_Port": 80,
"Bwd_Packet_Length_Std": 110.5,
"Packet_Length_Variance": 19000.0,
"Init_Win_bytes_forward": 8192.0,
"Total_Length_of_Bwd_Packets": 13500.0,
"Subflow_Bwd_Bytes": 13500.0,
"Fwd_Packet_Length_Mean": 285.0,
"Avg_Fwd_Segment_Size": 270.0,
"Min_Packet_Length": 50.0,
"Flow_Bytes_s": 14500.0,
"Flow_IAT_Max": 8500.0,
"Bwd_Packet_Length_Mean": 260.0,
"Flow_Packets_s": 3.1
}

Sample Response:
{
  "prediction": "BENIGN",
  "action_taken": "No action needed. Traffic is normal."
}


🛡 Attack Prevention Actions

Detected            Attack & Action
DDoS	            Block IP temporarily,alert network admin
PortScan	        Flag and monitor source IP
BruteForce 	        Lock account, force password reset
Others (generic)	Log incident, notify security team, and increase monitoring


🙌 Acknowledgments
CICIDS 2017 Dataset
Scikit-learn
FastAPI
Uvicorn

💡 Future Work
Integrate real-time packet sniffer for live predictions
Build a live monitoring dashboard
Add IP reputation scoring and threat intelligence feeds

💡 Future Work
✅ Integrate real-time packet sniffer for live attack detection
✅ Build a live monitoring dashboard to visualize attack patterns and traffic
✅ Enable live logging of all API calls and model predictions
✅ Auto-scaling and containerization (Docker + Kubernetes) for cloud deployment
✅ Add IP reputation scoring and external threat intelligence feeds
✅ Build alert system (email/SMS/Slack) for critical incidents