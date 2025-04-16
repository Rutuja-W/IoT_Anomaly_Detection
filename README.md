# 🛰️ IoT Anomaly Detection
**A Comparative Study of Machine Learning and Deep Learning Models for Detecting Network Anomalies in IoT Environments**

## 📌 Project Overview
This project addresses the critical need for anomaly detection in Internet of Things (IoT) networks by exploring and comparing a variety of Machine Learning (ML) and Deep Learning (DL) models. Our goal is to identify models that balance high detection performance with low computational cost, making them ideal for real-time, resource-constrained IoT deployments.

We used the **CIC-BCCC-NRC-IoMT-2024** dataset, which contains realistic labeled traffic from IoT devices under both benign and attack conditions. Models were trained, evaluated, and compared on accuracy, F1-score, recall, precision, and training time.

> 🔗 **Live Repo:** [https://github.com/Rutuja-W/IoT_Anomaly_Detection](https://github.com/Rutuja-W/IoT_Anomaly_Detection)  
> 🔒 Set to private. Collaborators will be added on request.

---

## 📊 Models Implemented

### 🔹 Classical Machine Learning Models
- ✅ Decision Tree
- ✅ Naive Bayes (Raw and Optimized)
- ✅ Logistic Regression

### 🔹 Deep Learning and Advanced ML Models
- ✅ Support Vector Machine (SVM)
- ✅ SVM with Stochastic Gradient Descent (Hinge Loss)
- ✅ Random Forest
- ✅ XGBoost
- ✅ Convolutional Neural Network (CNN)
- ✅ Isolation Forest (unsupervised baseline)

---

## 🧪 Dataset Used
- **Name:** CIC-BCCC-NRC-IoMT-2024
- **Source:** [CIC Research Group](http://cicresearch.ca/IOTDataset/CIC-BCCC-NRC-TabularIoTAttacks-2024)
- **Size:** Over 1 million rows of labeled network traffic
- **Content:** Realistic IoT scenarios with benign and attack flows (DoS, DDoS, C&C, Scans, etc.)

---

## 🔧 Methodology

1. **Data Cleaning & Preprocessing**
   - Dropped irrelevant identifiers (`Flow ID`, IPs, timestamps)
   - Encoded attack names using `LabelEncoder`
   - Normalized features using `StandardScaler`
   - Handled class imbalance using `SMOTE`

2. **Model Training & Evaluation**
   - Split data (80/20) using `train_test_split`
   - Trained each model using standard and optimized settings
   - Measured performance with classification report and confusion matrix
   - Visualized results using `matplotlib` and `seaborn`

3. **Deployment Support**
   - Built FastAPI prototype for external model access via HTTP
   - Designed for containerized + serverless deployment

---

## 📈 Key Results (Lightweight Models)

| Model               | Accuracy | F1 (Anomaly) | Training Time |
|---------------------|----------|--------------|----------------|
| Decision Tree       | 99.9%    | 0.99         | ~2 seconds     |
| Naive Bayes (raw)   | 72.1%    | 0.63         | <1 second      |
| Naive Bayes (scaled)| 95.3%    | 0.91         | ~2 seconds     |
| Logistic Regression | 99.6%    | 0.98         | ~1 second      |

---

## 📊 Key Results (Advanced Models)

| Model              | Accuracy | F1 (Anomaly) | Training Time       |
|--------------------|----------|--------------|----------------------|
| CNN                | 99.70%   | 0.94         | ~20 minutes          |
| SVM                | 99.71%   | 0.86         | ~20 minutes          |
| SGD (Hinge Loss)   | 99.57%   | 0.78         | ~6 seconds           |
| Random Forest      | 99.89%   | 0.94         | ~10 minutes          |
| XGBoost            | 99.95%   | 0.98         | ~34 seconds          |

---

## 🚀 Deployment (Prototype)

We created a **FastAPI** service prototype to enable external access to the trained models. Users can:
- Send network traffic feature vectors via POST requests
- Receive predictions and probabilities in JSON format
- Deploy in serverless or containerized cloud environments

---

## 🧠 Lessons Learned
- Data preprocessing (scaling + SMOTE) dramatically impacts model performance.
- Simple models like Logistic Regression are competitive with DL methods when tuned.
- Deep models (CNN, XGBoost) offer high accuracy but may be impractical for real-time IoT.
- Model deployment should consider interpretability, speed, and memory constraints.

---

## 📚 References

- CIC-BCCC-NRC-IoMT-2024 Dataset. [CIC Research](http://cicresearch.ca/IOTDataset)
- Scikit-learn Documentation. https://scikit-learn.org
- XGBoost: Chen & Guestrin, 2016

---

## 👩‍💻 Contributors

- **Gayathri Rayudu** – ML model development, evaluation, documentation  
- **Rutuja Wani** – DL modeling, dataset analysis, report writing  
- **Yanyi Li** – CNN implementation, deployment support  

---

## 📬 Contact
For questions or access to the repository:
📧 Email: `rayudu_g@sfu.ca`, `rutuja@example.com`, `yanyi@example.com`

---

