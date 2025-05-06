# Ophthalmic-Diagnosis-Detection-through-Deep-Learning

<img width="331" alt="Image" src="https://github.com/user-attachments/assets/5094f002-f7df-4d15-8b73-f1b694b958f8" />

## **Overview**  
This project leverages deep learning to detect ophthalmic diseases such as cataract, diabetic retinopathy, and glaucoma using medical images. The model is trained using convolutional neural networks (CNNs) to classify eye conditions and assist in early diagnosis.  

## **Features**  
✅ Automatic classification of ophthalmic diseases  
✅ Supports multiple categories: Cataract, Diabetic Retinopathy, Glaucoma, and Normal  
✅ Deep learning model trained using TensorFlow/Keras  
✅ Flask-based web interface for user interaction  
✅ Model deployment for real-time predictions  

## **Tech Stack**  
- **Programming Language:** Python  
- **Frameworks & Libraries:** TensorFlow, Keras, OpenCV, Flask  
- **Frontend:** HTML, CSS, JavaScript (for UI)  
- **Deployment:** Streamlit/Flask  

## **Project Structure**  
```
├── static/                # Static files (images, scripts)  
├── templates/             # HTML templates for UI  
├── test/                  # Test dataset  
│   ├── cataract/  
│   ├── diabetic_retinopathy/  
│   ├── glaucoma/  
│   ├── normal/  
├── train/                 # Training dataset  
│   ├── cataract/  
│   ├── diabetic_retinopathy/  
│   ├── glaucoma/  
│   ├── normal/  
├── uploads/               # Uploaded images for classification  
├── App.py                 # Main application script  
├── main.ipynb             # Jupyter Notebook for training  
├── model_dense.json       # Model architecture  
├── model_dense.weights.h5 # Trained model weights  
├── requirements.txt       # Dependencies  
└── README.md              # Project documentation  
```  

## **Installation & Setup**  
1. **Clone the Repository**  
   ```bash
   git clone https://github.com/your-username/ophthalmic-diagnosis.git  
   cd ophthalmic-diagnosis  
   ```  

2. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt  
   ```  

3. **Run the Web App**  
   ```bash
   python App.py  
   ```  
   Access the web interface at `http://localhost:5000`  

## **Model Training**  
- The model is trained on labeled ophthalmic datasets using deep learning techniques.  
- CNN-based architecture ensures high accuracy in disease detection.  
- Training is done in `main.ipynb`.  

## **Results**  
- Achieved [Provide Accuracy]% accuracy on the test dataset.  
- Model successfully classifies ophthalmic diseases with high precision.  



