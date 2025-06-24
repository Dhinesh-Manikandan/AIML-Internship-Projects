# Install dependencies (run only once)
!pip install mediapipe opencv-python tensorflow gtts pygame scikit-learn --quiet

from utils.data_preprocessor import prepare_dataset
from utils.model_trainer import build_and_train_model
from utils.realtime_predictor import run_realtime_recognition

# Set the path to your dataset
dataset_path = "D:/AI ML Intern Elevate Labs Benglore/Project/indian.csv"

# Step 1: Preprocess the dataset
X_train, X_test, y_train, y_test, label_encoder = prepare_dataset(dataset_path)

# Step 2: Train the model and save it
model_path = build_and_train_model(X_train, y_train, X_test, y_test)

# Step 3: Run real-time recognition
run_realtime_recognition(model_path, label_encoder)