# Set config
# TARGET_DIR='/Users/lukasalemu/Documents/00. Bank of England/01. Future FAME/model_framework/mlflow'
TARGET_DIR="./"
SERVE_ARTIFACTS=True

echo "Starting MLFLow Server for demo"
echo "Storing data in $TARGET_DIR"

# Start the mlflow server and UI
mlflow server \
  --backend-store-uri $TARGET_DIR \
  --registry-store-uri $TARGET_DIR \
  --default-artifact-root $TARGET_DIR \
  --host 0.0.0.0 \
  --port 8888

wait
