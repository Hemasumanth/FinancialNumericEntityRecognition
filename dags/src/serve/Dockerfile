FROM python:3.9-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY serve/predict.py /app/
COPY finerteam8-00bf2670c240.json /app/

# Install Flask and google-cloud-storage
RUN pip install tensorflow Flask google-cloud-storage joblib scikit-learn grpcio gcsfs python-dotenv google-cloud-logging google-cloud-bigquery google-auth
ENV AIP_HEALTH_ROUTE=/ping
ENV AIP_PREDICT_ROUTE=/predict
ENV AIP_HTTP_PORT=8080
ENV PROJECT_ID=finerteam8
ENV AIP_STORAGE_URI=gs://finer_data_bk/model
ENV BUCKET_NAME=finer_data_bk
ENV REGION=us-east1
ENV AIP_HTTP_PORT=8080
ENV BIGQUERY_TABLE_ID=finerteam8.finerdatabq.model_monitor
# Run serve.py when the container launches
ENTRYPOINT ["python", "predict.py"]