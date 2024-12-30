FROM python:3.9-slim

WORKDIR /

ENV AIP_STORAGE_URI=gs://finer_data_bk/model
ENV BUCKET_NAME=finer_data_bk
ENV BIGQUERY_TABLE_ID=finerteam8.finerdatabq.model_monitor
COPY trainer /trainer
COPY finerteam8-00bf2670c240.json /trainer

RUN pip install --no-cache-dir tensorflow gcsfs google-cloud-storage scikit-learn joblib grpcio python-dotenv mlflow numpy google-cloud-logging google-cloud-bigquery google-auth

CMD ["python3", "/trainer/gcp_train.py"]