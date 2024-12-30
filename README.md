# Financial Numeric Entity Recognition

This project, titled "Financial Numeric Entity Recognition," is at the forefront of applying advanced machine learning techniques and MLOps practices to the intricate field of financial data analysis. It aims to automate the way numeric entities in financial documents are identified and processed. By automating the recognition of these entities, the projects engineered to enhance the accuracy and efficiency of financial data processing, thereby addressing a crucial need in the finance sector.   

At the core of this project is a blend of powerful technologies and methodologies:

Python: As the backbone programming language, Python offers a combination of simplicity and power, providing a versatile platform for data manipulation, numerical computations, and machine learning model development. Its rich ecosystem of libraries and frameworks is instrumental in every phase of this project, from data preprocessing to model training and evaluation.

Airflow: Airflow plays a pivotal role in scheduling and orchestrating the workflow of our project. By defining Directed Acyclic Graphs (DAGs), we can meticulously schedule tasks such as data extraction, transformation, loading (ETL), and model training. Each task in our data pipeline is encapsulated as an Airflow operator, ensuring a controlled execution sequence and dependency management. For instance, a DAG in our setup may be designed to first download financial datasets, and perform data cleaning and numeric entity extraction, followed by feeding the processed data into TensorFlow models for NLP tasks.

TensorFlow: As a leading machine learning framework, TensorFlow is utilized for developing and training sophisticated machine learning models. These models are central to the project's objective of recognizing and interpreting financial numeric entities. TensorFlow's advanced capabilities allow for the implementation of deep learning techniques, which are essential in handling the complexities of financial data.

Docker: Docker containers are instrumental in our project for creating isolated environments tailored to specific parts of our workflow, such as data preprocessing, model training, and model inference. This isolation ensures that our application remains consistent across different stages, including development, testing, and production environments. Docker Compose is particularly useful for orchestrating these multi-container setups, effectively managing complex interdependencies within our application, like linking databases, Airflow components, and web servers.

MLflow: MLflow is utilized for managing the machine learning lifecycle, including experimentation, deployment, and auditing. It allows us to systematically track experiments, model training parameters, and results, thereby enabling a comparative analysis of different models and approaches. MLflow is particularly effective in monitoring the performance of our TensorFlow models, assisting us in fine-tuning parameters for optimal results.

Flask: Flask is employed as a lightweight web framework to deploy our TensorFlow models. It facilitates the creation of RESTful APIs, allowing for easy and accessible model inference. Flask’s simplicity and flexibility make it an ideal choice for quickly deploying models and serving predictions over the web.

Vertex AI: Vertex AI is leveraged in our FinancialNumericEntityRecognition project to streamline the deployment and scaling of our machine learning models, especially those developed using TensorFlow. It offers an integrated environment that simplifies the process of training, tuning, and deploying models at scale. We utilize Vertex AI for its advanced ML operations capabilities, which include automated model training (AutoML) and custom model training pipelines. This integration allows us to efficiently manage the lifecycle of our NLP models, from development to production, ensuring seamless and scalable deployment. Furthermore, Vertex AI's robust monitoring and management tools help in maintaining model performance, tracking usage metrics, and updating models as needed, thereby enhancing our ability to deliver accurate and timely financial entity recognition services.

DVC (Data Version Control): DVC is crucial for data and model versioning in our project. It helps in tracking changes in datasets and machine learning models, enabling us to maintain a history of modifications and experimentations. This feature is particularly valuable for reproducing results and rolling back to earlier versions if needed. DVC's seamless integration with cloud storage solutions enhances our capability to handle large datasets and model files efficiently.

GCP: GCP provides the robust and scalable infrastructure necessary for hosting our datasets and machine learning models. It supports our project with extensive data storage options, powerful computing resources, and reliable hosting services, ensuring high availability and performance of our data pipeline and machine learning models.

Each of these tools is carefully selected and integrated to create a robust and efficient platform for financial data analysis. The project stands as a testament to the power of combining cutting-edge technology with best practices in software development and data science, aiming to set new standards in the field of financial analytics.




## Data Card

### Train Dataset (`train.csv`)
- **Size:** 605,040 rows × 3 columns

| Variable Name | Role      | Type   | Description                                           |
|:--------------|:----------|:-------|:------------------------------------------------------|
| id            | ID        | Integer | Unique identifier for each entry.                     |
| tokens        | Feature   | Object  | Tokenized text data, likely sentences or phrases.     |
| ner_tags      | Feature   | Object  | NER tags associated with each token in the `tokens`.  |

### Test Dataset (`test.csv`)
- **Size:** 151,261 rows × 3 columns

| Variable Name | Role      | Type   | Description                                           |
|:--------------|:----------|:-------|:------------------------------------------------------|
| id            | ID        | Integer | Unique identifier for each entry.                     |
| tokens        | Feature   | Object  | Tokenized text data, likely sentences or phrases.     |
| ner_tags      | Feature   | Object  | NER tags associated with each token in the `tokens`.  |

### Overview
The datasets are structured for NLP tasks, specifically for Named Entity Recognition (NER). 
The `tokens` column in both datasets contains strings of tokenized text, which are sentences 
or phrases split into individual words or tokens. The `ner_tags` column appears to have labels 
for each token, used in NER tasks to classify each token into predefined categories like person 
names, organizations, locations, etc. The large size of the training set indicates its potential 
to train robust NLP models, capturing a wide variety of language patterns and nuances related 
to named entities.






## Getting Started

These instructions will give you a copy of the project up and running on
your local machine for development and testing purposes. See deployment
for notes on deploying the project on a live system.

### Prerequisites

Requirements for the software and other tools to build, test and push 
- [Docker](https://www.example.com)
- [Airflow](https://www.example.com)
- [ELK](https://www.example.com)
- [DVC](https://www.example.com)
- [GCP](https://www.example.com)

### Installing

A step by step series of examples that tell you how to get a development
environment running

**1. Clone the Repository:**

    git clone https://github.com/Harshan1823/FinancialNumericEntityRecognition.git

**2. Set Up a Virtual Environment (optional but recommended):**

    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\Activate`
    
**3. Install Dependencies**

    pip install -r requirements.txt
    
**4. Docker:**

    echo -e "AIRFLOW_UID=$(id -u)" > .env
    echo "AIRFLOW_HOME_DIR=$(pwd)" >> .env
    
    docker compose up airflow-init
    docker compose up

**5. DVC**
```
    dvc pull
```

## Data Pipeline Components

The data pipeline in the FinancialNumericEntityRecognition project consists of several interconnected modules, each dedicated to specific tasks within the data processing workflow. Utilizing Airflow and Docker, we orchestrate and containerize these modules, ensuring efficient and modular data processing.

### 1. Downloading Data:
- **download_data.py:** Responsible for downloading the financial dataset from specified sources.
- **unzip_data.py:** Extracts the content from the downloaded zip files for further processing.

### 2. Cleaning Data:
- **tfdata_cleaning.py:** Handles the initial cleaning and formatting of the financial data.
- **pre_process.py:** Further preprocesses the data, focusing on aspects crucial for NLP tasks such as tokenization and normalization.

### 3. Feature Engineering:
- **tokenise_data.py:** This module is crucial for breaking down text into tokens, a fundamental step in NLP.
- **custom_feature_engineering.py:** Tailored for extracting features specific to financial entities and numeric data, enhancing the input for machine learning models.

Each module reads data from an input path, processes it, and outputs the results for subsequent steps. Airflow ensures that these modules function cohesively, maintaining a smooth data processing flow.

## Data Pipeline using Airflow Dags

![Screenshot showing Data_Pipeline_Airflow_Dags](https://github.com/Harshan1823/FinancialNumericEntityRecognition/assets/22172209/2b6624ff-c94a-4ad0-af81-2b8b6ff000e4)

The above image shows the data preprocessing pipeline. Following are the names and descriptions of the tasks that this dag is executing:
1. **download_data_dag:** Downloads train.csv, test.csv, and validation.csv from hugging face datasets. The data is downloaded into a folder named “raw.”
2. **list_packages_task:** prints output of “pip list”
3. **split_data:** The dataset obtained from hugging face API was not split properly. There was a leak in training data to validation and test data. So we merged all three files, dropped duplicates, and split it again into train and test data. Results are stored in a folder named “inter.”
4. **Convert_TEST_List:** Converts columns in test.csv from strings to lists using “ast.literal_eval”. Results are stored in a JSON file named “test_pre_process.json” in a folder named “final.”
5. **Convert_TRAIN_List:** Converts columns in test.csv from strings to lists using “ast.literal_eval”. Results are stored in a JSON file named “train_pre_process.json” in a folder named “final.”
6. **Generate_Tokeniser:** Train a tokenizer on all the words in the training data, using Tokenizer from “tensorflow.keras.preprocessing.text”. This task creates a folder called “model_store” inside the project directory and saves a file called “tokenizerV1.pkl”
7. **TokenData:** Tokenizes the words in “train_pre_process.json” created in the “Convert_TRAIN_List” task and “test_pre_process.json” in the “Convert_TEST_List.” Results are stored in “train_token.json” and “test_token.json” inside the “final” folder created in the “Convert_TEST_List” task.
8. **DataStats:** Calculates statistics like average string length, min string length, max string length, the standard deviation of string lengths, and 95 percentile of string length. The results are appended to “stats.json” inside the “model_store” folder created in the “Generate_Tokeniser” task.
9. **Gcloud_upload:** “train_token.json,” “test_token.json,” “tokenizerV1.pkl” and “stats.json” to google cloud bucket.

## Tracking files on Google Cloud Storage using DVC

![Screenshot showing DVC_On_GCP_Bucket](https://github.com/Harshan1823/FinancialNumericEntityRecognition/assets/22172209/1c9869bd-c3c0-4341-960d-9a72c7c355a4)

We are tracking the files in the final folder created in the “Convert_TEST_List” task in the data preprocessing using DVC. We use Google Cloud Bucket to store the tracked file’s hash files. In the image above we can see the hash folder md5.

## Machine Learning Modeling Pipeline

Our machine learning pipeline is hosted on Google Cloud Platform (GCP) and integrates various tools for robust and scalable model development.

### Pipeline Components:

1. **Trainer:**
   - **train.py:** A Python script that trains the NLP model, focusing on financial entity recognition.
   - **Dockerfile:** Used to containerize the training environment, ensuring consistency across different platforms.

2. **Serve:**
   - **predict.py:** A Flask application for making predictions using the trained model.
   - **Dockerfile:** Ensures that the Flask app is containerized for reliable deployment.

3. **Model Management:**
   - **build.py:** Manages the training and serving pipeline, deploying the model on Vertex AI.

4. **Inference:**
   - **inference.py:** Handles incoming data for model predictions, demonstrating the model's real-world applicability.

### Experimental Tracking with MLflow:

We use MLflow for tracking our experiments, focusing on metrics that are critical for evaluating NLP models in the context of financial data. MLflow's integration allows us to monitor, compare, and optimize model parameters effectively.

![Screenshot showing MLFLow_Experiment_Tracking](https://github.com/Harshan1823/FinancialNumericEntityRecognition/assets/22172209/f21e8ee9-cb73-4e4c-a8c8-7d0ced4b156b)

The above image shows the chart, which shows the different hyperparameters and metrics tracked by the mlflow. We are choosing the model with the highest f1_macro.
We are logging the following parameters using mlfow.
1. *num_tokens:* Number of words in vocabulary
2. *num_tags:* Number of output labels
3. *d_model:* Size of hidden layer
4. *num_heads:* Number of attention heads
5. *dff:* Size of feed-forward hidden layers
6. *lstm_units:* Number of LSTM units in each LSTM layer
7. *epochs:* Number of iterations. 
8. *pad_length:* Max length of sentence in model
9. *training_time:* Training time of the model
10. *f1_macro:* A metric for evaluating the F1 score, which considers the balanced average across all classes.
11. *F1_micro:* An overall F1 score that calculates precision and recall across all classes, suitable for imbalanced datasets.
12. *Val_loss:* Validation loss, a measure of the model's performance on a validation dataset during training.
13. *Loss:* General term referring to the model's error, commonly used during training to optimize the model.
14. *accuracy:* A metric indicating the percentage of correctly classified instances in the dataset.
15. *f1_weighted:* An F1 score that considers class imbalances by calculating a weighted average.
16. *val_accuracy:* Validation accuracy, representing the accuracy of the model on a validation dataset during training.

### Model Staging and Production:

The models go through stages of development, from initial training to staging and eventually production deployment. We manage these stages using MLflow and Vertex AI, ensuring that our models are robust, efficient, and scalable.

## Deployment Process:

## Model Deployment:

- Configure necessary environment variables (eg: Project, Bucket, Service Account, Container)
- Initialize the Google Cloud AI platform.
- Containerize the training job to build container image
- Push the image to artifact registry
- Run the container of training job in Vertex AI
- Deploy the model
- Expose the endpoint

## Model Serving:

- Define and containerize a Flask web application, containing a form to get input
- Deploy the app on a Kubernetes cluster
- On submit, access the deployed model through an API and return predictions
- Render the results on the web application
 
    
