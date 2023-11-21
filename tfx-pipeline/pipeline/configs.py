# define global application settings, such as pipeline name and file path
import os

# Pipeline name will be used to identify this pipeline.
PIPELINE_NAME = 'sentiment-analysis-tfx'

# GCP related configs.

# Following code will retrieve your GCP project. You can choose which project 
# to use by setting GOOGLE_CLOUD_PROJECT environment variable.

GOOGLE_CLOUD_PROJECT = 'master-host-403612'


# Specify your GCS bucket name here. You have to use GCS to store output files 
# when running a pipeline with Kubeflow Pipeline on GCP or when running a job
# using Dataflow. Default is '<gcp_project_name>-kubeflowpipelines-default'.
# This bucket is created automatically when you deploy KFP from marketplace.
# GCS_BUCKET_NAME = GOOGLE_CLOUD_PROJECT + '-kubeflowpipelines-default'
GCS_BUCKET_NAME = GOOGLE_CLOUD_PROJECT + '-vertex-default'
GOOGLE_CLOUD_REGION = 'us-central1'

# GCP
TRANSFORM_MODULE_FILE = 'gs://{}/{}/modules/preprocessing.py'.format(GCS_BUCKET_NAME, PIPELINE_NAME)
TRAIN_MODULE_FILE = 'gs://{}/{}/modules/model.py'.format(GCS_BUCKET_NAME, PIPELINE_NAME)
TUNER_MODULE_PATH = 'gs://{}/{}/best_hyperparameters/'.format(GCS_BUCKET_NAME, PIPELINE_NAME)
DATA_PATH = 'gs://{}/{}/data/'.format(GCS_BUCKET_NAME, PIPELINE_NAME)
LABEL_ENCODER_FILE = 'gs://{}/{}/modules/label_encoder.pkl'.format(GCS_BUCKET_NAME, PIPELINE_NAME)

# Paths for users' Python module.
MODULE_ROOT = 'gs://{}/{}/modules/'.format(GCS_BUCKET_NAME, PIPELINE_NAME)
# Name of Vertex AI Endpoint.
ENDPOINT_NAME = 'prediction-' + PIPELINE_NAME

# LOCAL
LOCAL_TRANSFORM_MODULE_FILE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), r'../..', 'modules', 'preprocessing.py'))
LOCAL_TRAIN_MODULE_FILE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), r'../..', 'modules', 'model.py'))
LOCAL_TUNER_MODULE_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), r'../..', 'modules', 'best_hyperparameters'))
LOCAL_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), r'../..', 'modules', 'data'))
LOCAL_LABEL_ENCODER_FILE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), r'../..', 'modules', 'label_encoder.pkl'))

# Following image will be used to run pipeline components run if Kubeflow
# Pipelines used.
# This image will be automatically built by CLI if we use --build-image flag.
PIPELINE_IMAGE = f'gcr.io/{GOOGLE_CLOUD_PROJECT}/{PIPELINE_NAME}'


# (Optional) Uncomment below to use AI Platform training.
VERTEX_TRAINING_ARGS = {
    'project': GOOGLE_CLOUD_PROJECT,
            'worker_pool_specs': [{
                'machine_spec': {
                    'machine_type': 'n1-standard-4',
                },
                'replica_count': 1,
                'container_spec': {
                    'image_uri': 'gcr.io/tfx-oss-public/tfx:{}'.format(tfx.__version__),
                },
            }],
}

# A dict which contains the serving job parameters to be passed to Google
# Cloud AI Platform. For the full set of parameters supported by Google Cloud AI
# Platform, refer to
# https://cloud.google.com/ml-engine/reference/rest/v1/projects.models
# (Optional) Uncomment below to use AI Platform serving.
VERTEX_SERVING_ARGS = {
    'project_id': GOOGLE_CLOUD_PROJECT,
    'endpoint_name': ENDPOINT_NAME,
    # Remaining argument is passed to aiplatform.Model.deploy()
    # See https://cloud.google.com/vertex-ai/docs/predictions/deploy-model-api#deploy_the_model
    # for the detail.
    #
    # Machine type is the compute resource to serve prediction requests.
    # See https://cloud.google.com/vertex-ai/docs/predictions/configure-compute#machine-types
    # for available machine types and acccerators.
    'machine_type': 'n1-standard-4',
}