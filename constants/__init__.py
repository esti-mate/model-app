import importlib.resources as pkg_resources


MODEL_ID = "gpt2"
TOKENIZER_ID = "gpt2"
DEVICE = "cpu"
SEQUENCE_LENGTH = 1  # 20
BATCH_SIZE_RATIO = 0.2
LEARNING_RATE = 5e-4
EPOCH = 1

# Locations
AI_LOCATION = "us-central1"
GCP_PROJECT_ID = "esti-mate-411304"
FINAL_MODEL_BUCKET_PATH = "finished-models"
with pkg_resources.path("files", "") as p:
    INPUT_DIR = str(p)

with pkg_resources.path("trainer", "") as p:
    TRAINER_DIR = str(p)

# SERVICE_ACCOUNT_FILE = INPUT_DIR + "/service-acc.json"
