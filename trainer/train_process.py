import os
import glob
from transformers import GPT2Tokenizer, Pipeline, GPT2Config
from google.cloud import storage
from constants import (
    MODEL_ID,
    TOKENIZER_ID,
    DEVICE,
    FINAL_MODEL_BUCKET_PATH,
    INPUT_DIR,
    TRAINER_DIR,
    BATCH_SIZE_RATIO,
    SEQUENCE_LENGTH,
    LEARNING_RATE,
    EPOCH,
)

from .GPT2SP.GPT2ForSequenceClassification import (
    GPT2ForSequenceClassification as GPT2SP,
)
from .utils.train_gpt2 import (
    process_data,
    tokenize_text_list,
    prepare_dataloader,
    train,
)
from .utils.archive_model import create_model_file

from .utils.deploy_model import import_model


# from constants import INPUT_DIR, TRAINER_DIR

# model_path = "MickyMike/0-GPT2SP-appceleratorstudio"
# gpt2sp = GPT2SP.from_pretrained(model_path)
# gpt2sp.eval()

# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# tokenizer.pad_token = '[PAD]'


def get_gpt2sp_pipeline(model_configs: None) -> (GPT2SP, GPT2Tokenizer):
    model = MODEL_ID  # "MickyMike/0-GPT2SP-appceleratorstudio"
    gpt2sp = GPT2SP.from_pretrained(model, config=model_configs)
    gpt2sp.to(DEVICE)

    tokenizer = GPT2Tokenizer.from_pretrained(TOKENIZER_ID)
    tokenizer.pad_token = "[PAD]"
    return gpt2sp, tokenizer


def predict_sp(estimator: Pipeline, given_title: str) -> dict:
    return round(estimator(given_title).item(), 0)


def get_input_ids(tokenizer: GPT2Tokenizer, text: str):
    output = tokenizer(
        text, return_tensors="pt", truncation=True, padding="max_length", max_length=20
    )
    return output.data["input_ids"]


# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_FILE


# def inference_gpt2sp():
#     gpt2sp , tokenizer = get_gpt2sp_pipeline(None)

#     saved_state_dict_path = './outputs/gpt2sp_4'+'.pth'
#     state_dict = torch.load(saved_state_dict_path)

#     gpt2sp.load_state_dict(state_dict)
#     gpt2sp.eval()

#     input_text = request.json.get('text')
#     if not input_text:
#        return jsonify({'error': 'Please provide the text to generate from'}), 400

#     input_ids = get_input_ids(tokenizer,input_text)
#     logits = gpt2sp(input_ids)
#     logits = logits["logits"].detach().cpu().numpy()


#     # input_vec = tokenizer.encode(input_text, return_tensors='pt')
#     return logits.tolist()


def train_gpt2sp(output_dir):
    # load datasets from the marker
    train_text, train_labels = process_data(INPUT_DIR)
    # train the model
    # loading the gpt2 tokenizer
    config = GPT2Config(num_labels=1, pad_token_id=50256)
    # loading the GPTSP model
    gpt2sp, tokenizer = get_gpt2sp_pipeline(config)

    # Tokenize the training text
    train_tokens = tokenize_text_list(train_text, tokenizer, SEQUENCE_LENGTH)
    print(train_tokens["input_ids"][:5])

    # prepare dataloader
    dl = prepare_dataloader(train_tokens, train_labels, BATCH_SIZE_RATIO)

    trained_model = train(
        train_dl=dl,
        model=gpt2sp,
        LEARNING_RATE=LEARNING_RATE,
        device=DEVICE,
        epochs=EPOCH,
        output_dir=output_dir,
    )

    return "Training Completed"


def upload_to_gcp(bucket_name, source_folder, subdirectory):
    """
    Uploads files from the specified folder to the GCP bucket.
    :param bucket_name: Name of the GCP bucket.
    :param source_folder: Local folder to upload files from.
    """
    # Initialize a storage client
    print("****Uploading files to GCP bucket...")
    storage_client = storage.Client()

    # Ensure subdirectory string is properly formatted
    if not subdirectory.endswith("/"):
        subdirectory += "/"

    # Get the bucket object
    bucket = storage_client.bucket(bucket_name)

    # List files in the local directory
    files = glob.glob(os.path.join(source_folder, "*"))

    # Upload files to GCP bucket
    # for file_path in files:
    file_path = files[len(files) - 1]
    file_name = "model.mar"  # os.path.basename(file_path)

    # rename the model file to model.mar
    blob_path = f"{subdirectory}{file_name}"
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(file_path)

    print(f"File {file_name} uploaded to {blob_path}")
    return (
        bucket_name + "/" + subdirectory + file_name,
        f"{bucket_name}/{subdirectory[:-1]}",
    )


# Local directory to upload files from


def create_output_dir():
    print("Creating output dir : outputs | model_store in : " + "package_dir")
    if not os.path.exists(TRAINER_DIR + "/outputs"):
        os.makedirs(TRAINER_DIR + "/outputs")
        os.makedirs(TRAINER_DIR + "/model_store")
    return TRAINER_DIR + "/outputs"


def archive_model(export_path):
    create_model_file(export_path)


# Main Training steps
def startTraining():
    output_dir = create_output_dir()
    model_path = TRAINER_DIR + "/model_store"
    ORG_ID = "org01"

    train_gpt2sp(output_dir)
    archive_model(model_path)

    file_path, org_directory_path = upload_to_gcp(
        FINAL_MODEL_BUCKET_PATH, model_path, ORG_ID
    )

    import_model(
        "model_o2",
        artifact_path=org_directory_path,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.2-1:latest",
    )
