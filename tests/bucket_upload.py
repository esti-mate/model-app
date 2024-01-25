from trainer.train_process import upload_to_gcp
from constants import FINAL_MODEL_BUCKET_PATH, TRAINER_DIR


res, blob_path = upload_to_gcp(
    FINAL_MODEL_BUCKET_PATH, TRAINER_DIR + "/model_store", "org01"
)
print("RES : ", res, "path :", blob_path)
