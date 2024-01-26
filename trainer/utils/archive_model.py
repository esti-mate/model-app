import subprocess
import os
import tempfile
import shutil
import importlib.resources as pkg_resources
from constants import FILES_DIR, CONSTANTS_DIR, PREDICTOR_DIR

# Define the paths
with pkg_resources.path("trainer", "") as p:
    TRAINER_DIR = str(p)

with pkg_resources.path("predictor", "") as p:
    model_handler = str(p)


def create_model_file(export_path):
    new_path = "/root/.local/bin"
    current_path = os.environ.get("PATH", "")
    os.environ["PATH"] = new_path + ":" + current_path

    model_dir = FILES_DIR
    model_file = (
        TRAINER_DIR + "/GPT2SP/GPT2ForSequenceClassification.py"
    )  # Updated model file name
    serialized_file_name = (
        "outputs/gpt2sp_0.pth"  # If you named your weights file differently
    )
    handler_file = model_handler + "/model_handler.py"
    export_path = export_path

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    trainer_temp_dir = os.path.join(temp_dir, "cp", "trainer")
    constant_temp_dir = os.path.join(temp_dir, "cp", "constants")
    predictor_temp_dir = os.path.join(temp_dir, "cp", "predictor")
    files_temp_dir = os.path.join(temp_dir, "cp", "files")

    # Copy the 'trainer' directory to the temporary directory
    shutil.copytree(TRAINER_DIR, trainer_temp_dir)
    shutil.copytree(CONSTANTS_DIR, constant_temp_dir)
    shutil.copytree(PREDICTOR_DIR, predictor_temp_dir)
    shutil.copytree(FILES_DIR, files_temp_dir)

    # removing model files since it is included in archive
    shutil.rmtree(files_temp_dir + "/outputs")
    shutil.rmtree(files_temp_dir + "/model_store")

    paths_a = model_handler.split("/")
    paths_a.pop(-1)
    print("FILES_DIR", model_handler)
    requirements_path = "/".join(paths_a) + "/requirements.txt"
    print("requirements_path", requirements_path)

    subprocess.run(f'ls {"/".join(paths_a)}')

    additional_files = temp_dir + "/cp"
    model_name = "gpt2_model"
    version = "1.0"

    print(f"Creating model archive {model_name}.mar ...")

    # Create the .mar file
    command = f"torch-model-archiver --model-name {model_name} --version {version} --model-file {model_file} --serialized-file {model_dir}/{serialized_file_name} --handler {handler_file} --extra-files {additional_files} --export-path {export_path} --requirements-file {requirements_path}  --force"
    subprocess.run(command, shell=True)
