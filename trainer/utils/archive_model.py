import subprocess
import os
import pathlib
import importlib.resources as pkg_resources

# Define the paths
with pkg_resources.path('trainer','') as p:
    TRAINER_DIR = str(p)

with pkg_resources.path('predictor','') as p:
    model_handler = str(p)

def create_model_file(export_path):

    new_path = '/root/.local/bin'
    current_path = os.environ.get('PATH', '')
    os.environ['PATH'] = new_path + ':' + current_path
   

    model_dir = TRAINER_DIR
    model_file = 'GPT2SP/GPT2ForSequenceClassification.py'  # Updated model file name
    serialized_file_name = 'outputs/gpt2sp_0.pth'  # If you named your weights file differently
    handler_file = model_handler + '/model_handler.py'
    export_path = export_path
    model_name = 'gpt2_model'
    version = '1.0'

    print(f"Creating model archive {model_name}.mar ...")

    # Create the .mar file
    command = f"torch-model-archiver --model-name {model_name} --version {version} --model-file {model_dir}/{model_file} --serialized-file {model_dir}/{serialized_file_name} --handler {handler_file} --export-path {export_path} --force"
    subprocess.run(command, shell=True)