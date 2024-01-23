import os
import torch
from transformers import GPT2Config
from ts.torch_handler.base_handler import BaseHandler
from trainer.GPT2SP.GPT2ForSequenceClassification import GPT2ForSequenceClassification


class GPT2Handler(BaseHandler):
    def __init__(self):
        super(GPT2Handler, self).__init__()
        self.initialized = False
        self.models = {}

    def initialize(self, ctx):
        # Load the model and tokenizer
        config = GPT2Config(num_labels=1,pad_token_id=50256)
        gpt2sp,tokenize =  (config)
        self.model = gpt2sp
        self.tokenizer = tokenize

        # Load the saved state dict inputs/outputs/gpt2sp_0.pth
        saved_state_dict_path = '/home/seniyas/IIT/fyp/repos/my-training-project/trainer/outputs/gpt2sp_0.pth'
        state_dict = torch.load(saved_state_dict_path)
        self.model.load_state_dict(state_dict)
        self.model.to('cpu')
        self.model.eval()
        self.initialized = True

    def preprocess(self, data):
        # Preprocess input data
        input_text = data[0].get('text')
        if not input_text:
            raise Exception('Please provide the text to generate from')
       
        # Encode the input text
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
        return input_ids

    def load_model(self, model_identifier):
        # Check if the model is already loaded
        if model_identifier in self.models:
            return self.models[model_identifier]

        # Define the path based on the model identifier
        model_directory = f'/path/to/models/{model_identifier}'
        saved_state_dict_path = os.path.join(model_directory, 'model.pth')

        # Load the model and tokenizer
        config = GPT2Config(num_labels=1, pad_token_id=50256)
        model = GPT2ForSequenceClassification(config)
        state_dict = torch.load(saved_state_dict_path)
        model.load_state_dict(state_dict)
        model.to('cpu')
        model.eval()

        # Save the model in the cache
        self.models[model_identifier] = model
        return model


    def inference(self, input_ids):
        # Predict
        with torch.no_grad():
            logits = self.model(input_ids)["logits"]
        return logits

    def postprocess(self, inference_output):
        # Postprocess model predictions
        logits = inference_output.detach().cpu().numpy()
        return logits.tolist()