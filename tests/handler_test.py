import json
from predictor.model_handler import GPT2Handler

# Adjust the import according to your file structure

# Initialize handler
handler = GPT2Handler()
handler.initialize(ctx=None)  # ctx is context, often not needed for local tests

# Simulate an input request
input_text = "Your"
input_data = json.dumps({"text": input_text})
input_data = json.loads(input_data)  # Simulating a request body

# Process the input through the handler
try:
    # Preprocess
    preprocessed_data = handler.preprocess(data=[input_data])

    # Inference
    inference_result = handler.inference(input_ids=preprocessed_data)

    # Postprocess
    postprocessed_result = handler.postprocess(inference_output=inference_result)

    # Print the result
    print("Inference Output:", postprocessed_result)
except Exception as e:
    print("Error during model inference:", str(e))
