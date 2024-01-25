from google.cloud import aiplatform
from constants import GCP_PROJECT_ID, AI_LOCATION


# Import the model to vertex ai model registry
def import_model(
    display_name, artifact_path, serving_container_image_uri
) -> aiplatform.Model:
    aiplatform.init(project=GCP_PROJECT_ID, location=AI_LOCATION)

    model = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri="gs://" + artifact_path,
        serving_container_image_uri=serving_container_image_uri,
        sync=True,
    )
    print("Importing model to model registry!..." + model.display_name)
    model.wait()

    print(model.display_name, "Model imported! ", model.resource_name)

    return model


def deploy_model_to_endpoint(model, endpoint_name, traffic_percentage=100):
    aiplatform.init(project=GCP_PROJECT_ID, location=AI_LOCATION)

    endpoint = aiplatform.Endpoint(endpoint_name=endpoint_name)

    # deployed_model = endpoint.deploy(
    #     model=model,
    #     traffic_percentage=traffic_percentage,
    #     machine_type="n1-standard-4",
    #     sync=True,
    # )

    # endpoint.predict(traffic_percentage=traffic_percentage)

    # print(deployed_model.display_name, "deployed to endpoint", endpoint.display_name)

    # return deployed_model
