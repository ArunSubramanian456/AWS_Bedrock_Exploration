# STEP 1 : Configure your access via AWS CLI  in your local computer.
# STEP 2: Setup virtual environment and run `pip install -r requirements.txt` to install the required dependencies.
# STEP 3: Import the required libraries.
# Reference - https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-runtime_example_bedrock-runtime_InvokeModel_TitanText_section.html

import boto3
import json

from botocore.exceptions import ClientError

# STEP 4: Create a Bedrock Runtime client in the AWS Region of your choice.
client = boto3.client("bedrock-runtime", region_name="us-east-1")

# STEP 5: Set the model ID, e.g., Titan Text Premier.
model_id = "amazon.titan-text-premier-v1:0"

# STEP 6: Define the prompt for the model.
prompt = "Give some real-life examples of how GenAI is used in the Advertising industry"

# STEP 7: Format the request payload using the model's native structure.
native_request = {
    "inputText": prompt,
    "textGenerationConfig": {
        "maxTokenCount": 512,
        "temperature": 0.5,
        "topP" : 0.9
    },
}

# STEP 8: Convert the native request to JSON.
request = json.dumps(native_request)

# STEP 9: Invoke the model with the request
try:
    # Invoke the model with the request.
    response = client.invoke_model(modelId=model_id, body=request)

except (ClientError, Exception) as e:
    print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
    exit(1)

# STEP 10: Extract and print the response text
model_response = json.loads(response["body"].read())
response_text = model_response["results"][0]["outputText"]
print(response_text)

# STEP 11: Save the response to a file
try:
    with open("./Getting_Started/text_response.txt", "w", encoding="utf-8") as file:
        file.write(response_text)
    print("Response saved to response.txt")
except Exception as e:
    print(f"An error occurred while saving the response: {e}")
