
# STEP 1 : Configure your access via AWS CLI  in your local computer.
# STEP 2: Setup virtual environment and run `pip install -r requirements.txt` to install the required dependencies.
# STEP 3: Import the required libraries.
# Reference - https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-image.html

import base64
import io
import json
import boto3
from PIL import Image

from botocore.exceptions import ClientError

# STEP 4: Create a Bedrock Runtime client in the AWS Region of your choice.
bedrock = boto3.client(service_name='bedrock-runtime')

# STEP 5: Set the parameters
model_id = "amazon.titan-image-generator-v1"
accept = "application/json"
content_type = "application/json"

# STEP 6: Define the prompt for the model.
prompt = """A photograph of indian filter coffee in an brass tumbler along with newspaper on a table."""

# STEP 7: Format the request payload using the model's native structure.
body = json.dumps({
    "taskType": "TEXT_IMAGE",
    "textToImageParams": {
        "text": prompt
    },
    "imageGenerationConfig": {
        "numberOfImages": 1,
        "height": 1024,
        "width": 1024,
        "cfgScale": 8.0,  # Specifies how strongly the generated image should adhere to the prompt. Use a lower value to introduce more randomness in the generation
        "seed": 0 # Use to control and reproduce results. Determines the initial noise setting.
    }
})

# STEP 8: Invoke the model with the request
try:
    response = bedrock.invoke_model(
        body=body, modelId=model_id, accept=accept, contentType=content_type
    )

except ClientError as err:
    message = err.response["Error"]["Message"]
    print("A client error occured: " +
            format(message))
    
except ImageError as err:
    print(err.message)

else:
    print(
        f"Finished generating image with Amazon Titan Image Generator G1 model {model_id}.")

# STEP 9: Extract and print the response text
response_body = json.loads(response.get("body").read())
base64_image = response_body.get("images")[0]
base64_bytes = base64_image.encode('ascii')
image_bytes = base64.b64decode(base64_bytes)

if response_body.get("error") is not None:
    raise ImageError(f"Image generation error. Error is {finish_reason}")


image = Image.open(io.BytesIO(image_bytes))
image.show()

# STEP 10: Save the image
image.save("./Getting_Started/image_response.jpg")