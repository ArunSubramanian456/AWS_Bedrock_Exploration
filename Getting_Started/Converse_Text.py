# STEP 1 : Configure your access via AWS CLI  in your local computer.
# STEP 2: Setup virtual environment and run `pip install -r requirements.txt` to install the required dependencies.
# STEP 3: Import the required libraries.
# Reference - https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-runtime_example_bedrock-runtime_InvokeModel_TitanText_section.html
import boto3
import json
import logging

from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logging.basicConfig(filename='./Getting_Started/Converse_Text_Log.txt', level=logging.INFO,
                        format="%(asctime)s - %(levelname)s: %(message)s")



# STEP 4 : Create a Chatbot function
def generate_conversation(bedrock_client,
                          model_id,
                          system_prompts,
                          messages):
    """
    Sends messages to a model.
    Args:
        bedrock_client: The Boto3 Bedrock runtime client.
        model_id (str): The model ID to use.
        system_prompts (JSON) : The system prompts for the model to use.
        messages (JSON) : The messages to send to the model.

    Returns:
        response (JSON): The conversation that the model generated.

    """

    logger.info("Generating message with model %s", model_id)

    # Inference parameters to use.
    temperature = 0.5
    top_k = 200

    # Base inference parameters to use.
    inference_config = {"temperature": temperature}
    # Additional inference parameters to use.
    additional_model_fields = {"top_k": top_k}

    # Send the message.
    response = bedrock_client.converse(
        modelId=model_id,
        messages=messages,
        system=system_prompts,
        inferenceConfig=inference_config,
        additionalModelRequestFields=additional_model_fields
    )

    # Log token usage.
    token_usage = response['usage']
    logger.info("Input tokens: %s", token_usage['inputTokens'])
    logger.info("Output tokens: %s", token_usage['outputTokens'])
    logger.info("Total tokens: %s", token_usage['totalTokens'])
    logger.info("Stop reason: %s", response['stopReason'])

    return response



# STEP 5: Create a Bedrock Runtime client in the AWS Region of your choice.
client = boto3.client("bedrock-runtime", region_name="us-east-1")

# STEP 6: Set the model ID, e.g., Anthropic Claude 3 Sonnet.
model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

# STEP 7: Setup the system prompts and messages to send to the model.

system_prompts = [{"text": "You are an Financial expert that recommends US S&P500 top Tech stocks to buy based on their future earnings growth estimates."
                       "Only return tech stocks have market cap greater than 100 billion and are expected to grow by at least 10 percent in the next 5 years."}]

message_1 = {
    "role": "user",
    "content": [{"text": "Create a list of 10 stocks to buy in 2025."}]
}

message_2 = {
    "role": "user",
    "content": [{"text": "Make sure the stocks are from the US S&P500 Tech industry with market cap greater than 100 billion and are expected to grow by at least 10 percent in the next 5 years."}]
}

messages = []

# STEP 8: Initiate the converse model
try:
    # Log system prompts
    logger.info("System Prompts:")
    for prompt in system_prompts:
        logger.info(f"{prompt['text']}")

    # Start the conversation with the 1st message.
    messages.append(message_1)
    logger.info(f"User: {message_1['content'][0]['text']}")
    response = generate_conversation(client, model_id, system_prompts, messages)

    # Add the response message to the conversation.
    output_message = response['output']['message']
    messages.append(output_message)
    logger.info(f"Assistant: {output_message['content'][0]['text']}")

    # Continue the conversation with the 2nd message.
    messages.append(message_2)
    logger.info(f"User: {message_2['content'][0]['text']}")
    response = generate_conversation(client, model_id, system_prompts, messages)

    output_message = response['output']['message']
    messages.append(output_message)
    logger.info(f"Assistant: {output_message['content'][0]['text']}")

    # Show the complete conversation.
    for message in messages:
        print(f"Role: {message['role']}")
        for content in message['content']:
            print(f"Text: {content['text']}")
        print()

except ClientError as err:
    message = err.response['Error']['Message']
    logger.error("A client error occurred: %s", message)
    print(f"A client error occured: {message}")

else:
    print(f"Finished generating text with model {model_id}.")