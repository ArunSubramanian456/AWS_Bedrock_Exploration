# STEP 1 : Configure your access via AWS CLI  in your local computer.
# STEP 2: Setup virtual environment and run `pip install -r requirements.txt` to install the required dependencies.
# STEP 3: Import the required libraries.
# Reference - https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference-examples.html
import logging
import boto3


from botocore.exceptions import ClientError


logger = logging.getLogger(__name__)
logging.basicConfig(filename='./Getting_Started/Converse_Document_Log.txt', level=logging.INFO,
                        format="%(asctime)s - %(levelname)s: %(message)s")


def generate_message(bedrock_client,
                     model_id,
                     input_text,
                     input_document):
    """
    Sends a message to a model.
    Args:
        bedrock_client: The Boto3 Bedrock runtime client.
        model_id (str): The model ID to use.
        input text : The input message.
        input_document : The input document.

    Returns:
        response (JSON): The conversation that the model generated.

    """

    logger.info("Generating message with model %s", model_id)

    # Message to send.

    message = {
        "role": "user",
        "content": [
            {
                "text": input_text
            },
            {
                "document": {
                    "name": "MyDocument",
                    "format": "txt",
                    "source": {
                        "bytes": input_document
                    }
                }
            }
        ]
    }

    messages = [message]

    # Send the message.
    response = bedrock_client.converse(
        modelId=model_id,
        messages=messages
    )

    return response


def main():
    """
    Entrypoint for Anthropic Claude 3 Sonnet example.
    """

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")

    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    input_text = "What's in this document?"
    with open("./Getting_Started/input_document.txt", "rb") as f:
        input_document = f.read()

    try:
        logger.info(f"User: {input_text}" )
        logger.info(f"Document: {input_document}")

        bedrock_client = boto3.client(service_name="bedrock-runtime")

        response = generate_message(
            bedrock_client, model_id, input_text, input_document)

        output_message = response['output']['message']

        print(f"Role: {output_message['role']}")
        logger.info(f"{output_message['role']}")

        for content in output_message['content']:
            logger.info(f"Text: {content['text']}")
            print(f"Text: {content['text']}")

        token_usage = response['usage']
        logger.info(f"Input tokens:  {token_usage['inputTokens']}")
        logger.info(f"Output tokens:  {token_usage['outputTokens']}")
        logger.info(f"Total tokens:  {token_usage['totalTokens']}")
        logger.info(f"Stop reason: {response['stopReason']}")

    except ClientError as err:
        message = err.response['Error']['Message']
        logger.error("A client error occurred: %s", message)
        print(f"A client error occured: {message}")

    else:
        print(
            f"Finished generating text with model {model_id}.")


if __name__ == "__main__":
    main()
