# AWS Bedrock- Exploration

This repository contains a collection of projects that explore various use cases of AWS Bedrock. Each project is designed to demonstrate different features and capabilities of Bedrock. As I add more projects exploring more features, I plan to continuously update this repo.

## Introduction

Amazon Bedrock is a fully managed service that makes high-performing foundation models (FMs) from leading AI companies and Amazon available for your use through a unified API. You can choose from a wide range of foundation models to find the model that is best suited for your use case. Amazon Bedrock also offers a broad set of capabilities to build generative AI applications with security, privacy, and responsible AI. Using Amazon Bedrock, you can easily experiment with and evaluate top foundation models for your use cases, privately customize them with your data using techniques such as fine-tuning and Retrieval Augmented Generation (RAG), and build agents that execute tasks using your enterprise systems and data sources.

You can use Amazon Bedrock to do the following:
* Experiment with prompts and configurations
* Augment response generation with information from your data sources
* Create applications that reason through how to help a customer
* Adapt models to specific tasks and domains with training data
* Improve your FM-based application's efficiency and output
* Determine the best model for your use case
* Prevent inappropriate or unwanted content 
* Optimize your FM's latency

## Project Structure

*   `subfolder1/`: Description of project 1
    *   `<filename>.py`: Python file(s) to execute
    *   `.txt` or `.jpg`: Output file(s) that contains the response from LLMs
    *   `README.md` file that explains the specific project

*   `subfolder2/`: Description of project 2
    *   `<filename>.py`: Python file(s) to execute
    *   `.txt` or `.jpg`: Output file(s) that contains the response from LLMs
    *   `README.md` file that explains the specific projects

## Setup

Within the parent folder :
1. Create a virtual environment: `conda create -p my_venv python==3.12 -y` if you have Anaconda or `python -m venv my_venv` on Windows w/o Anaconda
2. Add a `.gitignore` file and update it with relevant exclusions

For each subproject:

1.  Navigate to the subfolder: `cd subfolder1`
2.  Activate the environment: `conda activate <path>\my_venv` if you have Anaconda or `<path>\my_venv\Scripts\activate`  on Windows w/o Anaconda
3.  Install dependencies: `pip install -r requirements.txt`

## Pre-requisites

* You need to have AWS Free-tier access
* Using IAM, create a user and attach relevant permission policies required (eg., AmazonBedrockFullAccess )
* Configure your access via AWS CLI. Refer to the documentation [here](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html)

![logo](https://github.com/ArunSubramanian456/AWS_Bedrock_Exploration/blob/main/AWS_Bedrock.png?raw=true)
