# AI-Powered Medical Assistant......
An AI-powered medical chatbot built with Flask, Pinecone, HuggingFace Embeddings, and Gemini. It retrieves medical knowledge via vector search and generates accurate, context-aware answers in real time.


# How to run?
### STEPS:



bash
Project repo: https://github.com/

### STEP 01- Create a conda environment after opening the repository

bash
conda create -n medibot python=3.10 -y


bash
conda activate medibot



### STEP 02- install the requirements
bash
pip install -r requirements.txt



### Create a .env file in the root directory and add your Pinecone & openai credentials as follows:

ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
GEMINI_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"



bash
# run the following command to store embeddings to pinecone
python store_index.py


bash
# Finally run the following command
python app.py


Now,
bash
open up localhost:



### Techstack Used:

- Python
- LangChain
- Flask
- GEMINI
- Pinecone

