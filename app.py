from flask import Flask, render_template, request
import google.generativeai as genai
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Pinecone index
index_name = "medicalbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    user_msg = request.form["msg"]
    print("User Input:", user_msg)

    docs = retriever.get_relevant_documents(user_msg)
    context_text = "\n".join([doc.page_content for doc in docs])

    prompt_text = (
        "You are a medical assistant. "
        "Use the following context to answer the question concisely (max 3 sentences). "
        "If you don't know the answer, say that you don't know.\n\n"
        f"{context_text}\n\n"
        f"Question: {user_msg}"
    )

    # New Google Generative AI API usage
    response = genai.chat(
        model="gemini-2.0",
        messages=[{"role": "user", "content": prompt_text}]
    )

    return str(response.last)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  
    app.run(host="0.0.0.0", port=port, debug=True)
