from flask import Flask, render_template, request
import google.generativeai as genai
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")   # or gemini-2.5-flash if enabled for you


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


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

  
    response = model.generate_content(prompt_text)
    return str(response.text)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  
    app.run(host="0.0.0.0", port=port, debug=True)
