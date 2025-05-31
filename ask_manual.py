import os
from dotenv import load_dotenv

# Load environment variables

from pathlib import Path
load_dotenv(dotenv_path=Path('.') / '.env')
openai_key = os.getenv("OPENAI_API_KEY")
print("🧪 Loaded key:", openai_key)

if not openai_key:
    raise Exception("❌ OPENAI_API_KEY not found. Check your .env file!")

# LangChain imports
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Load saved vector database
embedding = OpenAIEmbeddings(openai_api_key=openai_key)
db = Chroma(persist_directory="fluent_db", embedding_function=embedding)

# Set up GPT-4 for Q&A
llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=openai_key)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

# CLI assistant loop
print("🧪 Fluent Manual Assistant is ready. Ask your question (type 'exit' to quit):\n")

while True:
    query = input("❓: ")
    if query.lower() in ["exit", "quit"]:
        break

    try:
        answer = qa.run(query)
        print(f"\n🧠 {answer}\n")
    except Exception as e:
        print(f"⚠️ Error: {e}")

