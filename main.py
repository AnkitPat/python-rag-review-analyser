from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
import hashlib
import json
import os
from contextlib import asynccontextmanager
from langchain_core.documents import Document


print(os.getenv('OPENAI_API_KEY'))

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🔧 Building vector store...")
    memory_builder()
    yield
    print("🛑 Shutting down...")

# Initialize FastAPI
app = FastAPI(lifespan=lifespan)

persist_dir = "./chroma_reviews_db"

def get_embedding_function():
    return OpenAIEmbeddings(model="text-embedding-3-small")

def memory_builder():
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=get_embedding_function())
    
    # Function to load documents
    def load_documents():
        loader = JSONLoader('./data/dummy.json', jq_schema='.[]', text_content=False)
        documents = loader.load()
        return documents

    # Function to add documents to Chroma vector store
    def add_to_chroma(documents):
        existing_ids = set(vectordb._collection.get()["ids"])
        new_ids = []
        new_documents = []

        for document in documents:
            content_dict = json.loads(document.page_content)
            doc_id = hashlib.md5(content_dict['review'].encode()).hexdigest()
            print(doc_id)
            if doc_id not in existing_ids:
                new_ids.append(doc_id)
                document.metadata["id"] = doc_id
                new_documents.append(document)

        if len(new_documents):
            vectordb.add_documents(new_documents, ids=new_ids)
            vectordb.persist()
            print(f"Document added: {len(new_documents)}")
        else:
            print("✅ No new documents to add")

    documents = load_documents()
    print(documents)
    add_to_chroma(documents)

# Pydantic model for the input request to query reviews
class QueryRequest(BaseModel):
    text: str  # Text to query against the reviews
    k: int = 3  # Number of top results to return


# Function to query reviews and get a response
def query_review(text, k=3):
    db = Chroma(persist_directory=persist_dir, embedding_function=get_embedding_function())
    results = db.similarity_search_with_relevance_scores(text, k=k)
    
    context = '\n\n___\n\n'.join(
        [f"{i+1}. Review: {data['review']} \n Tags: {(', ').join(data['tags'])} \n Mood: {data['mood']} \n Moderate: {data['moderate']}"
         for i, (doc, _score) in enumerate(results)
         if (data := json.loads(doc.page_content))]
    )

    TEMPLATE = """
      You are a review moderation assistant. Analyze the given review and provide moderation suggestion, key tags, and mood.

      Review:
      "{user_review}"

      Here are similar past reviews and how they were handled:
      {last_reviews}

      Respond in JSON:
      {{
        "moderate": true | false,
        "tags": [tag1, tag2, ...],
        "mood": "calm" | "frustrated" | "sarcastic" | ...
      }}
    """

    prompt = ChatPromptTemplate.from_template(TEMPLATE)
    prompt = prompt.format_prompt(user_review=text, last_reviews=context)
    model = OpenAI()
    response = model.invoke(prompt)
    parsedReponse = json.loads(response)
    doc_id = hashlib.md5(text.encode()).hexdigest()
    db.add_documents([Document(page_content=json.dumps({"review": text, "tags": parsedReponse["tags"], "mood": parsedReponse["mood"], "moderate": parsedReponse["moderate"]}), metadata={"id": doc_id})], ids=[doc_id])
    print(response, 'response')
    print("Added one more document")
    return response

@app.post("/query_review/")
async def query_review_endpoint(request: QueryRequest):
    try:
        response = query_review(request.text, request.k)
        
        return {"response": json.loads(response)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/len_documents/")
async def get_documents_length():
    db = Chroma(persist_directory=persist_dir, embedding_function=get_embedding_function())
    num_docs = len(db._collection.get()["ids"])
    return {"document length": num_docs}

# Run with: uvicorn <filename>:app --reload
