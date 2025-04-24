from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
import hashlib
import json
import os
from contextlib import asynccontextmanager
from langchain_core.documents import Document
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import RetrievalQA
from embedding_cache import EmbeddingCache


print(os.getenv('OPENAI_API_KEY'))

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🔧 Building vector store...")
    memory_builder()
    yield
    print("🛑 Shutting down...")

# Initialize FastAPI
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["https://your-frontend-site.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
persist_dir = "./chroma_reviews_db"

def get_embedding_function():
    return OpenAIEmbeddings(model="text-embedding-3-small")

# def memory_builder():
#     vectordb = Chroma(persist_directory=persist_dir, embedding_function=get_embedding_function())
    
#     # Function to load documents
#     def load_documents():
#         loader = JSONLoader('./data/dummy.json', jq_schema='.[]', text_content=False)
#         documents = loader.load()
#         return documents

#     # Function to add documents to Chroma vector store
#     def add_to_chroma(documents):
#         existing_ids = set(vectordb._collection.get()["ids"])
#         new_ids = []
#         new_documents = []

#         for document in documents:
#             content_dict = json.loads(document.page_content)
#             doc_id = hashlib.md5(content_dict['review'].encode()).hexdigest()
#             print(doc_id)
#             if doc_id not in existing_ids:
#                 new_ids.append(doc_id)
#                 document.metadata["id"] = doc_id
#                 new_documents.append(document)

#         if len(new_documents):
#             vectordb.add_documents(new_documents, ids=new_ids)
#             vectordb.persist()
#             print(f"Document added: {len(new_documents)}")
#         else:
#             print("✅ No new documents to add")

#     documents = load_documents()
#     print(documents)
#     add_to_chroma(documents)


cache = EmbeddingCache()  # In-memory cache

def memory_builder():
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=get_embedding_function())
    existing_ids = set(vectordb._collection.get()["ids"])

    def load_documents():
        loader = JSONLoader('./data/dummy.json', jq_schema='.[]', text_content=False)
        return loader.load()

    documents = load_documents()

    new_documents = []
    new_ids = []
    texts_to_embed = []
    final_embeddings = []

    for doc in documents:
        content_dict = json.loads(doc.page_content)
        review_text = content_dict["review"]
        doc_id = cache.get_hash(review_text)

        if doc_id not in existing_ids:
            embedding = cache.get(review_text)
            if embedding is not None:
                final_embeddings.append(embedding)
            else:
                texts_to_embed.append(review_text)

            new_ids.append(doc_id)
            doc.metadata["id"] = doc_id
            new_documents.append(doc)

    if not new_documents:
        print("✅ No new documents to add.")
        return

    # Only compute embeddings for uncached texts
    if texts_to_embed:
        embedding_model = get_embedding_function()
        new_embeddings = embedding_model.embed_documents(texts_to_embed)

        for text, embedding in zip(texts_to_embed, new_embeddings):
            cache.set(text, embedding)
            final_embeddings.append(embedding)

    vectordb._collection.add(
        documents=[doc.page_content for doc in new_documents],
        metadatas=[doc.metadata for doc in new_documents],
        embeddings=final_embeddings,
        ids=new_ids,
    )

    vectordb.persist()
    print(f"✅ Added {len(new_documents)} new documents.")


# Pydantic model for the input request to query reviews
class QueryRequest(BaseModel):
    text: str  # Text to query against the reviews
    k: int = 3  # Number of top results to return

# function to get retriever 
def get_retriever():
    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
    )
    return vectordb.as_retriever(search_kwargs={"k": 3})

def query_review_using_retriever(text):
    rag_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
        You are a review moderation assistant.

        Use the following similar reviews to help you analyze the given review.

        {context}

        Review: "{question}"

        Provide your output in this JSON format:
        {{
        "moderate": true | false,
        "tags": [tag1, tag2, ...],
        "mood": "calm" | "frustrated" | "sarcastic" | ...
        }}
        """
    )
    retriever = get_retriever()
    llm = OpenAI(temperature=0)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",  # simple context stuffing
        chain_type_kwargs={"prompt": rag_prompt}
    )
    response = chain.run(text)
    return response



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
        response = query_review_using_retriever(request.text)
        
        return {"response": json.loads(response)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/len_documents/")
async def get_documents_length():
    db = Chroma(persist_directory=persist_dir, embedding_function=get_embedding_function())
    num_docs = len(db._collection.get()["ids"])
    return {"document length": num_docs}

# Run with: uvicorn <filename>:app --reload
