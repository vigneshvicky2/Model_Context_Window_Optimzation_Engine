from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import tiktoken
import os
from groq import Groq
from sentence_transformers import SentenceTransformer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

class MiniVectorDB:
    def __init__(self):
        self.embeddings = []
        self.documents = []

    def add(self, text, embedding):
        self.documents.append(text)
        self.embeddings.append(embedding)

    def search(self, query_embedding, top_k=2):
        if not self.embeddings: return []
        # Cosine similarity for the local embeddings
        similarities = [
            np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
            for emb in self.embeddings
        ]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(self.documents[i], similarities[i]) for i in top_indices]

class ContextOrchestrator:
    def __init__(self):
        # Initialize Groq client
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # Initialize Local Embedding Model
        print("Loading local embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.db = MiniVectorDB()
        
        self.tokenizer = tiktoken.get_encoding("cl100k_base") 
        
        # [FIXED] Increased Token Budget so it doesn't instantly drop large responses
        self.TOKEN_BUDGET = 4000 
        
    def get_embedding(self, text):
        # Generate embeddings locally as a NumPy array
        return self.embedding_model.encode(text)

    def process_query(self, user_query):
        query_embedding = self.get_embedding(user_query)
        raw_memories = self.db.search(query_embedding, top_k=3)
        
        filtered_memories = [mem[0] for mem in raw_memories if mem[1] > 0.3] 

        optimized_context = ""
        current_tokens = len(self.tokenizer.encode(user_query))
        
        for mem in filtered_memories:
            mem_tokens = len(self.tokenizer.encode(mem))
            if current_tokens + mem_tokens < self.TOKEN_BUDGET:
                optimized_context += f"- {mem}\n"
                current_tokens += mem_tokens
            else: 
                break 

        system_prompt = (
            "You are a highly capable AI assistant. Use the following retrieved memory "
            "to answer the user if relevant.\n\n"
            f"Relevant Memory:\n{optimized_context if optimized_context else 'No relevant memory found.'}"
        )

        # [FIXED] Using the correct model string for Groq's Qwen endpoint
        response = self.groq_client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]
        )
        llm_response = response.choices[0].message.content

        new_memory = f"User asked: {user_query} | AI replied: {llm_response}"
        self.db.add(new_memory, self.get_embedding(new_memory))

        return {"response": llm_response, "context_used": optimized_context}

# Initialize orchestrator
orchestrator = ContextOrchestrator()

class QueryRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    try:
        result = orchestrator.process_query(request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))