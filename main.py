from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import asyncio
import os
import tempfile
import traceback
from typing import Any

from dotenv import load_dotenv
load_dotenv()

# Neo4j
from neo4j import GraphDatabase

# neo4j-graphrag
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.retrievers import HybridRetriever, VectorCypherRetriever
from neo4j_graphrag.indexes import create_vector_index

# Gemini (new unified SDK)
from google import genai
from google.genai import types

# neo4j-graphrag LLM interface
from neo4j_graphrag.llm import LLMInterface, LLMResponse

# Hugging Face embeddings (local, free)
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings


NEO4J_URI = "neo4j+s://76e40fd9.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

GEMINI_MODEL = "gemini-2.5-flash"          # or "gemini-2.5-pro" if you have access

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

VECTOR_INDEX_NAME = "chunk_embedding"
VECTOR_DIMENSIONS = 384   # bge-small-en-v1.5 = 384


class GeminiLLM(LLMInterface):
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        api_key: str = None,
        system_prompt: str = None
    ):
        self.client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY"))
        self.model_name = model_name
        self.system_prompt = system_prompt or (
            "You are a helpful assistant that answers questions based only on the provided document content. "
            "Be accurate, concise, and do not add information that is not in the text. "
            "If the answer is not clear from the document, say 'Not enough information in the document'."
        )

    def invoke(self, *args, **kwargs) -> LLMResponse:
        # Extract input (first positional arg is usually the prompt)
        if args:
            input_text = args[0]
        else:
            input_text = kwargs.get("input") or kwargs.get("prompt", "")

        # Get system instruction from kwargs (preferred) or fallback
        system_instruction = kwargs.get("system_instruction") or kwargs.get("system_prompt")

        try:
            messages = []

            if system_instruction:
                messages.append({
                    "role": "model",
                    "parts": [{"text": system_instruction}]
                })

            messages.append({
                "role": "user",
                "parts": [{"text": input_text}]
            })

            config = types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=4096,
                response_mime_type="application/json"
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=messages,
                config=config
            )

            text = response.text.strip()

            # Cleanup
            if text.startswith(("```json", "```")):
                text = text.split("```", 2)[1].strip() if "```" in text else text

            return LLMResponse(content=text or '{}')

        except Exception as e:
            return LLMResponse(content=f'{{"error": "{str(e).replace('"', '\\"')}"}}')

    async def ainvoke(self, *args, **kwargs) -> LLMResponse:
        return self.invoke(*args, **kwargs)
app = FastAPI(title="PDF â†’ Knowledge Graph â†’ Q&A (Hugging Face + Gemini)")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

llm = GeminiLLM(model_name=GEMINI_MODEL)

# Hugging Face embeddings (local & free)
embedder = SentenceTransformerEmbeddings("BAAI/bge-small-en-v1.5")

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, detail="Only PDF files are supported")

    tmp_path = None
    try:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        print(f"[UPLOAD] Temp file: {tmp_path} ({os.path.getsize(tmp_path)} bytes)")

        kg_builder = SimpleKGPipeline(
            llm=llm,
            driver=driver,
            embedder=embedder,
            text_splitter=FixedSizeSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            ),
            from_pdf=True,
            perform_entity_resolution=True,
            on_error="IGNORE",

            entities=["Person", "Organization", "Location", "Date", "Product", "Skill", "Company", "Document"],
            relations=["WORKS_FOR", "LOCATED_IN", "HAS_SKILL", "MENTIONS", "IN_YEAR", "RELATED_TO", "PART_OF"],
        )

        print("[UPLOAD] Starting knowledge graph pipeline...")
        result = await kg_builder.run_async(
            file_path=tmp_path,
            document_metadata={
                "filename": file.filename,
                "uploaded_at": "2026-01-19",
                "source": "api_upload"
            }
        )
        print("[UPLOAD] Pipeline result:", result)

        # Create / ensure vector index
        try:
            create_vector_index(
                driver=driver,
                name=VECTOR_INDEX_NAME,
                label="Chunk",
                embedding_property="embedding",
                dimensions=VECTOR_DIMENSIONS,
                similarity_fn="cosine",
            )
            print("[UPLOAD] Vector index ready")
        except Exception as idx_err:
            print("[UPLOAD] Vector index warning (might already exist):", str(idx_err))

        return {
            "status": "success",
            "message": f"Document '{file.filename}' processed successfully",
            "result": str(result)
        }

    except Exception as e:
        full_tb = traceback.format_exc()
        print("[UPLOAD] ERROR:\n", full_tb)
        raise HTTPException(500, detail=f"Processing failed: {str(e)}\n\nTraceback:\n{full_tb}")

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
                print("[UPLOAD] Temp file removed")
            except:
                pass


class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5


# @app.post("/ask")
# async def ask_question(req: QuestionRequest):
#     try:
#         retriever = HybridRetriever(
#             driver=driver,
#             vector_index_name=VECTOR_INDEX_NAME,
#             embedder=embedder,
#             cypher_retriever = VectorCypherRetriever(
#                 driver=driver,
#                 index_name="chunk_embedding",
#                 embedder=embedder,
#                 retrieval_query="""
#                 MATCH (chunk:Chunk)
#                 CALL db.index.vector.queryNodes('chunk_embedding', $top_k, $embedding) YIELD node, score
#                 WHERE node = chunk
#                 OPTIONAL MATCH (chunk)-[:MENTIONS]->(entity)
#                 WITH chunk, score, collect(entity) AS entities
#                 RETURN 
#                     chunk.text AS text,
#                     score,
#                     chunk {.*} AS metadata,
#                     [e IN entities | e.name] AS mentioned_entities
#                 ORDER BY score DESC
#                 """
#             )
#         )

#         rag = GraphRAG(
#             llm=llm,
#             retriever=retriever
#         )

#         print(f"[ASK] Processing question: {req.question}")
#         response = await rag.search(
#             query_text=req.question,
#             retriever_config={"top_k": req.top_k}
#         )

#         return {
#             "question": req.question,
#             "answer": response.answer,
#             "context_count": len(response.retrieved_contexts) if hasattr(response, 'retrieved_contexts') else 0,
#         }

#     except Exception as e:
#         full_tb = traceback.format_exc()
#         print("[ASK] ERROR:\n", full_tb)
#         raise HTTPException(500, detail=f"Query failed: {str(e)}")


@app.post("/ask")
async def ask_question(req: QuestionRequest):
    try:
        # Option A: Simple hybrid (vector + full-text) - recommended first
        retriever = HybridRetriever(
            driver=driver,
            vector_index_name="chunk_embedding",
            fulltext_index_name="fulltext_chunks",  # create this index!
            embedder=embedder,
        )

        # Option B: Hybrid + Cypher traversal (uncomment if you want more graph power)
        # from neo4j_graphrag.retrievers import HybridCypherRetriever
        # retriever = HybridCypherRetriever(
        #     driver=driver,
        #     vector_index_name="chunk_embedding",
        #     fulltext_index_name="fulltext_chunks",
        #     retrieval_query="""
        #         MATCH (chunk:Chunk)
        #         CALL db.index.vector.queryNodes('chunk_embedding', $top_k, $embedding) 
        #             YIELD node, score
        #         WHERE node = chunk
        #         RETURN 
        #             chunk.text AS text, 
        #             score, 
        #             chunk {.*} AS metadata
        #         ORDER BY score DESC
        #         LIMIT $top_k
        #     """,
        #     embedder=embedder,
        # )

        rag = GraphRAG(
            llm=llm,
            retriever=retriever
        )

        print(f"[ASK] Processing question: {req.question}")
        response =  rag.search(
            query_text=req.question,
            retriever_config={"top_k": req.top_k}
        )

        return {
            "question": req.question,
            "answer": response.answer,
            "context_count": len(response.retrieved_contexts) if hasattr(response, 'retrieved_contexts') else 0,
        }

    except Exception as e:
        full_tb = traceback.format_exc()
        print("[ASK] ERROR:\n", full_tb)
        raise HTTPException(500, detail=f"Query failed: {str(e)}\n\nTraceback:\n{full_tb}")


@app.get("/health")
async def health_check():
    try:
        driver.verify_connectivity()
        return {"status": "healthy", "neo4j": "connected", "message": "OK"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}



if __name__ == "__main__":
    import uvicorn
    print("Starting server â†’ http://127.0.0.1:8000/docs")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )