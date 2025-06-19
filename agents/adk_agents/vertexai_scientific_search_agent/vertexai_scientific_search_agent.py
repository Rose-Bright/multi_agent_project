from ..callback_logging import log_query_to_model, log_model_response

from google.adk import Agent
from google.adk.memory import VertexAiRagMemoryService

# Vertex AI RAG Corpus resource name
rag_corpus_resource_name = "projects/genai-product-matching/locations/us-central1/ragCorpora/4611686018427387904"
similarity_top_k = 5
vector_distance_threshold = 0.7

rag_memory_service = VertexAiRagMemoryService(
    rag_corpus=rag_corpus_resource_name,
    similarity_top_k=similarity_top_k,
    vector_distance_threshold=vector_distance_threshold
)

def search_scientific_corpus(query: str):
    """Search the scientific corpus using VertexAiRagMemoryService."""
    return rag_memory_service(query)

# Root Agent
vertexai_scientific_search_agent = Agent(
   name="vertexai_scientific_search_agent",
   model="gemini-2.0-flash-001",
   description="Answer questions using your data store access.",
   instruction="You analyze new planet discoveries and engage with the scientific community on them.",
   before_model_callback=log_query_to_model,
   after_model_callback=log_model_response,
   tools=[search_scientific_corpus]  # Use the function, not the service object
)
