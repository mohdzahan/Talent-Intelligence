import os
import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import types
from agent.schemas import RecruiterAgentResponse

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TALENT_API_URL = os.getenv("TALENT_API_URL", "http://localhost:8000/search")

client = genai.Client(api_key=GEMINI_API_KEY)

async def query_talent_search_tool(query: str, limit: int = 3) -> list:
    """External HTTP tool call to the live EC2 FastAPI/Qdrant hybrid retrieval service."""
    async with httpx.AsyncClient(timeout=15.0) as http_client:
        response = await http_client.post(
            TALENT_API_URL,
            json={"query": query, "limit": limit, "rerank": True}
        )
        response.raise_for_status()
        data = response.json()
        return data.get("results", [])

def evaluate_candidates_with_llm(job_brief: str, candidates: list) -> RecruiterAgentResponse:
    """Uses structured outputs to evaluate retrieved candidate evidence against the hiring brief."""
    candidates_context = "\n\n".join([
        f"--- CANDIDATE ID: {idx + 1} ---\n"
        f"Cross-Encoder Score: {c.get('score', 0.0):.4f}\n"
        f"Extracted Skills: {', '.join(c.get('skills', []))}\n"
        f"CV Text: {c.get('text', '')}"
        for idx, c in enumerate(candidates)
    ])

    prompt = (
        f"HIRING REQUIREMENT:\n{job_brief}\n\n"
        f"RETRIEVED CANDIDATE PROFILES:\n{candidates_context}\n\n"
        "You are an enterprise AI Technical Recruiter. Evaluate every candidate based ONLY on the text evidence provided. "
        "Strictly grade technical alignment, flag missing prerequisites, and provide decisive hiring decisions."
    )

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RecruiterAgentResponse,
            temperature=0.1,
        ),
    )

    return RecruiterAgentResponse.model_validate_json(response.text)

async def run_recruiter_agent(job_brief: str, limit: int = 3) -> RecruiterAgentResponse:
    print(f"[Agent] Analyzing requirement: '{job_brief}'")
    print(f"[Agent] Calling live EC2 Retrieval Tool at: {TALENT_API_URL}")
    candidates = await query_talent_search_tool(query=job_brief, limit=limit)
    print(f"[Agent] Retrieved {len(candidates)} candidates from hybrid search. Generating evaluations...")
    return evaluate_candidates_with_llm(job_brief, candidates)