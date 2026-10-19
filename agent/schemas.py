from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field

class HiringRecommendation(str, Enum):
    STRONG_HIRE = "Strong Hire"
    BORDERLINE = "Borderline"
    REJECT = "Reject"

class CandidateEvaluation(BaseModel):
    candidate_id: int = Field(..., description="Unique ID of the candidate evaluated")
    match_score: float = Field(..., ge=0.0, le=100.0, description="Overall fit score from 0 to 100")
    recommendation: HiringRecommendation = Field(..., description="Actionable hiring decision")
    matched_skills: List[str] = Field(..., description="Required skills confirmed present in CV text")
    missing_skills: List[str] = Field(..., description="Required skills missing or unverified")
    text_evidence: str = Field(..., description="Direct quote or specific excerpt from CV proving technical fit")
    summary_rationale: str = Field(..., max_length=300, description="Concise justification for the hiring manager")

class RecruiterAgentResponse(BaseModel):
    query_analyzed: str = Field(..., description="The original recruiter requirement or job description")
    retrieved_candidate_count: int = Field(..., description="Number of candidates retrieved from Qdrant")
    evaluations: List[CandidateEvaluation] = Field(..., description="Structured evaluation for each retrieved candidate")