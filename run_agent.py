import asyncio
import json
from agent.orchestrator import run_recruiter_agent

async def main():
    test_brief = "Need an experienced Data Engineer with PySpark and AWS Glue to build big data pipelines."
    
    result = await run_recruiter_agent(job_brief=test_brief, limit=2)
    
    print("\n" + "=" * 60)
    print("RECRUITER AGENT EVALUATION REPORT")
    print("=" * 60)
    print(f"Query: {result.query_analyzed}")
    print(f"Candidates Evaluated: {result.retrieved_candidate_count}\n")
    
    for eval in result.evaluations:
        print(f"Candidate #{eval.candidate_id} | Score: {eval.match_score}/100 | Decision: {eval.recommendation.value}")
        print(f"  Matched Skills: {eval.matched_skills}")
        print(f"  Missing Skills: {eval.missing_skills}")
        print(f"  Evidence: \"{eval.text_evidence}\"")
        print(f"  Rationale: {eval.summary_rationale}\n")

if __name__ == "__main__":
    asyncio.run(main())