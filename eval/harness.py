import json
import os
import httpx
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall

def evaluate_pipeline():

    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("You must export OPENAI_API_KEY in your terminal.")

    with open("eval/qa_pairs.json", "r") as f:
        data = json.load(f)

    questions, ground_truths, contexts = [], [], []

    print("Querying FastAPI for contexts...")
    for item in data:
        # Hit your live FastAPI endpoint
        response = httpx.post(
            "http://localhost:8000/search", 
            json={"query": item["question"], "limit": 3},
            timeout=10.0
        )
        
        # Extract the actual text from the API response
        retrieved_docs = [res["text"] for res in response.json()["results"]]
        
        questions.append(item["question"])
        # Ragas expects ground truth to be a list of strings
        ground_truths.append([item["ground_truth"]]) 
        contexts.append(retrieved_docs)

    # Format into a Hugging Face dataset
    dataset = Dataset.from_dict({
        "question": questions,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    print("Running RAGAS evaluation...")
    result = evaluate(
        dataset=dataset,
        metrics=[context_precision, context_recall]
    )
    print("\n=== Baseline Scores ===")
    print(result)

if __name__ == "__main__":
    evaluate_pipeline()