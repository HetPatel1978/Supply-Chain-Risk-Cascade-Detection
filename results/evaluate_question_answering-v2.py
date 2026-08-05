import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

from groq import Groq


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
RESULTS_DIR = REPO_ROOT / "results"

JUDGE_MODEL = "openai/gpt-oss-120b"

QUESTIONS = [
    {
        "id": "Q1",
        "type": "direct",
        "question": "How could a disruption at TSMC directly affect Broadcom?",
    },
    {
        "id": "Q2",
        "type": "two_hop",
        "question": "How could a disruption at TSMC affect Huawei through Broadcom?",
    },
    {
        "id": "Q3",
        "type": "multi_hop",
        "question": "How could a disruption at UMC eventually affect Huawei?",
    },
    {
        "id": "Q4",
        "type": "direction",
        "question": (
            "Based on the retrieved relationships, why would "
            "a disruption at TSMC affect AMD?"
        ),
    },
    {
        "id": "Q5",
        "type": "insufficient_evidence",
        "question": (
            "What exact financial loss would AMD experience "
            "from a disruption at TSMC?"
        ),
    },
]


def load_project_module():
    build_context_path = next(SRC_DIR.rglob("build_context.py"))

    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    spec = importlib.util.spec_from_file_location(
        "build_context",
        build_context_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    print(f"Loaded pipeline from: {build_context_path}")
    return module


def run_question(module, graph, question):
    companies = module.find_companies_in_question(question, graph)
    start_company = module.find_company_in_question(question, graph)

    target_company = next(
        (
            company
            for company in companies
            if company != start_company
        ),
        None,
    )

    paths = module.retrieve_evidence(
        graph,
        start_node=start_company,
        max_hops=module.MAX_CASCADE_HOPS,
        max_paths=module.MAX_EVIDENCE_PATHS,
    )

    if target_company:
        paths = [
            path
            for path in paths
            if path["path"][-1] == target_company
        ]

    if not paths:
        raise ValueError("No supported graph path was found.")

    selected_paths = module.select_deepest_paths(
        paths,
        limit=module.MAX_SELECTED_PATHS,
    )

    context = module.build_rag_context(selected_paths)
    prompt = module.build_rag_prompt(question, context)
    answer = module.generate_answer(prompt)

    return {
        "start_company": start_company,
        "target_company": target_company,
        "paths": selected_paths,
        "context": context,
        "answer": answer,
    }


def judge_answer(client, question, context, answer):
    prompt = f"""
Evaluate this supply-chain RAG answer using only the retrieved evidence.

Score each item from 0 to 2:
- context_relevance
- faithfulness
- relationship_direction
- completeness
- answer_relevance

Return only JSON in this format:
{{
  "context_relevance": 0,
  "faithfulness": 0,
  "relationship_direction": 0,
  "completeness": 0,
  "answer_relevance": 0,
  "unsupported_claims": [],
  "direction_errors": [],
  "explanation": ""
}}

Question:
{question}

Retrieved evidence:
{context}

Answer:
{answer}
""".strip()

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content)

    result["total_score"] = sum(
        result[field]
        for field in [
            "context_relevance",
            "faithfulness",
            "relationship_direction",
            "completeness",
            "answer_relevance",
        ]
    )

    return result


def save_results(rows):
    output_path = RESULTS_DIR / "qa_evaluation_results.csv"

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "question_id",
                "question_type",
                "question",
                "start_company",
                "target_company",
                "selected_paths",
                "context_relevance",
                "faithfulness",
                "relationship_direction",
                "completeness",
                "answer_relevance",
                "total_score",
                "unsupported_claims",
                "direction_errors",
                "explanation",
                "answer",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to: {output_path}")


def main():
    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is not set.")

    module = load_project_module()
    triples = module.load_triples(module.MERGED_TRIPLES_PATH)
    graph = module.build_graph(triples)
    client = Groq(api_key=api_key)

    rows = []

    for item in QUESTIONS:
        print(f"\nRunning {item['id']}: {item['question']}")

        result = run_question(
            module,
            graph,
            item["question"],
        )

        evaluation = judge_answer(
            client,
            item["question"],
            result["context"],
            result["answer"],
        )

        print(result["answer"])
        print(
            f"\nJudge score: "
            f"{evaluation['total_score']}/10"
        )

        rows.append(
            {
                "question_id": item["id"],
                "question_type": item["type"],
                "question": item["question"],
                "start_company": result["start_company"],
                "target_company": result["target_company"] or "",
                "selected_paths": len(result["paths"]),
                "context_relevance": evaluation["context_relevance"],
                "faithfulness": evaluation["faithfulness"],
                "relationship_direction": evaluation[
                    "relationship_direction"
                ],
                "completeness": evaluation["completeness"],
                "answer_relevance": evaluation["answer_relevance"],
                "total_score": evaluation["total_score"],
                "unsupported_claims": "; ".join(
                    evaluation["unsupported_claims"]
                ),
                "direction_errors": "; ".join(
                    evaluation["direction_errors"]
                ),
                "explanation": evaluation["explanation"],
                "answer": result["answer"],
            }
        )

    save_results(rows)

    average = sum(
        row["total_score"]
        for row in rows
    ) / len(rows)

    print(f"Average score: {average:.1f}/10")


if __name__ == "__main__":
    main()
