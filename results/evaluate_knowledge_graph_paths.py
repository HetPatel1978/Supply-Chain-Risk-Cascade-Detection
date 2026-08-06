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

PATHS_TO_EVALUATE = [
    ["TSMC", "Broadcom"],
    ["TSMC", "AMD"],
    ["TSMC", "Broadcom", "Huawei"],
    ["UMC", "AMD", "TSMC", "Broadcom", "Huawei"],
    [
        "UMC",
        "GLOBALFOUNDRIES",
        "AMD",
        "TSMC",
        "Broadcom",
        "Huawei",
    ],
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

    print(f"Loaded graph pipeline from: {build_context_path}")
    return module


def find_path(module, graph, expected_path):
    all_paths = module.retrieve_evidence(
        graph,
        start_node=expected_path[0],
        max_hops=module.MAX_CASCADE_HOPS,
        max_paths=module.MAX_EVIDENCE_PATHS,
    )

    for result in all_paths:
        if result["path"] == expected_path:
            return result

    raise ValueError(
        f"Path not found: {' -> '.join(expected_path)}"
    )


def format_path(path_result):
    path = path_result["path"]
    relationships = path_result["relationships"]

    lines = [
        f"Path: {' -> '.join(path)}",
        f"Hop count: {len(path) - 1}",
        "",
    ]

    for step_index in range(len(path) - 1):
        left = path[step_index]
        right = path[step_index + 1]

        step_relationships = [
            relationship
            for relationship in relationships
            if {
                relationship["head"],
                relationship["tail"],
            }
            == {left, right}
        ]

        lines.append(
            f"STEP {step_index + 1}: {left} -> {right}"
        )

        for relationship in step_relationships:
            lines.extend(
                [
                    (
                        f"{relationship['head']} "
                        f"--{relationship['relation']}--> "
                        f"{relationship['tail']}"
                    ),
                    (
                        f"Source: "
                        f"{relationship.get('source_file', '')}"
                    ),
                    (
                        f"Evidence: "
                        f"{relationship.get('source_sentence', '')}"
                    ),
                ]
            )

        lines.append("")

    return "\n".join(lines)


def judge_path(client, path_text):
    prompt = f"""
Evaluate this knowledge-graph path using only the evidence below.

Risk direction rules:
- A --supplier_of--> B means disruption risk can move A to B.
- A --depends_on--> B means disruption risk can move B to A.
- A --customer_of--> B means disruption risk can move B to A.
- competitor_of does not establish a supply-chain cascade.

Score each item from 0 to 2:
- path_continuity
- evidence_coverage
- direction_consistency
- cascade_relevance
- path_quality

Return only JSON in this format:
{{
  "path_continuity": 0,
  "evidence_coverage": 0,
  "direction_consistency": 0,
  "cascade_relevance": 0,
  "path_quality": 0,
  "unsupported_steps": [],
  "direction_errors": [],
  "explanation": ""
}}

Path and evidence:
{path_text}
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
            "path_continuity",
            "evidence_coverage",
            "direction_consistency",
            "cascade_relevance",
            "path_quality",
        ]
    )

    return result


def save_results(rows):
    output_path = RESULTS_DIR / "kg_path_evaluation_results.csv"

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "path",
                "hop_count",
                "path_continuity",
                "evidence_coverage",
                "direction_consistency",
                "cascade_relevance",
                "path_quality",
                "total_score",
                "unsupported_steps",
                "direction_errors",
                "explanation",
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

    for path in PATHS_TO_EVALUATE:
        print(f"\nEvaluating: {' -> '.join(path)}")

        path_result = find_path(
            module,
            graph,
            path,
        )
        path_text = format_path(path_result)
        evaluation = judge_path(
            client,
            path_text,
        )

        print(
            f"Judge score: "
            f"{evaluation['total_score']}/10"
        )
        print(evaluation["explanation"])

        rows.append(
            {
                "path": " -> ".join(path),
                "hop_count": len(path) - 1,
                "path_continuity": evaluation[
                    "path_continuity"
                ],
                "evidence_coverage": evaluation[
                    "evidence_coverage"
                ],
                "direction_consistency": evaluation[
                    "direction_consistency"
                ],
                "cascade_relevance": evaluation[
                    "cascade_relevance"
                ],
                "path_quality": evaluation["path_quality"],
                "total_score": evaluation["total_score"],
                "unsupported_steps": "; ".join(
                    evaluation["unsupported_steps"]
                ),
                "direction_errors": "; ".join(
                    evaluation["direction_errors"]
                ),
                "explanation": evaluation["explanation"],
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
