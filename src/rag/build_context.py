import os

from groq import Groq

from graph.build_graph import (
    MERGED_TRIPLES_PATH,
    build_graph,
    load_triples,
)
from graph.retrieve_paths import retrieve_evidence
from graph.visualize_paths import visualize_paths


MAX_CASCADE_HOPS = 5
MAX_EVIDENCE_PATHS = 25
MAX_SELECTED_PATHS = 3


def build_rag_context(evidence_paths: list[dict]) -> str:
    """Convert retrieved graph evidence into an LLM-readable context."""
    if not evidence_paths:
        return "No relevant supply-chain relationships were found."

    context_sections = []

    for path_number, result in enumerate(evidence_paths, start=1):
        path = result["path"]
        path_text = " -> ".join(path)
        hop_count = len(path) - 1

        lines = [
            f"PATH {path_number}",
            f"Graph path: {path_text}",
            f"Hop count: {hop_count}",
            "",
        ]

        for relationship_number, relationship in enumerate(
            result["relationships"],
            start=1,
        ):
            confidence = relationship.get("confidence")
            confidence_text = (
                f"{confidence:.2f}"
                if isinstance(confidence, (int, float))
                else "not provided"
            )

            source_sentence = relationship.get("source_sentence") or ""

            if len(source_sentence) > 500:
                source_sentence = source_sentence[:500] + "..."

            lines.extend(
                [
                    f"Relationship {relationship_number}:",
                    (
                        f"{relationship['head']} "
                        f"--{relationship['relation']}--> "
                        f"{relationship['tail']}"
                    ),
                    f"Confidence: {confidence_text}",
                    f"Source: {relationship['source_file']}",
                    f"Evidence: {source_sentence}",
                    "",
                ]
            )

        context_sections.append("\n".join(lines))

    return "\n\n".join(context_sections)


def find_company_in_question(
    question: str,
    graph,
) -> str:
    """Find the company identified as the source of the disruption."""
    question_lower = question.lower()

    companies = [
        company
        for company in graph.nodes
        if isinstance(company, str)
    ]

    disruption_phrases = [
        "disruption in ",
        "disruption at ",
        "disruption of ",
        "disruption to ",
    ]

    # First look for the company named after a disruption phrase.
    for phrase in disruption_phrases:
        phrase_position = question_lower.find(phrase)

        if phrase_position == -1:
            continue

        text_after_phrase = question_lower[
            phrase_position + len(phrase):
        ]

        matches = [
            company
            for company in companies
            if text_after_phrase.startswith(company.lower())
        ]

        if matches:
            return max(matches, key=len)

    # Otherwise, use the first company mentioned in the question.
    matches = []

    for company in companies:
        position = question_lower.find(company.lower())

        if position != -1:
            matches.append((position, company))

    if not matches:
        raise ValueError(
            "Could not identify a company in the question. "
            "Use a company name exactly as it appears in the graph."
        )

    matches.sort(key=lambda item: item[0])
    return matches[0][1]


def find_companies_in_question(
    question: str,
    graph,
) -> list[str]:
    """Find company names mentioned in the user's question."""
    question_lower = question.lower()

    matches = [
        company
        for company in graph.nodes
        if isinstance(company, str)
        and company.lower() in question_lower
    ]

    return sorted(
        matches,
        key=lambda company: question_lower.find(company.lower()),
    )


def select_deepest_paths(
    evidence_paths: list[dict],
    limit: int = MAX_SELECTED_PATHS,
) -> list[dict]:
    """Keep the deepest retrieved paths for the answer and visualization."""
    sorted_paths = sorted(
        evidence_paths,
        key=lambda result: len(result["path"]),
        reverse=True,
    )

    return sorted_paths[:limit]


def build_rag_prompt(
    question: str,
    context: str,
) -> str:
    """Build a grounded prompt using the retrieved graph evidence."""
    return f"""
You are analyzing supply-chain risk using evidence from SEC filings.

Answer the user's question using only the retrieved evidence below.

Rules:
- Do not invent companies or relationships.
- Treat extracted relationships as potentially noisy.
- Clearly separate direct evidence from inference.
- If the evidence is insufficient, say so.
- Cite the source filename for each important claim.
- A direct relationship has exactly 1 hop.
- Any path with 2 or more hops is a multi-hop cascade.
- Use the term multi-hop cascade only when the risk passes through at least one intermediate company.
- Do not downplay a supported direct dependency merely because it is not multi-hop.
- When supplier_of and customer_of describe the same company pair using the same source evidence, treat them as one relationship rather than two separate findings.
- Do not interpret extraction confidence scores as calibrated probabilities.
- Treat longer cascade paths as progressively less certain.
- Do not claim a multi-hop cascade unless every edge has supporting evidence.
- Use the provided hop count for each path.
- Do not recalculate or change the provided hop count.
- A hop is one graph relationship between two companies.
- Label a path as direct only when its provided hop count is exactly 1.
- Label every path with a hop count greater than 1 as a multi-hop cascade.

Answer format:
- Begin with a brief 1-2 sentence summary.
- Mention no more than 3 distinct paths.
- Combine paths that share most of the same companies.
- Do not repeat the same path or relationship.
- Distinguish direct relationships from multi-hop relationships.
- Do not list every confidence score.
- Keep the full answer under 250 words.
- Use clear, non-technical language suitable for a classroom demonstration.

User question:
{question}

Retrieved evidence:
{context}

Answer:
""".strip()


def generate_answer(prompt: str) -> str:
    """Send the grounded prompt to Groq and return the answer."""
    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set in the environment."
        )

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.1,
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    triples = load_triples(MERGED_TRIPLES_PATH)
    graph = build_graph(triples)

    question = input(
        "Enter a supply-chain risk question: "
    ).strip()

    if not question:
        raise ValueError("A question is required.")

    companies = find_companies_in_question(
        question=question,
        graph=graph,
    )

    start_company = find_company_in_question(
        question=question,
        graph=graph,
    )

    target_company = next(
        (
            company
            for company in companies
            if company != start_company
        ),
        None,
    )

    evidence_paths = retrieve_evidence(
        graph,
        start_node=start_company,
        max_hops=MAX_CASCADE_HOPS,
        max_paths=MAX_EVIDENCE_PATHS,
    )

    if target_company:
        evidence_paths = [
            result
            for result in evidence_paths
            if result["path"][-1] == target_company
        ]

    if not evidence_paths:
        print("\nNo supported graph path was found for that question.")
        raise SystemExit

    max_depth_found = max(
        len(result["path"]) - 1
        for result in evidence_paths
    )

    selected_paths = select_deepest_paths(
        evidence_paths,
        limit=MAX_SELECTED_PATHS,
    )

    print(f"\nMaximum cascade depth found: {max_depth_found} hops")
    print(
        f"Using the {len(selected_paths)} deepest paths "
        "for the answer and visualization."
    )

    context = build_rag_context(selected_paths)

    prompt = build_rag_prompt(
        question=question,
        context=context,
    )

    answer = generate_answer(prompt)

    print(f"\nDetected disrupted company: {start_company}")

    if target_company:
        print(f"Detected target company: {target_company}")

    print("\nANSWER")
    print("=" * 80)
    print(answer)

    visualize_paths(
        evidence_paths=selected_paths,
        disrupted_company=start_company,
        target_company=target_company,
    )
