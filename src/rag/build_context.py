from graph.build_graph import (
    MERGED_TRIPLES_PATH,
    build_graph,
    load_triples,
)
from graph.retrieve_paths import retrieve_evidence
import os
from groq import Groq
from graph.visualize_paths import visualize_paths

MAX_CASCADE_HOPS = 5
MAX_EVIDENCE_PATHS = 100

def build_rag_context(evidence_paths: list[dict]) -> str:
    """Convert retrieved graph evidence into an LLM-readable context."""
    if not evidence_paths:
        return "No relevant supply-chain relationships were found."

    context_sections = []

    for path_number, result in enumerate(evidence_paths, start=1):
        path_text = " -> ".join(result["path"])

        lines = [
            f"PATH {path_number}",
            f"Graph path: {path_text}",
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
                    f"Evidence: {relationship['source_sentence']}",
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

    # First look for the company named directly after a disruption phrase.
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
- A one-hop supplier-to-customer relationship is a direct supply-chain impact.
- Use the term multi-hop cascade only when the risk passes through at least one intermediate company.
- Do not downplay a supported direct dependency merely because it is not multi-hop.
- When supplier_of and customer_of describe the same company pair using the same source evidence, treat them as one relationship rather than two separate findings.
- Do not interpret extraction confidence scores as calibrated probabilities.

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

    context = build_rag_context(evidence_paths)

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
        evidence_paths=evidence_paths,
        disrupted_company=start_company,
        target_company=target_company,
    )
