import json

from modules.evaluation_pipeline import run_evaluation


BENCHMARKS = {
    "project": {
        "question": "Tell me about a project you worked on.",
        "answers": {
            "weak": (
                "I worked on a hiring platform with my team. We built some features "
                "and it worked better after launch."
            ),
            "average": (
                "I built an interview dashboard in FastAPI with PostgreSQL. "
                "I added caching for repeated candidate lookups and cut report "
                "generation time by about 25%."
            ),
            "strong": (
                "I designed, owned, and was responsible for ARIES, an AI interview "
                "evaluation platform. I owned the architecture and was responsible for "
                "the optimization work. I implemented speaker-aware emotion recognition, "
                "integrated FastAPI services, "
                "containerized the stack with Docker, added Redis caching, and "
                "reduced feature extraction time from over 5 hours to 22 minutes "
                "using caching and parallel processing."
            ),
        },
    },
    "behavioral": {
        "question": "Tell me about a time you resolved a conflict on a team.",
        "answers": {
            "weak": (
                "There was a disagreement on the team, so we talked and eventually "
                "finished the work."
            ),
            "average": (
                "Two engineers disagreed on how to expose a new API. I brought them "
                "together, listed the trade-offs, and proposed a smaller FastAPI "
                "contract so we could unblock the sprint. That reduced rework by 15% "
                "and kept the release on schedule."
            ),
            "strong": (
                "On a payments project, two senior engineers were blocked on a database "
                "and API design decision. I led a technical review, created a scoring "
                "rubric for FastAPI, PostgreSQL, latency, migration risk, and "
                "maintainability, and drove a decision the same day. We shipped two "
                "weeks earlier, reduced query latency by 35%, and saved 2 days of "
                "follow-up rework."
            ),
        },
    },
    "technical": {
        "question": "Explain how you optimized a slow API in production.",
        "answers": {
            "weak": (
                "I worked on a slow API once. I looked at the code and changed a query."
            ),
            "average": (
                "I owned a slow FastAPI endpoint, profiled it, added PostgreSQL indexes, "
                "and cached repeated reads. That cut latency by about 30%."
            ),
            "strong": (
                "I designed and owned a slow FastAPI service, and I was responsible for "
                "the production optimization plan. I started with tracing and p95 "
                "metrics, fixed N+1 PostgreSQL queries, added Redis caching for hot "
                "reads, and moved feature extraction into async workers. In production, "
                "that cut p95 latency from 1.8 seconds to 220 ms and reduced timeouts "
                "by 45%."
            ),
        },
    },
    "leadership": {
        "question": "Describe how you managed people, motivated a team, and drove alignment during a difficult project.",
        "answers": {
            "weak": (
                "We were on a difficult project, and I helped the team stay calm so we could finish."
            ),
            "average": (
                "I managed a 4-person team building an internal analytics service. I split "
                "the work into milestones, ran weekly risk reviews, and helped the team "
                "recover from a missed deadline so we still shipped on time and reduced "
                "escalations by 20%."
            ),
            "strong": (
                "I managed and led a 6-engineer cross-functional team migrating a monolith into "
                "Dockerized FastAPI services on AWS. I owned the rollout plan and was "
                "responsible for the architecture decisions around PostgreSQL and Redis. "
                "We shipped in three phases with zero downtime, cut incident volume by "
                "40%, and saved 6 hours of rollback work during each release."
            ),
        },
    },
}

ORDER = ["weak", "average", "strong"]
METRICS = ["final_score", "technical_depth_score", "ownership_score", "impact_score"]


def evaluate(question: str, answer: str, category: str) -> dict:
    result = run_evaluation(question, answer, audio_path=None, duration=None, category=category)
    return {
        "question_type": result.get("question_type"),
        "grade": result.get("grade"),
        "final_score": result.get("scores", {}).get("final", 0.0),
        "technical_depth_score": result.get("technical_depth_score", 0.0),
        "ownership_score": result.get("ownership_score", 0.0),
        "impact_score": result.get("impact_score", 0.0),
    }


def monotonic(values: list[float]) -> bool:
    return values[0] < values[1] < values[2]


def main() -> None:
    results = {}
    checks = {}

    for benchmark_type, payload in BENCHMARKS.items():
        question = payload["question"]
        results[benchmark_type] = {}
        checks[benchmark_type] = {}
        for level in ORDER:
            answer = payload["answers"][level]
            results[benchmark_type][level] = evaluate(question, answer, benchmark_type.title())

        for metric in METRICS:
            values = [results[benchmark_type][level][metric] for level in ORDER]
            checks[benchmark_type][metric] = {
                "values": values,
                "monotonic": monotonic(values),
            }

    print(
        json.dumps(
            {
                "results": results,
                "checks": checks,
                "all_checks_passed": all(
                    entry["monotonic"]
                    for benchmark_checks in checks.values()
                    for entry in benchmark_checks.values()
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
