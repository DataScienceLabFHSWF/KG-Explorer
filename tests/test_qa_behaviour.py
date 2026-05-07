"""
Pytest wrapper around the behavioural QA test runner.

Runs the subset of test-question categories that are expected to pass
reliably, and fails if the pass rate drops below the threshold. Sentinel /
expected-to-fail items are excluded from the regression gate.

Usage:
    pytest tests/test_qa_behaviour.py -v
    pytest tests/test_qa_behaviour.py -v --tb=short -k "structural"
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

DATASET_PATH = Path(__file__).with_name("qa_test_questions.json")

# Minimum acceptable per-category pass rate (0–1).
# Adjust when genuinely improving or adding harder questions.
PASS_RATE_THRESHOLDS: dict[str, float] = {
    "definition": 0.60,
    "structural": 0.60,
    "mechanism": 0.30,   # harder category, currently 1/3
    "comparative": 1.00,
    "trend": 1.00,
    "path": 1.00,
    # missing_expected: gated separately (sentinel must fire, not fail)
}

# --- helpers ----------------------------------------------------------------

def _load_dataset() -> list[dict]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))["questions"]


def _agent():
    """Lazy-loaded agent singleton (avoids re-initialising Neo4j per test)."""
    from analysis.llm_graph_qa import FusionCypherAgent
    if not hasattr(_agent, "_instance"):
        _agent._instance = FusionCypherAgent()
    return _agent._instance


def _check(result: dict, expect: dict) -> tuple[bool, list[str]]:
    """Shared check logic — mirrors tests/run_qa_tests._check_expectations."""
    from tests.run_qa_tests import _check_expectations
    return _check_expectations(result, expect)


# --- parametrised per-question tests ----------------------------------------

def pytest_collect_file(parent, file_path):
    """Not used; collection happens via generate_tests below."""


def _questions_for_pytest() -> list[pytest.param]:
    items = _load_dataset()
    params = []
    for it in items:
        # Skip items explicitly expected to produce a sentinel —
        # they are tested separately below.
        cat = it.get("category", "")
        marks = []
        if it.get("expected_to_fail"):
            marks.append(pytest.mark.xfail(reason=it["expected_to_fail"], strict=False))
        params.append(
            pytest.param(it, id=it["id"], marks=marks)
        )
    return params


@pytest.mark.parametrize("item", _questions_for_pytest())
def test_question(item):
    """Each question in the dataset is one test. Xfail items are allowed to
    fail without breaking the suite."""
    cat = item.get("category", "")
    if cat == "missing_expected":
        # These are tested in test_sentinel_fires below.
        pytest.skip("sentinel-category tested separately")

    agent = _agent()
    t0 = time.time()
    result = agent.ask(item["question"])
    elapsed = time.time() - t0

    passed, reasons = _check(result, item.get("expect", {}))
    assert passed, (
        f"[{item['id']}] {item['question']}\n"
        f"  linked={[n for n, _ in (result.get('linked_entities') or [])]}\n"
        f"  rows={len(result.get('context') or [])}  "
        f"abstracts={len(result.get('abstracts') or [])}  "
        f"missing={result.get('missing_from_graph')}  "
        f"coverage={result.get('coverage_score')}  "
        f"elapsed={elapsed:.1f}s\n"
        + "\n".join(f"  - {r}" for r in reasons)
    )


@pytest.mark.parametrize("item", [
    pytest.param(it, id=it["id"])
    for it in _load_dataset()
    if it.get("category") == "missing_expected"
])
def test_sentinel_fires(item):
    """For questions expected to be missing from the graph, the sentinel or
    low coverage must trigger."""
    agent = _agent()
    result = agent.ask(item["question"])
    sentinel_ok = (
        result.get("missing_from_graph")
        or (result.get("coverage_score") is not None and result["coverage_score"] < 0.4)
    )
    assert sentinel_ok, (
        f"[{item['id']}] Expected sentinel/low-coverage for: {item['question']}\n"
        f"  missing_from_graph={result.get('missing_from_graph')}  "
        f"coverage={result.get('coverage_score')}"
    )


# --- category-level pass-rate tests (regression gates) ----------------------

@pytest.fixture(scope="session")
def all_results():
    """Run the full dataset once per session and cache results."""
    from tests.run_qa_tests import _run_one
    agent = _agent()
    dataset = _load_dataset()
    return [_run_one(agent, it) for it in dataset]


@pytest.mark.parametrize("cat,threshold", PASS_RATE_THRESHOLDS.items())
def test_category_pass_rate(cat, threshold, all_results):
    """Category-level regression gate: pass rate must not drop below threshold."""
    items = [r for r in all_results if r["category"] == cat]
    if not items:
        pytest.skip(f"No items in category '{cat}'")
    rate = sum(1 for r in items if r["passed"]) / len(items)
    assert rate >= threshold, (
        f"Category '{cat}': pass rate {rate:.0%} < threshold {threshold:.0%}\n"
        + "\n".join(
            f"  FAIL [{r['id']}]: {r['reasons']}"
            for r in items if not r["passed"]
        )
    )
