import json, sys

data = json.load(open("output/qa_test_results.json", encoding="utf-8"))
ids = {"feedback-01", "feedback-02", "feedback-04"}
for item in data:
    if item["id"] not in ids:
        continue
    print(f"\n{'='*60}")
    print(f"ID: {item['id']}")
    print(f"REASONS: {item['reasons']}")
    cypher = str(item.get("cypher") or "")[:300]
    print(f"CYPHER: {cypher}")
    answer = str(item.get("answer") or "")[:500]
    print(f"ANSWER: {answer}")
