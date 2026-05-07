from analysis.llm_graph_qa import FusionCypherAgent, _detect_category_hint
a = FusionCypherAgent()
q = "Which fusion devices appear most often in the literature?"
cat = _detect_category_hint(q)
print("Category hint:", cat)
cypher, rows = a._run_ranking_cypher(cat)
print(f"Rows returned: {len(rows)}")
for r in rows[:8]:
    print(f"  {r['papers']:5d}  {r['entity']}")
