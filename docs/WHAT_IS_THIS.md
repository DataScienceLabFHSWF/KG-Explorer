# What Is Felix Working On? 🌟

## The Short Version

I'm building a **map of everything scientists know about nuclear fusion** — the same kind of energy that powers the sun — and using maths to find **what they haven't figured out yet**.

## A Bit More Detail

Scientists have written **over 8,000 papers** about nuclear fusion (the technology that could give us unlimited, clean energy). A team in Italy read all of those papers using AI and pulled out the **50,000 important concepts** mentioned in them — things like "tokamak" (a donut-shaped reactor), "plasma" (super-hot gas), "deuterium" (a type of hydrogen fuel), etc.

I took all those concepts and built a giant **knowledge graph** — think of it as a web where every concept is a dot, and whenever two concepts appear together in a paper, they get connected by a line. The more papers mention them together, the thicker the line.

Then I used a bunch of mathematical tools to analyse this web:

- **Which concepts are the most important?** -> "tokamak" is by far #1
- **Are there clusters of related topics?** -> Yes, 1,402 research communities
- **Are there gaps in the research?** -> Yes! We found **45 holes** where scientists study pairs of concepts but never all three together
- **What connections are missing?** -> We predicted 200 links that should exist but don't
- **Which concepts connect different research groups?** -> "tokamak", "electron", "magnetic field" bridge the most communities
- **Does the vocabulary follow natural laws?** -> Yes! Entity frequencies follow Zipf's law (power-law distribution)
- **Can we build a formal ontology?** -> Yes! We auto-generated an OWL 2 ontology with 2,487 triples from the graph structure
- **Can we ask questions?** -> Yes! A graph query interface lets you explore paths, trends, and gaps interactively

## Why Does This Matter?

If we can find the **blind spots** in fusion research, we can suggest where scientists should look next. It's like having a GPS for scientific discovery -- instead of wandering randomly, you can see where the unexplored territory is.

We also built a **data acquisition pipeline** that can download open-access papers (including arXiv preprints as fallback for paywalled ones) and feed them into a knowledge graph builder for even deeper analysis.

## What Does It Look Like?

The project generates **32 charts**, an interactive web visualisation you can click around in, an OWL ontology file, and a "gap report" listing specific research ideas nobody has tried yet. There's also a command-line query tool where you can ask things like `"tokamak -> stellarator"` to find the shortest path between concepts.

## The Tech Stack (If You're Curious)

Python, Neo4j (a graph database), and a lot of maths (topology, spectral theory, information theory, formal concept analysis, Zipf's law). Plus a data acquisition pipeline using OpenAlex for open-access paper metadata and arXiv for preprints. All running in Docker on this machine.

---

*Made with ☕ and too many late nights at the GAIA Lab, FH Südwestfalen*
