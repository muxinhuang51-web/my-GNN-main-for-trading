# Academic Opportunity GraphRAG

A minimal graph-based assistant for discovering RA, visiting student, and PhD opportunities through faculty collaboration networks.

The MVP starts from a small list of seed professors, fetches public webpages and OpenAlex author data, expands through coauthors, detects opportunity signals, ranks candidates by research fit, and generates Markdown reports plus cold-email drafts.

## Why This Exists

Finding research opportunities is not just keyword search. A professor who is hiring often has collaborators in the same area, and those collaborators may also have openings. This project turns that intuition into a graph:

- Nodes: professors, institutions, webpages, papers, openings
- Edges: seed-to-professor, coauthor links, homepage links
- Signals: RA/visiting/PhD keywords, research-area match, recency hints, direct email availability

## Quick Start

```powershell
cd academic-opportunity-graphrag
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_mvp.py --seeds seeds/example_seeds.json --profile profiles/muxin.json --out outputs/demo
```

If you prefer not to create a virtual environment:

```powershell
python run_mvp.py --seeds seeds/example_seeds.json --profile profiles/muxin.json --out outputs/demo
```

## Add Your Own Seeds

Edit `seeds/example_seeds.json` or create a new seed file:

```json
[
  {
    "name": "Professor Name",
    "chinese_name": "中文名",
    "institution": "University",
    "homepage": "https://example.edu/~prof",
    "areas": ["graph neural networks", "AI for Finance"],
    "notes": "Why this professor is a seed.",
    "coauthors": [
      {
        "name": "Collaborator Name",
        "institution": "Another University",
        "homepage": "https://example.edu/~collaborator"
      }
    ]
  }
]
```

Manual `coauthors` are useful because social platforms, screenshots, and personal webpages often reveal collaborators before public APIs do. The script will fetch each collaborator homepage, detect emails/opening signals, and add a coauthor edge to `graph.json`.

## Outputs

The run creates:

- `opportunities.csv`: ranked opportunities
- `opportunities.md`: human-readable shortlist
- `graph.json`: nodes and edges for later visualization or GNN experiments
- `email_drafts/`: professor-specific cold email drafts
- `run_log.json`: source URLs and fetch status

## MVP Scope

This first version intentionally avoids paid APIs and API keys. It uses:

- Public professor homepages
- OpenAlex author search and coauthor expansion
- Regex-based extraction for emails, opening signals, and research keywords
- Rule-based opportunity ranking

Later versions can add:

- LLM-based extraction and fit summaries
- Semantic Scholar / DBLP adapters
- Graph visualization
- Personalized PageRank over the collaboration graph
- A Streamlit UI
- A GNN/ranking model trained on accepted/rejected outreach outcomes

## Project Framing

One-line pitch:

> A graph-based LLM-ready agent for discovering hidden RA, visiting student, and PhD opportunities from faculty collaboration networks.
