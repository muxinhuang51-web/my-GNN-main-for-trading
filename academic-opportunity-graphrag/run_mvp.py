from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urlparse

import requests
from bs4 import BeautifulSoup


OPENING_KEYWORDS = [
    "prospective student",
    "prospective students",
    "open positions",
    "opening",
    "openings",
    "research assistant",
    "ra",
    "visiting student",
    "visiting students",
    "intern",
    "internship",
    "phd",
    "ph.d",
    "join us",
    "recruiting",
    "招生",
    "招收",
    "实习生",
    "科研助理",
    "访问学生",
    "博士生",
    "硕士生",
    "本科生",
]

ROLE_KEYWORDS = {
    "RA": ["research assistant", "科研助理", "ra"],
    "Visiting Student": ["visiting student", "visiting students", "访问学生"],
    "Intern": ["intern", "internship", "实习生", "科研实习"],
    "PhD": ["phd", "ph.d", "博士生", "博士"],
    "Master": ["master", "mphil", "硕士生", "硕士"],
}

DEFAULT_FIT_KEYWORDS = [
    "graph neural network",
    "graph neural networks",
    "gnn",
    "graph learning",
    "geometric deep learning",
    "financial graph",
    "finance",
    "fintech",
    "quantitative finance",
    "quantitative trading",
    "algorithmic trading",
    "reinforcement learning",
    "large language model",
    "llm",
    "machine learning",
    "data mining",
    "portfolio",
    "market microstructure",
]


@dataclass
class Person:
    name: str
    chinese_name: str = ""
    institution: str = ""
    homepage: str = ""
    areas: list[str] = field(default_factory=list)
    notes: str = ""
    source_seed: str = ""
    openalex_id: str = ""
    email: str = ""
    page_title: str = ""
    text: str = ""
    opening_hits: list[str] = field(default_factory=list)
    fit_hits: list[str] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    score: float = 0.0
    fetch_status: str = ""

    @property
    def node_id(self) -> str:
        key = self.openalex_id or self.homepage or self.name
        return stable_id(key)


def stable_id(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_url(url: str, timeout: int = 15) -> tuple[str, str]:
    headers = {
        "User-Agent": "AcademicOpportunityGraphRAG/0.1 (+https://github.com/muxinhuang51-web)"
    }
    try:
        resp = requests.get(url, timeout=timeout, headers=headers)
        resp.raise_for_status()
        return resp.text, f"ok:{resp.status_code}"
    except Exception as exc:  # noqa: BLE001 - keep MVP resilient
        return "", f"error:{type(exc).__name__}:{exc}"


def page_to_text(raw_html: str) -> tuple[str, str]:
    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    title = normalize_space(soup.title.get_text(" ")) if soup.title else ""
    body = normalize_space(soup.get_text(" "))
    return html.unescape(title), html.unescape(body)


def extract_email(text: str) -> str:
    match = re.search(r"[\w.\-+]+@[\w.\-]+\.\w+", text)
    return match.group(0) if match else ""


def hits_for_keywords(text: str, keywords: list[str]) -> list[str]:
    lower = text.lower()
    hits = []
    for kw in keywords:
        if kw.lower() in lower:
            hits.append(kw)
    return sorted(set(hits), key=str.lower)


def detect_roles(text: str) -> list[str]:
    lower = text.lower()
    roles = []
    for role, keywords in ROLE_KEYWORDS.items():
        if any(kw.lower() in lower for kw in keywords):
            roles.append(role)
    return roles


def openalex_search_author(name: str) -> dict[str, Any] | None:
    url = f"https://api.openalex.org/authors?search={quote_plus(name)}&per-page=1"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 429:
            return None
        data = resp.json()
        results = data.get("results", [])
        return results[0] if results else None
    except Exception:
        return None


def openalex_works_for_author(author_id: str, limit: int = 8) -> list[dict[str, Any]]:
    clean_id = author_id.rsplit("/", 1)[-1]
    url = (
        "https://api.openalex.org/works"
        f"?filter=authorships.author.id:{clean_id}"
        f"&sort=publication_year:desc&per-page={limit}"
    )
    try:
        data = requests.get(url, timeout=20).json()
        return data.get("results", [])
    except Exception:
        return []


def coauthors_from_works(works: list[dict[str, Any]], seed_name: str, max_coauthors: int) -> list[dict[str, str]]:
    seen: dict[str, dict[str, str]] = {}
    for work in works:
        for authorship in work.get("authorships", []):
            author = authorship.get("author", {})
            name = author.get("display_name", "")
            if not name or name.lower() == seed_name.lower():
                continue
            institutions = authorship.get("institutions", [])
            institution = institutions[0].get("display_name", "") if institutions else ""
            aid = author.get("id", "")
            if aid and aid not in seen:
                seen[aid] = {
                    "name": name,
                    "institution": institution,
                    "openalex_id": aid,
                    "work_title": work.get("title", ""),
                }
            if len(seen) >= max_coauthors:
                return list(seen.values())
    return list(seen.values())


def enrich_seed(seed: dict[str, Any], profile_keywords: list[str]) -> tuple[Person, list[dict[str, str]], list[dict[str, Any]]]:
    person = Person(
        name=seed.get("name", ""),
        chinese_name=seed.get("chinese_name", ""),
        institution=seed.get("institution", ""),
        homepage=seed.get("homepage", ""),
        areas=seed.get("areas", []),
        notes=seed.get("notes", ""),
        source_seed=seed.get("name", ""),
    )

    raw_html = ""
    if person.homepage:
        raw_html, person.fetch_status = fetch_url(person.homepage)
        if raw_html:
            person.page_title, person.text = page_to_text(raw_html)
            person.email = extract_email(person.text)
    else:
        person.fetch_status = "no-homepage"

    combined = " ".join([person.text, " ".join(person.areas), person.notes])
    person.opening_hits = hits_for_keywords(combined, OPENING_KEYWORDS)
    person.fit_hits = hits_for_keywords(combined, profile_keywords)
    person.roles = detect_roles(combined)

    author = openalex_search_author(person.name)
    works: list[dict[str, Any]] = []
    if author:
        person.openalex_id = author.get("id", "")
        if not person.institution:
            inst = author.get("last_known_institution") or {}
            person.institution = inst.get("display_name", "")
        works = openalex_works_for_author(person.openalex_id)
    manual_coauthors = seed.get("coauthors", [])
    api_coauthors = coauthors_from_works(works, person.name, max_coauthors=8)
    coauthors = []
    seen = set()
    for co in manual_coauthors + api_coauthors:
        key = co.get("openalex_id") or co.get("homepage") or co.get("name")
        if key and key not in seen:
            seen.add(key)
            coauthors.append(co)
    return person, coauthors, works


def score_person(person: Person, is_seed: bool) -> float:
    score = 0.0
    score += 10.0 if is_seed else 0.0
    score += min(len(person.opening_hits), 8) * 8.0
    score += min(len(person.fit_hits), 10) * 4.0
    score += 4.0 if person.email else 0.0
    score += 3.0 if person.homepage else 0.0
    score += 3.0 if any(role in person.roles for role in ["RA", "Visiting Student", "Intern"]) else 0.0
    if "phd" in [hit.lower() for hit in person.opening_hits]:
        score += 1.0
    return round(score, 2)


def make_email_draft(person: Person, profile: dict[str, Any]) -> str:
    interests = "、".join(["图神经网络", "量化金融", "人工智能在金融市场建模中的应用"])
    tasks = "、".join(profile.get("can_help_with", [])[:6])
    role = "科研实习、本科访问学生或科研助理"
    area_line = "、".join(person.fit_hits[:5]) if person.fit_hits else "人工智能与金融科技交叉研究"

    professor_name = person.chinese_name or (person.name.split()[-1] if re.search(r"[A-Za-z]", person.name) else person.name)
    greeting = f"尊敬的{professor_name}老师您好："

    return textwrap.dedent(
        f"""
        {greeting}

        我是{profile['school']}{profile['major']}专业本科生{profile['name']}，目前主要关注{interests}。过去半年中，我持续进行科研训练，重点围绕金融图神经网络建模与量化交易展开探索，尝试将股票视为节点，基于资产间关系构建图结构，并利用 GNN 对股票关联、市场状态和交易信号进行建模。相关探索性工作整理在 GitHub：

        {profile['project']}

        我了解到您/课题组的研究方向与 {area_line} 等主题相关，这与我的专业背景和当前科研兴趣高度契合。我的本科专业训练覆盖金融工程、统计建模和计量方法，同时具备较扎实的 Python 编程基础，也在持续学习 PyTorch、图神经网络、强化学习和大模型在金融场景中的应用。

        我非常希望能以{role}的形式参与课题组相关工作。如果有机会加入，我愿意协助完成{tasks}等任务。虽然我目前仍处在科研训练早期，但我执行力较强，也愿意按照团队要求长期、稳定地投入学习和研究。

        随信附上我的简历。若老师方便，我也非常愿意通过线上会议进一步介绍我的背景、已有探索项目和暑期/秋季可投入时间。感谢老师在百忙之中阅读我的邮件，期待有机会向您学习。

        祝好！

        {profile['name']}
        {profile['school']} {profile['major']}
        邮箱：{profile['email']}
        GitHub: {profile['github']}
        个人主页：{profile['homepage']}
        """
    ).strip()


def graph_node(person: Person, kind: str) -> dict[str, Any]:
    return {
        "id": person.node_id,
        "kind": kind,
        "name": person.name,
        "chinese_name": person.chinese_name,
        "institution": person.institution,
        "homepage": person.homepage,
        "email": person.email,
        "areas": person.areas,
        "opening_hits": person.opening_hits,
        "fit_hits": person.fit_hits,
        "roles": person.roles,
        "score": person.score,
        "openalex_id": person.openalex_id,
    }


def write_outputs(out_dir: Path, people: list[Person], graph: dict[str, Any], profile: dict[str, Any], run_log: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    drafts_dir = out_dir / "email_drafts"
    drafts_dir.mkdir(exist_ok=True)

    ranked = sorted(people, key=lambda p: p.score, reverse=True)

    csv_path = out_dir / "opportunities.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "score",
                "name",
                "institution",
                "email",
                "homepage",
                "roles",
                "opening_hits",
                "fit_hits",
                "source_seed",
                "notes",
            ],
        )
        writer.writeheader()
        for p in ranked:
            writer.writerow(
                {
                    "score": p.score,
                    "name": p.name,
                    "institution": p.institution,
                    "email": p.email,
                    "homepage": p.homepage,
                    "roles": "; ".join(p.roles),
                    "opening_hits": "; ".join(p.opening_hits),
                    "fit_hits": "; ".join(p.fit_hits),
                    "source_seed": p.source_seed,
                    "notes": p.notes,
                }
            )

    md_lines = [
        "# Ranked Academic Opportunities",
        "",
        f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| Rank | Score | Name | Institution | Roles | Fit Hits | Opening Hits | Homepage |",
        "|---:|---:|---|---|---|---|---|---|",
    ]
    for idx, p in enumerate(ranked, start=1):
        md_lines.append(
            "| {rank} | {score} | {name} | {inst} | {roles} | {fit} | {opening} | {home} |".format(
                rank=idx,
                score=p.score,
                name=p.name,
                inst=p.institution,
                roles=", ".join(p.roles[:4]),
                fit=", ".join(p.fit_hits[:6]),
                opening=", ".join(p.opening_hits[:6]),
                home=p.homepage,
            )
        )
    (out_dir / "opportunities.md").write_text("\n".join(md_lines), encoding="utf-8")

    save_json(out_dir / "graph.json", graph)
    save_json(out_dir / "run_log.json", run_log)

    for p in ranked[:10]:
        filename = re.sub(r"[^A-Za-z0-9_\-]+", "_", p.name).strip("_") or p.node_id
        (drafts_dir / f"{filename}.md").write_text(make_email_draft(p, profile), encoding="utf-8")


def run(seeds_path: Path, profile_path: Path, out_dir: Path, max_second_hop: int) -> None:
    seeds = load_json(seeds_path)
    profile = load_json(profile_path)
    profile_keywords = sorted(set(DEFAULT_FIT_KEYWORDS + profile.get("interests", [])), key=str.lower)

    people_by_key: dict[str, Person] = {}
    edges: list[dict[str, Any]] = []
    run_log: list[dict[str, Any]] = []

    for seed in seeds:
        person, coauthors, works = enrich_seed(seed, profile_keywords)
        person.score = score_person(person, is_seed=True)
        people_by_key[person.node_id] = person
        run_log.append(
            {
                "name": person.name,
                "homepage": person.homepage,
                "fetch_status": person.fetch_status,
                "openalex_id": person.openalex_id,
                "works_found": len(works),
                "coauthors_found": len(coauthors),
            }
        )

        for co in coauthors[:max_second_hop]:
            co_person = Person(
                name=co["name"],
                chinese_name=co.get("chinese_name", ""),
                institution=co.get("institution", ""),
                source_seed=person.name,
                openalex_id=co.get("openalex_id", ""),
                homepage=co.get("homepage", ""),
                notes=f"Coauthor via: {co.get('work_title', '')}",
            )
            if co_person.homepage:
                raw_html, co_person.fetch_status = fetch_url(co_person.homepage)
                if raw_html:
                    co_person.page_title, co_person.text = page_to_text(raw_html)
                    co_person.email = extract_email(co_person.text)
            combined = " ".join([co_person.name, co_person.institution, co_person.notes])
            if co_person.text:
                combined = f"{combined} {co_person.text}"
            co_person.fit_hits = hits_for_keywords(combined, profile_keywords)
            co_person.opening_hits = hits_for_keywords(combined, OPENING_KEYWORDS)
            co_person.roles = detect_roles(combined)
            co_person.score = score_person(co_person, is_seed=False)
            people_by_key.setdefault(co_person.node_id, co_person)
            edges.append(
                {
                    "source": person.node_id,
                    "target": co_person.node_id,
                    "kind": "coauthor",
                    "source_seed": person.name,
                    "evidence": co.get("work_title", ""),
                }
            )

    nodes = []
    seed_names = {seed.get("name", "") for seed in seeds}
    for p in people_by_key.values():
        nodes.append(graph_node(p, "seed_professor" if p.name in seed_names else "coauthor"))

    graph = {
        "nodes": nodes,
        "edges": edges,
        "meta": {
            "seed_count": len(seeds),
            "node_count": len(nodes),
            "edge_count": len(edges),
            "profile": profile.get("name", ""),
        },
    }
    write_outputs(out_dir, list(people_by_key.values()), graph, profile, run_log)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Academic Opportunity GraphRAG MVP")
    parser.add_argument("--seeds", type=Path, required=True, help="Path to seed professor JSON")
    parser.add_argument("--profile", type=Path, required=True, help="Path to applicant profile JSON")
    parser.add_argument("--out", type=Path, default=Path("outputs/demo"), help="Output directory")
    parser.add_argument("--max-second-hop", type=int, default=8, help="Max coauthors per seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.seeds, args.profile, args.out, args.max_second_hop)
