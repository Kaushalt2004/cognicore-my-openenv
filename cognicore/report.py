"""
CogniCore Report Generator — Beautiful HTML reports.

Generates standalone HTML reports with charts, tables, and scores
that can be shared with teams.

Usage::

    from cognicore.report import ReportGenerator

    report = ReportGenerator("SafetyClassification-v1")
    report.add_episode(env, agent)
    report.export("results.html")
"""

from __future__ import annotations

import time
import os
from typing import Any, Dict, List, Optional

import cognicore
from cognicore.agents.base_agent import RandomAgent


class ReportGenerator:
    """Generate beautiful standalone HTML reports."""

    def __init__(self, title: str = "CogniCore Agent Report"):
        self.title = title
        self.sections: List[Dict[str, Any]] = []
        self.metrics: Dict[str, Any] = {}
        self._episodes: List[Dict] = []

    def add_metric(self, name: str, value: Any, unit: str = ""):
        self.metrics[name] = {"value": value, "unit": unit}

    def add_section(self, title: str, content: str):
        self.sections.append({"title": title, "content": content})

    def add_episode(self, env_id: str, difficulty: str = "easy", agent=None):
        """Run an episode and record results."""
        env = cognicore.make(env_id, difficulty=difficulty)
        if agent is None:
            agent = RandomAgent(env.action_space)

        obs = env.reset()
        steps = []
        while True:
            action = agent.act(obs)
            obs, reward, done, _, info = env.step(action)
            er = info.get("eval_result", {})
            if hasattr(agent, 'learn'):
                agent.learn(reward, info)
            steps.append({
                "category": er.get("category", "?"),
                "correct": er.get("correct", False),
                "reward": reward.total,
                "memory_bonus": reward.memory_bonus,
                "streak_penalty": reward.streak_penalty,
            })
            if done:
                break

        stats = env.episode_stats()
        self._episodes.append({
            "env_id": env_id,
            "difficulty": difficulty,
            "accuracy": stats.accuracy,
            "score": env.get_score(),
            "steps": steps,
            "correct": stats.correct_count,
            "total": stats.steps,
        })

    def export(self, path: str = "cognicore_report.html") -> str:
        """Export as standalone HTML file."""
        html = self._build_html()
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        return path

    def _build_html(self) -> str:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # Build metrics cards
        metric_cards = ""
        if self.metrics:
            for name, data in self.metrics.items():
                metric_cards += f'''
                <div class="metric-card">
                    <div class="metric-value">{data["value"]}{data["unit"]}</div>
                    <div class="metric-label">{name}</div>
                </div>'''

        # Build episode results
        episode_rows = ""
        for i, ep in enumerate(self._episodes, 1):
            status = "pass" if ep["accuracy"] >= 0.5 else "fail"
            episode_rows += f'''
            <tr class="{status}">
                <td>{i}</td>
                <td>{ep["env_id"]}</td>
                <td>{ep["difficulty"]}</td>
                <td>{ep["accuracy"]:.0%}</td>
                <td>{ep["score"]:.4f}</td>
                <td>{ep["correct"]}/{ep["total"]}</td>
            </tr>'''

        # Build step heatmap
        heatmap_cells = ""
        for ep in self._episodes:
            for s in ep["steps"]:
                color = "#4ade80" if s["correct"] else "#f87171"
                heatmap_cells += f'<div class="heat-cell" style="background:{color}" title="{s["category"]}: {"OK" if s["correct"] else "FAIL"} (reward: {s["reward"]:+.2f})"></div>'

        # Build category breakdown
        cat_stats = {}
        for ep in self._episodes:
            for s in ep["steps"]:
                cat = s["category"]
                if cat not in cat_stats:
                    cat_stats[cat] = {"correct": 0, "total": 0}
                cat_stats[cat]["total"] += 1
                if s["correct"]:
                    cat_stats[cat]["correct"] += 1

        cat_rows = ""
        for cat, data in sorted(cat_stats.items(), key=lambda x: x[1]["correct"]/max(x[1]["total"],1)):
            acc = data["correct"] / data["total"] if data["total"] else 0
            bar_width = int(acc * 100)
            bar_color = "#4ade80" if acc >= 0.7 else "#fbbf24" if acc >= 0.4 else "#f87171"
            cat_rows += f'''
            <tr>
                <td>{cat}</td>
                <td>{data["correct"]}/{data["total"]}</td>
                <td>
                    <div class="bar-bg"><div class="bar-fill" style="width:{bar_width}%;background:{bar_color}"></div></div>
                </td>
                <td>{acc:.0%}</td>
            </tr>'''

        # Custom sections
        custom_sections = ""
        for sec in self.sections:
            custom_sections += f'''
            <div class="section">
                <h2>{sec["title"]}</h2>
                <pre>{sec["content"]}</pre>
            </div>'''

        # Overall stats
        total_correct = sum(ep["correct"] for ep in self._episodes)
        total_steps = sum(ep["total"] for ep in self._episodes)
        overall_acc = total_correct / total_steps if total_steps else 0

        return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{self.title}</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:'Segoe UI',system-ui,-apple-system,sans-serif; background:#0f172a; color:#e2e8f0; padding:2rem; }}
.container {{ max-width:1100px; margin:0 auto; }}
h1 {{ font-size:2rem; margin-bottom:0.25rem; background:linear-gradient(135deg,#60a5fa,#a78bfa); -webkit-background-clip:text; background-clip:text; color:transparent; }}
.subtitle {{ color:#94a3b8; margin-bottom:2rem; }}
.metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:1rem; margin-bottom:2rem; }}
.metric-card {{ background:#1e293b; border:1px solid #334155; border-radius:12px; padding:1.5rem; text-align:center; }}
.metric-value {{ font-size:2rem; font-weight:700; color:#60a5fa; }}
.metric-label {{ color:#94a3b8; font-size:0.85rem; margin-top:0.25rem; }}
.section {{ background:#1e293b; border:1px solid #334155; border-radius:12px; padding:1.5rem; margin-bottom:1.5rem; }}
h2 {{ font-size:1.2rem; color:#f1f5f9; margin-bottom:1rem; }}
table {{ width:100%; border-collapse:collapse; }}
th {{ text-align:left; padding:0.6rem; color:#94a3b8; border-bottom:1px solid #334155; font-size:0.85rem; }}
td {{ padding:0.6rem; border-bottom:1px solid #1e293b; }}
tr.pass td:nth-child(4) {{ color:#4ade80; }}
tr.fail td:nth-child(4) {{ color:#f87171; }}
.heatmap {{ display:flex; flex-wrap:wrap; gap:3px; margin:1rem 0; }}
.heat-cell {{ width:18px; height:18px; border-radius:3px; cursor:pointer; }}
.bar-bg {{ background:#334155; border-radius:4px; height:8px; width:100%; }}
.bar-fill {{ height:8px; border-radius:4px; transition:width 0.3s; }}
pre {{ background:#0f172a; padding:1rem; border-radius:8px; overflow-x:auto; font-size:0.85rem; color:#94a3b8; }}
.footer {{ text-align:center; color:#475569; margin-top:2rem; font-size:0.8rem; }}
</style>
</head>
<body>
<div class="container">
<h1>{self.title}</h1>
<p class="subtitle">Generated {timestamp} | CogniCore v{cognicore.__version__}</p>

<div class="metrics">
    <div class="metric-card"><div class="metric-value">{overall_acc:.0%}</div><div class="metric-label">Overall Accuracy</div></div>
    <div class="metric-card"><div class="metric-value">{len(self._episodes)}</div><div class="metric-label">Episodes</div></div>
    <div class="metric-card"><div class="metric-value">{total_steps}</div><div class="metric-label">Total Steps</div></div>
    <div class="metric-card"><div class="metric-value">{total_correct}</div><div class="metric-label">Correct</div></div>
    {metric_cards}
</div>

<div class="section">
<h2>Step Heatmap</h2>
<div class="heatmap">{heatmap_cells}</div>
<p style="color:#64748b;font-size:0.8rem;margin-top:0.5rem">Green = correct, Red = incorrect. Hover for details.</p>
</div>

<div class="section">
<h2>Episode Results</h2>
<table>
<tr><th>#</th><th>Environment</th><th>Difficulty</th><th>Accuracy</th><th>Score</th><th>Correct</th></tr>
{episode_rows}
</table>
</div>

<div class="section">
<h2>Category Breakdown</h2>
<table>
<tr><th>Category</th><th>Score</th><th>Performance</th><th>Accuracy</th></tr>
{cat_rows}
</table>
</div>

{custom_sections}

<div class="footer">CogniCore v{cognicore.__version__} | {len(cognicore.__all__)} exports | {len(cognicore.list_envs())} environments</div>
</div>
</body>
</html>'''
