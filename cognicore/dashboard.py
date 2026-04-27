"""
CogniCore Interactive Dashboard — Claude-inspired beige conversational UI.

Features:
  - Chat-style interface to interact with environments
  - Live reward breakdown with animations
  - Memory visualization
  - Streak tracking with visual indicators
  - Agent comparison panel
  - Episode history timeline

Usage::
    cognicore dashboard --port 8050
"""

from __future__ import annotations

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CogniCore - Interactive Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap" rel="stylesheet">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }

  :root {
    --bg: #FAF6F1;
    --bg-dark: #F0EBE3;
    --surface: #FFFFFF;
    --surface-alt: #F7F3EE;
    --border: #E8E0D4;
    --border-hover: #D4C8B8;
    --text: #2D2A26;
    --text-secondary: #6B6560;
    --text-dim: #9B9590;
    --accent: #C4703D;
    --accent-light: #E8A06C;
    --accent-bg: #FFF3EB;
    --green: #3D8C6C;
    --green-bg: #E8F5EE;
    --red: #C44040;
    --red-bg: #FDEDED;
    --orange: #C4903D;
    --orange-bg: #FFF8EB;
    --blue: #3D6EC4;
    --blue-bg: #EBF0FF;
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.06);
    --shadow-md: 0 4px 12px rgba(0,0,0,0.08);
    --shadow-lg: 0 8px 24px rgba(0,0,0,0.1);
    --radius: 12px;
    --radius-lg: 16px;
  }

  body {
    font-family: 'Inter', -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
  }

  /* ---- Top Nav ---- */
  .topnav {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 0.8rem 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
    box-shadow: var(--shadow-sm);
  }
  .topnav .logo {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 1.3rem;
    font-weight: 600;
    color: var(--accent);
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .topnav .logo span {
    width: 28px; height: 28px;
    background: var(--accent);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 0.9rem;
    font-weight: 700;
  }
  .topnav .nav-links {
    display: flex;
    gap: 0.5rem;
  }
  .topnav .nav-links button {
    font-family: inherit;
    padding: 0.5rem 1rem;
    border: none;
    background: transparent;
    color: var(--text-secondary);
    font-size: 0.85rem;
    font-weight: 500;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.15s;
  }
  .topnav .nav-links button:hover { background: var(--bg-dark); color: var(--text); }
  .topnav .nav-links button.active { background: var(--accent-bg); color: var(--accent); }

  /* ---- Layout ---- */
  .container {
    max-width: 1100px;
    margin: 0 auto;
    padding: 1.5rem 2rem;
  }

  /* ---- Stats Bar ---- */
  .stats-bar {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
  }
  .stat-pill {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 0.5rem 1.2rem;
    font-size: 0.85rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    box-shadow: var(--shadow-sm);
  }
  .stat-pill .num { font-weight: 700; color: var(--accent); }

  /* ---- Section Tabs ---- */
  .tab-section { display: none; }
  .tab-section.active { display: block; }

  /* ---- Chat Interface ---- */
  .chat-container {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-md);
    overflow: hidden;
  }
  .chat-header {
    padding: 1rem 1.5rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--surface-alt);
  }
  .chat-header h2 {
    font-size: 1rem;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .chat-header .status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--green);
    display: inline-block;
  }
  .chat-controls {
    display: flex;
    gap: 0.5rem;
    align-items: center;
  }
  .chat-controls select {
    font-family: inherit;
    padding: 0.4rem 0.8rem;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--surface);
    color: var(--text);
    font-size: 0.8rem;
    cursor: pointer;
  }
  .chat-controls select:hover { border-color: var(--border-hover); }

  .chat-messages {
    padding: 1.5rem;
    max-height: 500px;
    overflow-y: auto;
    scroll-behavior: smooth;
  }
  .msg {
    margin-bottom: 1rem;
    animation: msgIn 0.3s ease-out;
  }
  @keyframes msgIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
  .msg-system {
    background: var(--surface-alt);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.2rem;
    font-size: 0.85rem;
    color: var(--text-secondary);
    line-height: 1.5;
  }
  .msg-step {
    display: flex;
    gap: 1rem;
    align-items: flex-start;
  }
  .msg-step .step-icon {
    width: 36px; height: 36px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.8rem;
    flex-shrink: 0;
  }
  .msg-step .step-icon.correct { background: var(--green-bg); color: var(--green); }
  .msg-step .step-icon.wrong { background: var(--red-bg); color: var(--red); }
  .msg-step .step-body {
    flex: 1;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0.8rem 1rem;
  }
  .msg-step .step-body .step-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.4rem;
  }
  .msg-step .step-body .step-header .step-num {
    font-weight: 600;
    font-size: 0.85rem;
  }
  .tag {
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
  }
  .tag-correct { background: var(--green-bg); color: var(--green); }
  .tag-wrong { background: var(--red-bg); color: var(--red); }
  .tag-streak { background: var(--orange-bg); color: var(--orange); }
  .tag-memory { background: var(--blue-bg); color: var(--blue); }
  .tag-novelty { background: var(--accent-bg); color: var(--accent); }

  .step-details {
    font-size: 0.8rem;
    color: var(--text-secondary);
    line-height: 1.5;
  }
  .step-details .pred-truth {
    display: flex;
    gap: 1rem;
    margin-top: 0.3rem;
  }
  .step-details .pred-truth span { font-family: monospace; font-size: 0.78rem; }

  .reward-chips {
    display: flex;
    gap: 0.3rem;
    margin-top: 0.5rem;
    flex-wrap: wrap;
  }
  .reward-chip {
    font-size: 0.7rem;
    padding: 0.2rem 0.5rem;
    border-radius: 6px;
    font-weight: 600;
    font-family: monospace;
  }
  .reward-chip.positive { background: var(--green-bg); color: var(--green); }
  .reward-chip.negative { background: var(--red-bg); color: var(--red); }
  .reward-chip.neutral { background: var(--bg-dark); color: var(--text-dim); }

  .msg-summary {
    background: var(--accent-bg);
    border: 1px solid #E8D4C0;
    border-radius: var(--radius);
    padding: 1.2rem 1.5rem;
  }
  .msg-summary h3 {
    font-size: 0.95rem;
    color: var(--accent);
    margin-bottom: 0.6rem;
  }
  .msg-summary .summary-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.8rem;
  }
  .summary-stat {
    text-align: center;
  }
  .summary-stat .val {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text);
  }
  .summary-stat .lbl {
    font-size: 0.7rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  /* ---- Streak Bar ---- */
  .streak-bar {
    display: flex;
    gap: 3px;
    margin-top: 0.6rem;
  }
  .streak-dot {
    width: 20px;
    height: 6px;
    border-radius: 3px;
    transition: background 0.3s;
  }
  .streak-dot.correct { background: var(--green); }
  .streak-dot.wrong { background: var(--red); }
  .streak-dot.pending { background: var(--border); }

  /* ---- Chat Input ---- */
  .chat-input-area {
    padding: 1rem 1.5rem;
    border-top: 1px solid var(--border);
    background: var(--surface-alt);
    display: flex;
    gap: 0.8rem;
    align-items: center;
  }
  .chat-input-area .run-btn {
    font-family: inherit;
    padding: 0.6rem 1.5rem;
    border: none;
    background: var(--accent);
    color: white;
    font-size: 0.85rem;
    font-weight: 600;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.15s;
    white-space: nowrap;
  }
  .chat-input-area .run-btn:hover { background: #B3622E; transform: translateY(-1px); }
  .chat-input-area .run-btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
  .chat-input-area .run-info {
    font-size: 0.8rem;
    color: var(--text-dim);
    flex: 1;
  }

  /* ---- Memory Panel ---- */
  .side-panels {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-top: 1.5rem;
  }
  .panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-sm);
    overflow: hidden;
  }
  .panel-header {
    padding: 0.8rem 1.2rem;
    border-bottom: 1px solid var(--border);
    background: var(--surface-alt);
    font-size: 0.85rem;
    font-weight: 600;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .panel-body { padding: 1rem 1.2rem; }
  .panel-body.empty {
    text-align: center;
    padding: 2rem;
    color: var(--text-dim);
    font-size: 0.85rem;
  }

  /* Memory entries */
  .mem-entry {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.5rem 0;
    border-bottom: 1px solid var(--bg-dark);
    font-size: 0.8rem;
  }
  .mem-entry:last-child { border: none; }
  .mem-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .mem-dot.ok { background: var(--green); }
  .mem-dot.fail { background: var(--red); }
  .mem-cat { font-weight: 500; flex: 1; }
  .mem-pred { color: var(--text-dim); font-family: monospace; font-size: 0.75rem; }

  /* Reward summary bars */
  .reward-bar-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.4rem 0;
    font-size: 0.8rem;
  }
  .reward-bar-label { width: 80px; color: var(--text-secondary); text-align: right; }
  .reward-bar-track {
    flex: 1;
    height: 8px;
    background: var(--bg-dark);
    border-radius: 4px;
    overflow: hidden;
    position: relative;
  }
  .reward-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.5s ease;
  }
  .reward-bar-fill.pos { background: var(--green); }
  .reward-bar-fill.neg { background: var(--red); }
  .reward-bar-val { width: 50px; font-weight: 600; font-family: monospace; font-size: 0.78rem; }

  /* Env cards */
  .env-cards {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 1rem;
    margin-top: 1rem;
  }
  .env-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem;
    cursor: pointer;
    transition: all 0.2s;
    box-shadow: var(--shadow-sm);
  }
  .env-card:hover { border-color: var(--accent); box-shadow: var(--shadow-md); transform: translateY(-2px); }
  .env-card h3 { font-size: 0.95rem; margin-bottom: 0.3rem; }
  .env-card p { font-size: 0.8rem; color: var(--text-secondary); line-height: 1.4; }
  .env-card .env-badges { display: flex; gap: 0.3rem; margin-top: 0.6rem; }
  .env-badge {
    font-size: 0.65rem; padding: 0.15rem 0.5rem;
    border-radius: 20px; font-weight: 600;
  }
  .env-badge.easy { background: var(--green-bg); color: var(--green); }
  .env-badge.medium { background: var(--orange-bg); color: var(--orange); }
  .env-badge.hard { background: var(--red-bg); color: var(--red); }

  /* Leaderboard */
  table { width: 100%; border-collapse: collapse; }
  th { text-align: left; padding: 0.6rem 1rem; font-size: 0.75rem; color: var(--text-dim);
       text-transform: uppercase; letter-spacing: 0.04em; border-bottom: 1px solid var(--border); }
  td { padding: 0.7rem 1rem; font-size: 0.85rem; border-bottom: 1px solid var(--bg-dark); }
  tr:hover td { background: var(--surface-alt); }

  /* ---- Responsive ---- */
  @media (max-width: 768px) {
    .side-panels { grid-template-columns: 1fr; }
    .msg-summary .summary-grid { grid-template-columns: repeat(2, 1fr); }
    .container { padding: 1rem; }
    .stats-bar { flex-wrap: wrap; }
  }
</style>
</head>
<body>

<div class="topnav">
  <div class="logo"><span>C</span> CogniCore</div>
  <div class="nav-links">
    <button class="active" onclick="showTab('interact')">Interact</button>
    <button onclick="showTab('envs')">Environments</button>
    <button onclick="showTab('leaders')">Leaderboard</button>
  </div>
</div>

<div class="container">
  <div class="stats-bar" id="stats-bar">
    <div class="stat-pill"><span class="num" id="s-envs">--</span> environments</div>
    <div class="stat-pill"><span class="num" id="s-sessions">0</span> active sessions</div>
    <div class="stat-pill"><span class="num" id="s-total">0</span> steps run</div>
  </div>

  <!-- ---- INTERACT TAB ---- -->
  <div class="tab-section active" id="tab-interact">
    <div class="chat-container">
      <div class="chat-header">
        <h2><span class="status-dot"></span> Interactive Runner</h2>
        <div class="chat-controls">
          <select id="env-sel"></select>
          <select id="diff-sel">
            <option value="easy">Easy</option>
            <option value="medium">Medium</option>
            <option value="hard">Hard</option>
          </select>
        </div>
      </div>
      <div class="chat-messages" id="chat">
        <div class="msg msg-system">
          Welcome to CogniCore. Select an environment and click <b>Run Episode</b> to watch an agent interact.
          <br><br>You'll see <b>memory bonuses</b> when the agent recognizes past categories, <b>streak penalties</b> for consecutive failures, and <b>novelty bonuses</b> for first encounters.
        </div>
      </div>
      <div class="chat-input-area">
        <button class="run-btn" id="run-btn" onclick="runEpisode()">Run Episode</button>
        <div class="run-info" id="run-info">Ready. Choose an environment above.</div>
      </div>
    </div>

    <div class="side-panels">
      <div class="panel">
        <div class="panel-header">Memory <span id="mem-count" style="color:var(--text-dim);font-weight:400;font-size:0.8rem">0 entries</span></div>
        <div class="panel-body empty" id="mem-panel">Run an episode to see memory build up</div>
      </div>
      <div class="panel">
        <div class="panel-header">Reward Components</div>
        <div class="panel-body" id="reward-panel">
          <div class="reward-bar-row"><span class="reward-bar-label">Base</span><div class="reward-bar-track"><div class="reward-bar-fill pos" id="rb-base" style="width:0"></div></div><span class="reward-bar-val" id="rv-base">0</span></div>
          <div class="reward-bar-row"><span class="reward-bar-label">Memory</span><div class="reward-bar-track"><div class="reward-bar-fill pos" id="rb-mem" style="width:0"></div></div><span class="reward-bar-val" id="rv-mem">0</span></div>
          <div class="reward-bar-row"><span class="reward-bar-label">Reflection</span><div class="reward-bar-track"><div class="reward-bar-fill pos" id="rb-ref" style="width:0"></div></div><span class="reward-bar-val" id="rv-ref">0</span></div>
          <div class="reward-bar-row"><span class="reward-bar-label">Streak</span><div class="reward-bar-track"><div class="reward-bar-fill neg" id="rb-str" style="width:0"></div></div><span class="reward-bar-val" id="rv-str">0</span></div>
          <div class="reward-bar-row"><span class="reward-bar-label">Novelty</span><div class="reward-bar-track"><div class="reward-bar-fill pos" id="rb-nov" style="width:0"></div></div><span class="reward-bar-val" id="rv-nov">0</span></div>
          <div class="reward-bar-row"><span class="reward-bar-label">Confidence</span><div class="reward-bar-track"><div class="reward-bar-fill pos" id="rb-conf" style="width:0"></div></div><span class="reward-bar-val" id="rv-conf">0</span></div>
        </div>
      </div>
    </div>
  </div>

  <!-- ---- ENVS TAB ---- -->
  <div class="tab-section" id="tab-envs">
    <h2 style="font-size:1.2rem;margin-bottom:0.5rem;font-family:'Source Serif 4',serif">Available Environments</h2>
    <p style="color:var(--text-secondary);font-size:0.85rem;margin-bottom:1rem">Every environment includes built-in memory, reflection, structured rewards, and PROPOSE-Revise.</p>
    <div class="env-cards" id="env-cards"></div>
  </div>

  <!-- ---- LEADERBOARD TAB ---- -->
  <div class="tab-section" id="tab-leaders">
    <h2 style="font-size:1.2rem;margin-bottom:1rem;font-family:'Source Serif 4',serif">Leaderboard</h2>
    <div class="panel">
      <table>
        <thead><tr><th>Rank</th><th>Agent</th><th>Environment</th><th>Score</th><th>Accuracy</th></tr></thead>
        <tbody id="lb-body"><tr><td colspan="5" style="text-align:center;color:var(--text-dim);padding:2rem">Run episodes to populate the leaderboard</td></tr></tbody>
      </table>
    </div>
  </div>
</div>

<script>
const API = window.location.origin;
let totalSteps = 0;
let activeSessions = 0;

function showTab(name) {
  document.querySelectorAll('.tab-section').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.nav-links button').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  document.querySelector(`[onclick="showTab('${name}')"]`).classList.add('active');
}

async function fetchJSON(url) {
  try { const r = await fetch(API + url); return await r.json(); }
  catch(e) { return null; }
}

async function postJSON(url, body) {
  try {
    const r = await fetch(API + url, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
    return await r.json();
  } catch(e) { return null; }
}

function addMsg(html, cls='') {
  const chat = document.getElementById('chat');
  const d = document.createElement('div');
  d.className = 'msg ' + cls;
  d.innerHTML = html;
  chat.appendChild(d);
  chat.scrollTop = chat.scrollHeight;
}

function updateRewardBars(reward) {
  const map = {
    base: [reward.base_score, 'pos'], mem: [reward.memory_bonus, 'pos'],
    ref: [reward.reflection_bonus, 'pos'], str: [reward.streak_penalty, 'neg'],
    nov: [reward.novelty_bonus, 'pos'], conf: [reward.confidence_calibration||0, 'pos']
  };
  for (const [k, [v, cls]] of Object.entries(map)) {
    const bar = document.getElementById('rb-' + k);
    const val = document.getElementById('rv-' + k);
    const pct = Math.min(Math.abs(v) * 100, 100);
    bar.style.width = pct + '%';
    bar.className = 'reward-bar-fill ' + (v >= 0 ? 'pos' : 'neg');
    val.textContent = (v >= 0 ? '+' : '') + v.toFixed(3);
    val.style.color = v > 0 ? 'var(--green)' : v < 0 ? 'var(--red)' : 'var(--text-dim)';
  }
}

function updateMemPanel(memories) {
  const panel = document.getElementById('mem-panel');
  document.getElementById('mem-count').textContent = memories.length + ' entries';
  if (!memories.length) { panel.innerHTML = '<div class="empty">No memories yet</div>'; return; }
  const recent = memories.slice(-8).reverse();
  panel.innerHTML = recent.map(m =>
    `<div class="mem-entry">
      <div class="mem-dot ${m.correct ? 'ok' : 'fail'}"></div>
      <span class="mem-cat">${m.category || '?'}</span>
      <span class="mem-pred">${(m.predicted||'').slice(0,15)}</span>
    </div>`
  ).join('');
}

function getRandomAction(envId) {
  if (envId.includes('Safety')) return {classification: ['SAFE','UNSAFE','NEEDS_REVIEW'][Math.floor(Math.random()*3)]};
  if (envId.includes('Math')) return {answer: Math.floor(Math.random()*1000)};
  if (envId.includes('Code')) return {bug_line: Math.floor(Math.random()*10)+1, fix_type: ['wrong_operator','syntax_error','missing_check'][Math.floor(Math.random()*3)]};
  if (envId.includes('Conversation')) return {response: ['empathetic_action','dismissive','personalized_suggestion'][Math.floor(Math.random()*3)]};
  if (envId.includes('Planning')) return {order: ['A','B','C','D','E'].sort(() => Math.random() - 0.5)};
  if (envId.includes('Summar')) return {summary: 'A brief test summary of the text.'};
  return {action: 'default'};
}

async function runEpisode() {
  const envId = document.getElementById('env-sel').value;
  const diff = document.getElementById('diff-sel').value;
  const btn = document.getElementById('run-btn');
  const info = document.getElementById('run-info');
  btn.disabled = true;

  addMsg(`<div class="msg-system">Starting <b>${envId}</b> (${diff}) with RandomAgent...</div>`);
  info.textContent = 'Creating session...';

  const session = await postJSON(`/envs/${envId}/create`, {difficulty: diff});
  if (!session) { info.textContent = 'Error'; btn.disabled = false; return; }
  const sid = session.session_id;
  activeSessions++;
  document.getElementById('s-sessions').textContent = activeSessions;

  await postJSON(`/sessions/${sid}/reset`, {});

  let done = false, step = 0, streak = 0, correct_count = 0;
  let streakDots = [];
  let allMemories = [];

  while (!done) {
    step++;
    totalSteps++;
    document.getElementById('s-total').textContent = totalSteps;
    info.textContent = `Step ${step}...`;

    const action = getRandomAction(envId);
    const result = await postJSON(`/sessions/${sid}/step`, {action});
    if (!result) break;

    done = result.done;
    const er = result.info?.eval_result || {};
    const reward = result.reward;
    const correct = er.correct;

    if (correct) { streak = Math.max(streak+1, 1); correct_count++; }
    else { streak = Math.min(streak-1, -1); }
    streakDots.push(correct);

    // Build tags
    let tags = '';
    if (correct) tags += '<span class="tag tag-correct">Correct</span> ';
    else tags += '<span class="tag tag-wrong">Wrong</span> ';
    if (reward.memory_bonus > 0) tags += '<span class="tag tag-memory">Memory +' + reward.memory_bonus.toFixed(2) + '</span> ';
    if (reward.streak_penalty < 0) tags += '<span class="tag tag-streak">Streak ' + reward.streak_penalty.toFixed(2) + '</span> ';
    if (reward.novelty_bonus > 0) tags += '<span class="tag tag-novelty">Novelty +' + reward.novelty_bonus.toFixed(2) + '</span> ';

    // Build reward chips
    let chips = '';
    const comps = [
      ['base', reward.base_score], ['memory', reward.memory_bonus],
      ['streak', reward.streak_penalty], ['novelty', reward.novelty_bonus],
      ['total', reward.total]
    ];
    chips = comps.filter(([,v]) => v !== 0).map(([n,v]) =>
      `<span class="reward-chip ${v>0?'positive':v<0?'negative':'neutral'}">${n}: ${v>=0?'+':''}${v.toFixed(2)}</span>`
    ).join('');

    // Streak bar
    const streakHtml = streakDots.map(c =>
      `<div class="streak-dot ${c ? 'correct' : 'wrong'}"></div>`
    ).join('');

    addMsg(`
      <div class="msg-step">
        <div class="step-icon ${correct ? 'correct' : 'wrong'}">${step}</div>
        <div class="step-body">
          <div class="step-header">
            <span class="step-num">${er.category || 'Step ' + step}</span>
            ${tags}
          </div>
          <div class="step-details">
            <div class="pred-truth">
              <span>pred: <b>${(er.predicted||'').slice(0,25)}</b></span>
              <span>truth: <b>${(er.ground_truth||'').slice(0,25)}</b></span>
              <span>reward: <b>${reward.total >= 0 ? '+':''}${reward.total.toFixed(2)}</b></span>
            </div>
            <div class="reward-chips">${chips}</div>
            <div class="streak-bar">${streakHtml}</div>
          </div>
        </div>
      </div>
    `);

    // Track memory
    allMemories.push({category: er.category, predicted: er.predicted, correct});
    updateMemPanel(allMemories);
    updateRewardBars(reward);

    await new Promise(r => setTimeout(r, 120));
  }

  // Final stats
  const stats = await fetchJSON(`/sessions/${sid}/stats`);
  if (stats) {
    addMsg(`
      <div class="msg-summary">
        <h3>Episode Complete</h3>
        <div class="summary-grid">
          <div class="summary-stat"><div class="val">${stats.score.toFixed(3)}</div><div class="lbl">Score</div></div>
          <div class="summary-stat"><div class="val">${(stats.accuracy*100).toFixed(0)}%</div><div class="lbl">Accuracy</div></div>
          <div class="summary-stat"><div class="val">${stats.correct_count}/${stats.steps}</div><div class="lbl">Correct</div></div>
          <div class="summary-stat"><div class="val">${stats.memory_entries_created}</div><div class="lbl">Memories</div></div>
        </div>
      </div>
    `);
  }

  await fetch(API + `/sessions/${sid}`, {method:'DELETE'});
  activeSessions--;
  document.getElementById('s-sessions').textContent = activeSessions;
  info.textContent = 'Done! Run another episode or switch environments.';
  btn.disabled = false;
}

async function init() {
  const envData = await fetchJSON('/envs');
  if (!envData) return;
  const envs = envData.environments;
  document.getElementById('s-envs').textContent = envs.length;

  // Populate select
  const sel = document.getElementById('env-sel');
  const baseEnvs = envs.filter(e => !e.id.includes('Easy-v1') && !e.id.includes('Medium-v1') && !e.id.includes('Hard-v1'));
  baseEnvs.forEach(e => { const o = document.createElement('option'); o.value = e.id; o.textContent = e.id; sel.appendChild(o); });

  // Env cards
  const cardsEl = document.getElementById('env-cards');
  const groups = {};
  envs.forEach(e => {
    const base = e.id.replace(/-Easy|-Medium|-Hard/g, '');
    if (!groups[base]) groups[base] = {id: base, desc: e.description, diffs: new Set()};
    if (e.id.includes('Easy')) groups[base].diffs.add('easy');
    else if (e.id.includes('Medium')) groups[base].diffs.add('medium');
    else if (e.id.includes('Hard')) groups[base].diffs.add('hard');
  });
  Object.values(groups).forEach(g => {
    const desc = g.desc.length > 90 ? g.desc.slice(0,87) + '...' : g.desc;
    const badges = [...g.diffs].map(d => `<span class="env-badge ${d}">${d}</span>`).join('');
    cardsEl.innerHTML += `<div class="env-card" onclick="document.getElementById('env-sel').value='${g.id}';showTab('interact')">
      <h3>${g.id}</h3><p>${desc}</p><div class="env-badges">${badges}</div></div>`;
  });

  // Leaderboard
  try {
    const lb = await fetchJSON('/leaderboard');
    if (lb && lb.rankings && lb.rankings.length) {
      document.getElementById('lb-body').innerHTML = lb.rankings.map(r =>
        `<tr><td>#${r.rank}</td><td>${r.agent_id}</td><td>${r.env_id}</td>
         <td>${r.score.toFixed(4)}</td><td>${(r.accuracy*100).toFixed(0)}%</td></tr>`
      ).join('');
    }
  } catch(e) {}
}

init();
</script>
</body>
</html>"""


def create_dashboard_app():
    """Create a FastAPI app that serves the dashboard + API."""
    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse
        from cognicore.server.app import create_app as create_api_app
    except ImportError:
        raise ImportError("Dashboard requires FastAPI. Install with: pip install cognicore[server]")

    app = create_api_app()

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard():
        return DASHBOARD_HTML

    @app.get("/leaderboard")
    def get_leaderboard():
        from cognicore.leaderboard import Leaderboard
        lb = Leaderboard()
        return {
            "rankings": lb.get_rankings(top_k=50),
            "stats": lb.get_stats(),
        }

    return app


def serve_dashboard(host: str = "127.0.0.1", port: int = 8050) -> None:
    """Start the dashboard server."""
    try:
        import uvicorn
    except ImportError:
        raise ImportError("Dashboard requires uvicorn. Install with: pip install cognicore[server]")

    import cognicore
    print(f"\nCogniCore Dashboard v{cognicore.__version__}")
    print(f"  Dashboard: http://localhost:{port}/dashboard")
    print(f"  API Docs:  http://localhost:{port}/docs")
    print()

    app = create_dashboard_app()
    uvicorn.run(app, host=host, port=port, log_level="warning")
