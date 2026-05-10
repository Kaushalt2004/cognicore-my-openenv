// CogniCore Visual Demo — Interactive Training Visualization
const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);

// Tab navigation
$$('.nav-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    $$('.nav-btn').forEach(b => b.classList.remove('active'));
    $$('.tab-content').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
    $(`#tab-${btn.dataset.tab}`).classList.add('active');
    if(btn.dataset.tab==='bench') renderBenchmarks();
    if(btn.dataset.tab==='survival') initSurvivalBars();
    if(btn.dataset.tab==='arena') initArena();
  });
});

// ═══════ MAZE RUNNER ═══════
class Maze {
  constructor(size) {
    this.size = size;
    this.walls = new Set();
    this.agent = [1,1];
    this.goal = [size-2, size-2];
    this.visited = new Map();
    this.path = [];
    this.steps = 0;
    this.reward = 0;
    this.generate();
  }
  key(r,c) { return `${r},${c}`; }
  generate() {
    this.walls.clear();
    for(let r=0;r<this.size;r++) for(let c=0;c<this.size;c++) this.walls.add(this.key(r,c));
    const stack = [[1,1]];
    this.walls.delete(this.key(1,1));
    while(stack.length) {
      const [r,c] = stack[stack.length-1];
      const neighbors = [];
      for(const [dr,dc] of [[-2,0],[2,0],[0,-2],[0,2]]) {
        const nr=r+dr, nc=c+dc;
        if(nr>=1 && nr<this.size-1 && nc>=1 && nc<this.size-1 && this.walls.has(this.key(nr,nc)))
          neighbors.push([nr,nc,r+Math.floor(dr/2),c+Math.floor(dc/2)]);
      }
      if(neighbors.length) {
        const [nr,nc,wr,wc] = neighbors[Math.floor(Math.random()*neighbors.length)];
        this.walls.delete(this.key(nr,nc));
        this.walls.delete(this.key(wr,wc));
        stack.push([nr,nc]);
      } else stack.pop();
    }
    this.walls.delete(this.key(...this.goal));
    for(const [dr,dc] of [[-1,0],[1,0],[0,-1],[0,1]]) {
      const nr=this.goal[0]+dr, nc=this.goal[1]+dc;
      if(nr>=1&&nr<this.size-1&&nc>=1&&nc<this.size-1) this.walls.delete(this.key(nr,nc));
    }
    this.agent = [1,1];
    this.visited.clear();
    this.visited.set(this.key(1,1), 1);
    this.path = [[1,1]];
    this.steps = 0;
    this.reward = 0;
  }
  step(action) {
    const dirs = [[-1,0],[1,0],[0,-1],[0,1]];
    const [dr,dc] = dirs[action];
    const nr=this.agent[0]+dr, nc=this.agent[1]+dc;
    const nk = this.key(nr,nc);
    this.steps++;
    if(this.walls.has(nk)||nr<0||nr>=this.size||nc<0||nc>=this.size) { this.reward-=0.3; return 'wall'; }
    const oldDist = Math.abs(this.goal[0]-this.agent[0])+Math.abs(this.goal[1]-this.agent[1]);
    this.agent = [nr,nc];
    this.visited.set(nk, (this.visited.get(nk)||0)+1);
    this.path.push([nr,nc]);
    const newDist = Math.abs(this.goal[0]-nr)+Math.abs(this.goal[1]-nc);
    if(nr===this.goal[0]&&nc===this.goal[1]) { this.reward+=10; return 'goal'; }
    this.reward += newDist<oldDist ? 0.1 : -0.1;
    if(this.visited.get(nk)>1) this.reward -= 0.05;
    return 'step';
  }
}

// Q-Learning Agent
class QAgent {
  constructor() { this.q = {}; this.lr=0.15; this.gamma=0.95; this.epsilon=1.0; this.decay=0.997; }
  getQ(s,a) { return (this.q[`${s}_${a}`]||0); }
  setQ(s,a,v) { this.q[`${s}_${a}`]=v; }
  act(state) {
    if(Math.random()<this.epsilon) return Math.floor(Math.random()*4);
    let best=-Infinity, ba=0;
    for(let a=0;a<4;a++) { const v=this.getQ(state,a); if(v>best){best=v;ba=a;} }
    return ba;
  }
  learn(s,a,r,s2,done) {
    let maxQ2=done?0:-Infinity;
    if(!done) for(let a2=0;a2<4;a2++) maxQ2=Math.max(maxQ2,this.getQ(s2,a2));
    const target = r + this.gamma*maxQ2;
    this.setQ(s,a, this.getQ(s,a) + this.lr*(target-this.getQ(s,a)));
    this.epsilon *= this.decay;
  }
}

// Rendering
const mazeCanvas = $('#maze-canvas');
const mazeCtx = mazeCanvas.getContext('2d');
let maze, agent, training=false, episodeRewards=[], bestReward=-Infinity, episode=0;

function initMaze() {
  const size = parseInt($('#maze-size').value);
  maze = new Maze(size);
  agent = new QAgent();
  episodeRewards = [];
  bestReward = -Infinity;
  episode = 0;
  drawMaze();
}

function drawMaze() {
  const w=mazeCanvas.width, h=mazeCanvas.height;
  const cs = Math.floor(Math.min(w,h)/maze.size);
  mazeCtx.fillStyle='#0a0a0f'; mazeCtx.fillRect(0,0,w,h);
  const ox=Math.floor((w-cs*maze.size)/2), oy=Math.floor((h-cs*maze.size)/2);

  for(let r=0;r<maze.size;r++) for(let c=0;c<maze.size;c++) {
    const x=ox+c*cs, y=oy+r*cs, k=maze.key(r,c);
    if(maze.walls.has(k)) {
      mazeCtx.fillStyle='#2a2a3a'; mazeCtx.fillRect(x,y,cs,cs);
    } else if(maze.visited.has(k)) {
      const v=Math.min((maze.visited.get(k)||0)*20, 60);
      mazeCtx.fillStyle=`rgb(${20+v},${25+v/2},${35+v})`; mazeCtx.fillRect(x,y,cs,cs);
    } else {
      mazeCtx.fillStyle='#16162a'; mazeCtx.fillRect(x,y,cs,cs);
    }
    mazeCtx.strokeStyle='#1a1a28'; mazeCtx.strokeRect(x,y,cs,cs);
  }
  // Goal glow
  const gx=ox+maze.goal[1]*cs, gy=oy+maze.goal[0]*cs;
  const pulse = Math.abs(Math.sin(Date.now()/300))*0.4+0.6;
  mazeCtx.fillStyle=`rgba(255,200,50,${pulse})`; 
  mazeCtx.beginPath(); mazeCtx.roundRect(gx+3,gy+3,cs-6,cs-6,4); mazeCtx.fill();
  mazeCtx.fillStyle='#0a0a0f'; mazeCtx.font=`bold ${cs/2}px Inter`; mazeCtx.textAlign='center';
  mazeCtx.fillText('G',gx+cs/2,gy+cs/2+cs/6);
  // Agent glow
  const ax=ox+maze.agent[1]*cs, ay=oy+maze.agent[0]*cs;
  const grad=mazeCtx.createRadialGradient(ax+cs/2,ay+cs/2,0,ax+cs/2,ay+cs/2,cs);
  grad.addColorStop(0,'rgba(0,200,150,0.3)'); grad.addColorStop(1,'rgba(0,200,150,0)');
  mazeCtx.fillStyle=grad; mazeCtx.fillRect(ax-cs/2,ay-cs/2,cs*2,cs*2);
  mazeCtx.fillStyle=`rgba(0,255,180,${pulse})`;
  mazeCtx.beginPath(); mazeCtx.roundRect(ax+2,ay+2,cs-4,cs-4,6); mazeCtx.fill();
  mazeCtx.fillStyle='#0a0a0f'; mazeCtx.fillText('A',ax+cs/2,ay+cs/2+cs/6);

  // Stats
  const total = maze.size*maze.size - maze.walls.size;
  const explored = Math.round(maze.visited.size/Math.max(total,1)*100);
  $('#s-ep').textContent = episode;
  $('#s-steps').textContent = maze.steps;
  $('#s-reward').textContent = maze.reward.toFixed(1);
  $('#s-best').textContent = bestReward>-Infinity ? bestReward.toFixed(1) : '—';
  $('#s-explored').textContent = explored+'%';
  $('#s-reward').style.color = maze.reward>0?'#00c896':'#ff4466';
}

async function trainMaze() {
  if(training) return;
  training = true;
  $('#btn-start').textContent = '⏸ Training...';
  const maxEp = 80;
  for(let ep=0; ep<maxEp && training; ep++) {
    maze.generate();
    episode = ep+1;
    let state = maze.key(...maze.agent);
    for(let s=0; s<maze.size*maze.size*3 && training; s++) {
      const action = agent.act(state);
      const result = maze.step(action);
      const newState = maze.key(...maze.agent);
      const done = result==='goal' || s>=maze.size*maze.size*3-1;
      const r = result==='goal'?10:result==='wall'?-0.3:0;
      agent.learn(state, action, r, newState, done);
      state = newState;
      if(s%3===0) { drawMaze(); await sleep(8); }
      if(done) break;
    }
    episodeRewards.push(maze.reward);
    if(maze.reward>bestReward) bestReward=maze.reward;
    drawChart();
    drawMemoryMap();
    await sleep(30);
  }
  training = false;
  $('#btn-start').textContent = '▶ Train Agent';
}

function sleep(ms) { return new Promise(r=>setTimeout(r,ms)); }

// Learning curve chart
const chartCtx = $('#chart-canvas').getContext('2d');
function drawChart() {
  const w=$('#chart-canvas').width, h=$('#chart-canvas').height;
  chartCtx.fillStyle='#0a0a0f'; chartCtx.fillRect(0,0,w,h);
  if(!episodeRewards.length) return;
  const pad=40;
  const minR=Math.min(...episodeRewards,-5), maxR=Math.max(...episodeRewards,5);
  const range=maxR-minR||1;
  // Grid
  chartCtx.strokeStyle='#1a1a28'; chartCtx.lineWidth=1;
  for(let i=0;i<5;i++) {
    const y=pad+(h-2*pad)*i/4;
    chartCtx.beginPath(); chartCtx.moveTo(pad,y); chartCtx.lineTo(w-10,y); chartCtx.stroke();
    chartCtx.fillStyle='#555'; chartCtx.font='10px JetBrains Mono';
    chartCtx.fillText((maxR-range*i/4).toFixed(1), 2, y+4);
  }
  // Line
  chartCtx.beginPath(); chartCtx.strokeStyle='#00c896'; chartCtx.lineWidth=2;
  const step=(w-pad-10)/Math.max(episodeRewards.length-1,1);
  episodeRewards.forEach((r,i) => {
    const x=pad+i*step, y=pad+(h-2*pad)*(1-(r-minR)/range);
    i===0?chartCtx.moveTo(x,y):chartCtx.lineTo(x,y);
  });
  chartCtx.stroke();
  // Moving average
  if(episodeRewards.length>5) {
    chartCtx.beginPath(); chartCtx.strokeStyle='#ffc832'; chartCtx.lineWidth=2;
    const win=5;
    for(let i=win;i<episodeRewards.length;i++) {
      const avg=episodeRewards.slice(i-win,i).reduce((a,b)=>a+b,0)/win;
      const x=pad+i*step, y=pad+(h-2*pad)*(1-(avg-minR)/range);
      i===win?chartCtx.moveTo(x,y):chartCtx.lineTo(x,y);
    }
    chartCtx.stroke();
  }
  // Labels
  chartCtx.fillStyle='#00c896'; chartCtx.font='11px Inter'; chartCtx.fillText('Episode Reward',pad,14);
  chartCtx.fillStyle='#ffc832'; chartCtx.fillText('Moving Avg (5)', pad+120, 14);
}

// Memory heatmap
const memCtx = $('#memory-canvas').getContext('2d');
function drawMemoryMap() {
  const w=$('#memory-canvas').width, h=$('#memory-canvas').height;
  memCtx.fillStyle='#0a0a0f'; memCtx.fillRect(0,0,w,h);
  const entries = Object.keys(agent.q).length;
  const cols=Math.ceil(Math.sqrt(entries)), rows=Math.ceil(entries/Math.max(cols,1));
  if(!entries) return;
  const cw=Math.floor(w/Math.max(cols,1)), ch=Math.floor(h/Math.max(rows,1));
  let i=0;
  const vals=Object.values(agent.q);
  const maxV=Math.max(...vals.map(Math.abs),1);
  for(const [k,v] of Object.entries(agent.q)) {
    const col=i%cols, row=Math.floor(i/cols);
    const norm=v/maxV;
    const r=norm>0?0:Math.min(255,Math.floor(-norm*200));
    const g=norm>0?Math.min(255,Math.floor(norm*200)):0;
    memCtx.fillStyle=`rgb(${r},${g},${40+Math.abs(norm)*60})`;
    memCtx.fillRect(col*cw,row*ch,cw-1,ch-1);
    i++;
  }
  memCtx.fillStyle='#7a7a90'; memCtx.font='11px Inter';
  memCtx.fillText(`Q-Table: ${entries} entries`, 8, h-8);
}

$('#btn-start').addEventListener('click', () => { if(training){training=false;}else trainMaze(); });
$('#btn-reset').addEventListener('click', () => { training=false; initMaze(); });
$('#maze-size').addEventListener('change', () => { training=false; initMaze(); });

// ═══════ ARENA ═══════
const agents = [
  {name:'Q-Learning',elo:1200,w:0,l:0,color:'#00c896'},
  {name:'SARSA',elo:1200,w:0,l:0,color:'#4488ff'},
  {name:'Bandit',elo:1200,w:0,l:0,color:'#8855ff'},
  {name:'Genetic',elo:1200,w:0,l:0,color:'#ff8844'},
  {name:'Random',elo:1200,w:0,l:0,color:'#ff4466'},
];
let matches=[], eloHistory=agents.map(()=>[1200]);

function initArena() {
  agents.forEach(a=>{a.elo=1200;a.w=0;a.l=0;});
  matches=[]; eloHistory=agents.map(()=>[1200]);
  renderLeaderboard(); renderEloChart();
}

function simulateMatch(a,b) {
  const pA = 1/(1+Math.pow(10,(b.elo-a.elo)/400));
  const scoreA = Math.random()<pA ? 1 : 0;
  const K=32;
  a.elo += K*(scoreA-pA); b.elo += K*((1-scoreA)-(1-pA));
  if(scoreA) { a.w++; b.l++; } else { b.w++; a.l++; }
  return {a:a.name, b:b.name, winner: scoreA?a.name:b.name};
}

async function runTournament() {
  $('#btn-arena').textContent='Running...'; $('#btn-arena').disabled=true;
  const bracket=$('#arena-bracket'); bracket.innerHTML='';
  for(let round=0;round<3;round++) {
    for(let i=0;i<agents.length;i++) for(let j=i+1;j<agents.length;j++) {
      const m=simulateMatch(agents[i],agents[j]);
      matches.push(m);
      const card=document.createElement('div');
      card.className='match-card active';
      card.innerHTML=`<div class="agents"><span>${m.a}</span><span class="vs">vs</span><span>${m.b}</span></div><div class="result"><span class="winner">${m.winner} wins</span></div>`;
      bracket.prepend(card);
      agents.forEach((a,k)=>eloHistory[k].push(Math.round(a.elo)));
      renderLeaderboard(); renderEloChart();
      await sleep(200);
      card.classList.remove('active');
    }
  }
  $('#btn-arena').textContent='▶ Run Tournament'; $('#btn-arena').disabled=false;
}

function renderLeaderboard() {
  const sorted=[...agents].sort((a,b)=>b.elo-a.elo);
  const tbody=$('#leaderboard tbody');
  tbody.innerHTML=sorted.map((a,i)=>{
    const cls=i<3?`rank-${i+1}`:'';
    const medals=['🥇','🥈','🥉'];
    const rank=i<3?medals[i]:`${i+1}.`;
    const pct=a.w+a.l?Math.round(a.w/(a.w+a.l)*100):0;
    return `<tr class="${cls}"><td>${rank}</td><td>${a.name}</td><td>${Math.round(a.elo)}</td><td>${a.w}</td><td>${a.l}</td><td>${pct}%</td></tr>`;
  }).join('');
}

function renderEloChart() {
  const c=$('#elo-chart'), ctx=c.getContext('2d');
  const w=c.width, h=c.height; ctx.fillStyle='#0a0a0f'; ctx.fillRect(0,0,w,h);
  const allE=eloHistory.flat();
  if(allE.length<2) return;
  const minE=Math.min(...allE)-20, maxE=Math.max(...allE)+20, range=maxE-minE||1;
  const pad=30, maxLen=Math.max(...eloHistory.map(h=>h.length));
  agents.forEach((a,i)=>{
    const hist=eloHistory[i]; if(hist.length<2) return;
    ctx.beginPath(); ctx.strokeStyle=a.color; ctx.lineWidth=2;
    hist.forEach((v,j)=>{
      const x=pad+j*(w-pad-10)/(maxLen-1), y=pad+(h-2*pad)*(1-(v-minE)/range);
      j===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
    }); ctx.stroke();
    ctx.fillStyle=a.color; ctx.font='10px Inter';
    const lastY=pad+(h-2*pad)*(1-(hist[hist.length-1]-minE)/range);
    ctx.fillText(a.name, w-pad-60, lastY);
  });
}

$('#btn-arena').addEventListener('click', ()=>{ initArena(); runTournament(); });

// ═══════ SURVIVAL ═══════
function initSurvivalBars() {
  const bars=$('#survival-bars');
  bars.innerHTML=['Health','Hunger','Food','Materials','Shelter','Tools','Threat'].map(name=>{
    const colors={Health:'#00c896',Hunger:'#ff4466',Food:'#ffc832',Materials:'#aaa',Shelter:'#4488ff',Tools:'#8855ff',Threat:'#ff4466'};
    return `<div class="bar-row"><span class="bar-label">${name}</span><div class="bar-track"><div class="bar-fill" id="bar-${name.toLowerCase()}" style="width:${name==='Health'?100:name==='Hunger'?0:name==='Food'?25:0}%;background:${colors[name]}"></div></div><span class="bar-value" id="val-${name.toLowerCase()}">${name==='Health'?100:name==='Food'?5:0}</span></div>`;
  }).join('');
}

async function runSurvival() {
  const log=$('#survival-log'); log.innerHTML='';
  let hp=100,hunger=0,food=5,mat=0,shelter=0,tools=0,threat=0,day=0,alive=true;
  const acts=['Forage','Hunt','Build','Craft','Rest','Explore','Defend'];
  const chartCtx=$('#survival-chart').getContext('2d');
  const history=[];

  while(alive && day<150) {
    day++;
    hunger=Math.min(100,hunger+2);
    if(hunger>80) hp-=3;
    const action=acts[Math.floor(Math.random()*acts.length)];
    let event=action, cls='';

    if(action==='Forage'){const f=Math.floor(Math.random()*3);food+=f;mat+=Math.floor(Math.random()*2);event=`Foraged ${f} food`;cls='event-good';}
    else if(action==='Hunt'){if(tools>0&&Math.random()<0.6){food+=Math.floor(Math.random()*4)+3;event='Hunt success!';cls='event-good';}else{hp-=5;event='Hunt failed, injured';cls='event-danger';}}
    else if(action==='Build'){if(mat>=3){mat-=3;shelter=Math.min(5,shelter+1);event=`Built shelter (${shelter})`;cls='event-good';}else event='Not enough materials';}
    else if(action==='Craft'){if(mat>=2){mat-=2;tools++;event=`Crafted tool (${tools})`;cls='event-good';}else event='Not enough materials';}
    else if(action==='Rest'){const h=Math.min(15,100-hp);hp+=h;event=`Rested +${h}hp`;}
    else if(action==='Explore'){mat+=Math.floor(Math.random()*4)+1;event='Explored, found materials';cls='event-good';}
    else if(action==='Defend'&&threat>0){threat=0;event='Defended!';cls='event-good';}

    if(hunger>40&&food>0){const e=Math.min(food,2);food-=e;hunger=Math.max(0,hunger-e*25);}
    if(Math.random()<0.1&&shelter<2){hp-=8;event+=' ⛈ STORM';cls='event-danger';}
    if(Math.random()<0.08){threat=Math.floor(Math.random()*3)+1;}
    if(threat>0&&action!=='Defend'){hp-=threat*3;cls='event-danger';}
    if(hp<=0){alive=false;event='☠ DIED';cls='event-danger';}

    history.push({hp,hunger,food});
    log.innerHTML=`<div class="event ${cls}">Day ${day}: ${event}</div>`+log.innerHTML;
    updateBars(hp,hunger,food,mat,shelter,tools,threat);
    drawSurvivalChart(chartCtx,history);
    await sleep(60);
  }
  if(alive) log.innerHTML=`<div class="event event-good">🎉 Survived ${day} days!</div>`+log.innerHTML;
}

function updateBars(hp,hunger,food,mat,shelter,tools,threat) {
  const set=(id,v,max)=>{const el=$(`#bar-${id}`);if(el){el.style.width=Math.max(0,Math.min(100,v/max*100))+'%';} const vl=$(`#val-${id}`);if(vl)vl.textContent=Math.round(v);};
  set('health',hp,100);set('hunger',hunger,100);set('food',food,20);set('materials',mat,20);set('shelter',shelter,5);set('tools',tools,5);set('threat',threat,5);
}

function drawSurvivalChart(ctx,history) {
  const c=$('#survival-chart'),w=c.width,h=c.height;
  ctx.fillStyle='#0a0a0f';ctx.fillRect(0,0,w,h);
  if(!history.length) return;
  const pad=30, len=history.length;
  const draw=(key,color,max)=>{
    ctx.beginPath();ctx.strokeStyle=color;ctx.lineWidth=2;
    history.forEach((d,i)=>{const x=pad+i*(w-pad-10)/(Math.max(len-1,1)),y=pad+(h-2*pad)*(1-d[key]/max);i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);});
    ctx.stroke();
  };
  draw('hp','#00c896',100); draw('hunger','#ff4466',100); draw('food','#ffc832',20);
  ctx.fillStyle='#00c896';ctx.font='10px Inter';ctx.fillText('Health',pad,14);
  ctx.fillStyle='#ff4466';ctx.fillText('Hunger',pad+60,14);
  ctx.fillStyle='#ffc832';ctx.fillText('Food',pad+120,14);
}

$('#btn-survive').addEventListener('click',runSurvival);

// ═══════ BENCHMARKS ═══════
const benchData = [
  {name:'GridWorld-v0',desc:'5×5 navigation with traps',results:{PPO:{score:1.5,color:'#00c896'},DQN:{score:1.5,color:'#4488ff'},A2C:{score:0.8,color:'#8855ff'},Random:{score:-4.2,color:'#ff4466'}}},
  {name:'MazeRunner-v0',desc:'8×8 procedural maze',results:{PPO:{score:-33.2,color:'#00c896'},DQN:{score:-40.1,color:'#4488ff'},A2C:{score:-28.5,color:'#8855ff'},Random:{score:-55.0,color:'#ff4466'}}},
  {name:'Survival-v0',desc:'Long-horizon survival',results:{PPO:{score:199.9,color:'#00c896'},DQN:{score:120.3,color:'#4488ff'},A2C:{score:145.2,color:'#8855ff'},Random:{score:15.0,color:'#ff4466'}}},
  {name:'BattleArena-v0',desc:'2-player grid combat',results:{PPO:{score:19.2,color:'#00c896'},DQN:{score:12.1,color:'#4488ff'},A2C:{score:15.8,color:'#8855ff'},Random:{score:-3.5,color:'#ff4466'}}},
];

function renderBenchmarks() {
  const grid=$('#bench-grid'); grid.innerHTML='';
  benchData.forEach(env=>{
    const scores=Object.values(env.results).map(r=>r.score);
    const minS=Math.min(...scores), maxS=Math.max(...scores), range=maxS-minS||1;
    let html=`<div class="bench-card"><h3>cognicore/${env.name}</h3><div class="desc">${env.desc}</div>`;
    Object.entries(env.results).sort((a,b)=>b[1].score-a[1].score).forEach(([algo,{score,color}])=>{
      const pct=Math.max(5,((score-minS)/range)*100);
      html+=`<div class="bench-bar"><span class="algo" style="color:${color}">${algo}</span><div class="track"><div class="fill" style="width:${pct}%;background:${color}"><span class="score">${score>0?'+':''}${score.toFixed(1)}</span></div></div></div>`;
    });
    html+='</div>';
    grid.innerHTML+=html;
  });
}

// Init
initMaze();
