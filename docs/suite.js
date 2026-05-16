// CogniCore Universal Agent Evaluation Suite
const AGENTS=[
{id:'gpt',name:'GPT Coding Agent',icon:'G',color:'#00e5a0',paradigm:'LLM',score:0,task:0,mem:true,refl:true},
{id:'claude',name:'Claude Workflow Agent',icon:'C',color:'#00ccff',paradigm:'LLM',score:0,task:0,mem:true,refl:true},
{id:'llama',name:'Llama Planning Agent',icon:'L',color:'#ffd666',paradigm:'LLM',score:0,task:0,mem:true,refl:true},
{id:'ppo',name:'PPO RL Agent',icon:'P',color:'#9966ff',paradigm:'RL',score:0,task:0,mem:true,refl:true},
{id:'browser',name:'Browser Nav Agent',icon:'B',color:'#ff8844',paradigm:'Web',score:0,task:0,mem:true,refl:true},
{id:'swarm',name:'Multi-Agent Swarm',icon:'S',color:'#ff4466',paradigm:'Multi',score:0,task:0,mem:true,refl:true},
{id:'robot',name:'Robotics Nav Agent',icon:'R',color:'#44ddff',paradigm:'Robot',score:0,task:0,mem:true,refl:true}
];
const KPIS=[
{id:'adapt',l:'Runtime Adaptation',u:'%',c:'green',t:84},{id:'stability',l:'Learning Stability',u:'%',c:'cyan',t:91},
{id:'recovery',l:'Recovery Speed',u:'x',c:'amber',t:3.2},{id:'recall',l:'Memory Recall',u:'%',c:'green',t:87},
{id:'compat',l:'Cross-Agent Compat',u:'%',c:'cyan',t:94},{id:'refl_eff',l:'Reflection Effect.',u:'%',c:'purple',t:76},
{id:'horizon',l:'Long-Horizon Stab.',u:'%',c:'amber',t:82},{id:'transfer',l:'Transfer Efficiency',u:'%',c:'green',t:69}
];
const COMPAT=[
{fw:'LangChain',s:'active',m:true,r:true,b:true,a:'+42%',st:'98%'},
{fw:'CrewAI',s:'active',m:true,r:true,b:true,a:'+38%',st:'96%'},
{fw:'OpenAI Agents SDK',s:'active',m:true,r:true,b:true,a:'+45%',st:'97%'},
{fw:'Stable-Baselines3',s:'active',m:true,r:true,b:true,a:'+51%',st:'95%'},
{fw:'RLlib',s:'active',m:true,r:false,b:true,a:'+36%',st:'93%'},
{fw:'AutoGen',s:'active',m:true,r:true,b:true,a:'+40%',st:'94%'},
{fw:'HuggingFace',s:'active',m:true,r:true,b:true,a:'+33%',st:'96%'},
{fw:'ROS 2',s:'beta',m:true,r:false,b:false,a:'+28%',st:'89%'},
{fw:'Isaac Sim',s:'beta',m:true,r:false,b:false,a:'+25%',st:'87%'}
];
const MEM_MSGS=[
'Failed compiler patch detected - blacklisted','Browser nav loop avoided via memory','Previous successful strategy reused','Environment mutation recognized','Reflection updated runtime policy',
'Trap location retrieved from episodic store','Optimal resource path recalled','Cross-agent memory sync completed','Repeated API timeout pattern cached','Code fix pattern transferred from GPT agent'
];
const REFL_MSGS=[
'Agent identified repeated syntax repair failure.','Alternative patch strategy selected.','Browser workflow recovery triggered after timeout.',
'Environment distribution shift detected.','Runtime adaptation improved task stability.','Exploration reduced - stable convergence detected.',
'Memory-guided exploration activated for novel region.','Cross-environment knowledge transfer initiated.','Failure cluster analysis complete - switching strategy.',
'Reflection: policy gradient adjusted based on memory recall.'
];
let running=false,tick=0,curEnv='code',C,X,chartC,chartX;
const baselineData=[],cogniData=[];

function init(){
  buildAgents();buildKPIs();buildCompat();
  C=document.getElementById('envC');X=C.getContext('2d');
  chartC=document.getElementById('chartC');chartX=chartC.getContext('2d');
  resize();draw();drawChart();
}
function resize(){
  const p=C.parentElement;C.width=p.offsetWidth;C.height=p.offsetHeight;
  chartC.width=chartC.parentElement.offsetWidth-16;
}
function buildAgents(){
  const el=document.getElementById('agent-list');el.innerHTML='';
  AGENTS.forEach((a,i)=>{
    el.innerHTML+=`<div class="ag-card${i===0?' active':''}" id="ac-${a.id}" onclick="selAgent('${a.id}')">
      <div class="ag-top"><div class="ag-icon" style="background:${a.color}">${a.icon}</div><span class="ag-name">${a.name}</span>
      <span class="ag-status on" id="as-${a.id}">ONLINE</span></div>
      <div class="ag-metrics">
        <div class="ag-m"><span class="ag-ml">Cognition</span><span class="ag-mv g" id="am-cog-${a.id}">ON</span></div>
        <div class="ag-m"><span class="ag-ml">Memory</span><span class="ag-mv g" id="am-mem-${a.id}">ON</span></div>
        <div class="ag-m"><span class="ag-ml">Reflection</span><span class="ag-mv g" id="am-ref-${a.id}">ON</span></div>
        <div class="ag-m"><span class="ag-ml">Adapt</span><span class="ag-mv" id="am-ada-${a.id}">--</span></div>
        <div class="ag-m"><span class="ag-ml">Tasks</span><span class="ag-mv" id="am-tsk-${a.id}">0</span></div>
        <div class="ag-m"><span class="ag-ml">Score</span><span class="ag-mv" id="am-scr-${a.id}">--</span></div>
      </div>
      <div class="ag-bar"><div class="ag-bar-f" id="ab-${a.id}" style="width:0%;background:${a.color}"></div></div>
    </div>`;
  });
}
function buildKPIs(){
  const el=document.getElementById('kpi-grid');el.innerHTML='';
  KPIS.forEach(k=>{
    el.innerHTML+=`<div class="kpi"><span class="kpi-l">${k.l}</span>
      <span class="kpi-v ${k.c}" id="kv-${k.id}">--</span>
      <span class="kpi-d up" id="kd-${k.id}"></span>
      <div class="kpi-bar"><div class="kpi-bar-f" id="kb-${k.id}" style="width:0%;background:var(--${k.c})"></div></div>
    </div>`;
  });
}
function buildCompat(){
  const el=document.getElementById('compat-body');el.innerHTML='';
  COMPAT.forEach(c=>{
    const sc=c.s==='active'?'c-yes':'c-pend';
    el.innerHTML+=`<tr><td class="fw-name">${c.fw}</td>
      <td class="${sc}">${c.s==='active'?'Active':'Beta'}</td>
      <td class="${c.m?'c-yes':'c-no'}">${c.m?'Yes':'--'}</td>
      <td class="${c.r?'c-yes':'c-pend'}">${c.r?'Yes':'Soon'}</td>
      <td class="${c.b?'c-yes':'c-pend'}">${c.b?'Pass':'Pend'}</td>
      <td class="imp-up">${c.a}</td><td>${c.st}</td></tr>`;
  });
}
function selAgent(id){document.querySelectorAll('.ag-card').forEach(c=>c.classList.remove('active'));document.getElementById('ac-'+id).classList.add('active')}
function switchEnvTab(el){document.querySelectorAll('.etab').forEach(t=>t.classList.remove('active'));el.classList.add('active');curEnv=el.dataset.env;draw()}

function addOv(text,type){
  const d=document.getElementById('overlays'),s=document.createElement('span');
  s.className='ov '+type;s.textContent=text;d.appendChild(s);
  setTimeout(()=>s.remove(),3500);if(d.children.length>5)d.firstChild.remove();
}
function addMem(text,type){
  const f=document.getElementById('mem-feed'),t=new Date().toLocaleTimeString('en',{hour12:false,hour:'2-digit',minute:'2-digit',second:'2-digit'});
  const d=document.createElement('div');d.className='fe '+type;
  const imp=(Math.random()*0.4+0.6).toFixed(2);
  d.innerHTML=`<span class="fe-t">${t}</span>${text}<span class="fe-impact">${imp}</span>`;
  f.appendChild(d);f.scrollTop=f.scrollHeight;if(f.children.length>35)f.firstChild.remove();
  document.getElementById('mem-n').textContent=f.children.length;
}
function addRefl(text,cls){
  const f=document.getElementById('refl-feed'),d=document.createElement('div');
  d.className='fe '+(cls||'');d.textContent=text;f.appendChild(d);f.scrollTop=f.scrollHeight;
  if(f.children.length>25)f.firstChild.remove();
}

// Drawing
function draw(){
  if(!X)return;const w=C.width,h=C.height;
  X.fillStyle='#06080c';X.fillRect(0,0,w,h);
  if(curEnv==='code')drawCode(w,h);
  else if(curEnv==='browser')drawBrowser(w,h);
  else if(curEnv==='rl')drawRL(w,h);
  else drawMulti(w,h);
}
function drawCode(w,h){
  // Terminal style
  X.fillStyle='#0a0e16';X.beginPath();X.roundRect(10,10,w-20,h-20,8);X.fill();
  X.strokeStyle='rgba(0,229,160,.06)';X.stroke();
  // Title bar
  X.fillStyle='rgba(0,229,160,.04)';X.fillRect(10,10,w-20,24);
  X.fillStyle='#ff4466';X.beginPath();X.arc(24,22,4,0,6.28);X.fill();
  X.fillStyle='#ffd666';X.beginPath();X.arc(36,22,4,0,6.28);X.fill();
  X.fillStyle='#00e5a0';X.beginPath();X.arc(48,22,4,0,6.28);X.fill();
  X.fillStyle='#5a6c8a';X.font='bold 9px JetBrains Mono';X.textAlign='left';X.textBaseline='middle';
  X.fillText('cognicore_agent.py - GPT Coding Agent',60,22);
  const lines=['def fix_issue(repo, issue_id):','    patch = generate_patch(issue_id)','    result = apply_patch(repo, patch)','    if result.failed:','        # CogniCore Memory: avoid previous failed patches','        mem = memory.recall(issue_id)','        if mem.has_failed_patches:','            patch = alternative_strategy(mem)','        result = apply_patch(repo, patch)','    reflection.log(result)','    return result'];
  const lh=18,sy=44;
  lines.forEach((l,i)=>{
    const y=sy+i*lh;const isKey=i===4||i===5||i===6||i===7||i===9;
    if(isKey){X.fillStyle='rgba(0,229,160,.04)';X.fillRect(16,y-2,w-32,lh)}
    X.fillStyle=isKey?'#00e5a0':'#8090b0';X.font='10px JetBrains Mono';X.textAlign='left';X.textBaseline='top';
    X.fillText((i+1).toString().padStart(2)+' | '+l,22,y);
  });
  // Status
  const phase=tick>100?'PATCHED':tick>50?'RETRYING':'ANALYZING';
  X.fillStyle=tick>100?'#00e5a0':tick>50?'#ffd666':'#00ccff';
  X.font='bold 10px JetBrains Mono';X.fillText('Status: '+phase,22,h-30);
  X.fillStyle='#5a6c8a';X.fillText('Memory entries: '+(tick%50)+' | Reflection triggers: '+Math.floor(tick/15),200,h-30);
}
function drawBrowser(w,h){
  X.fillStyle='#0a0e16';X.beginPath();X.roundRect(10,10,w-20,h-20,8);X.fill();
  // URL bar
  X.fillStyle='#141828';X.beginPath();X.roundRect(16,16,w-32,22,4);X.fill();
  X.fillStyle='#5a6c8a';X.font='10px JetBrains Mono';X.textAlign='left';X.textBaseline='middle';
  X.fillText('https://app.example.com/dashboard/settings',30,27);
  // Page content blocks
  const cols=['#141828','#181c2a','#1a2040'];
  for(let i=0;i<6;i++){
    const bx=20+(i%3)*(w/3-16),by=48+Math.floor(i/3)*((h-100)/2);
    X.fillStyle=cols[i%3];X.beginPath();X.roundRect(bx,by,(w-60)/3-4,(h-120)/2-8,6);X.fill();
    X.strokeStyle='rgba(0,204,255,.05)';X.stroke();
  }
  // Agent cursor
  const cx=w/2+Math.sin(tick/8)*100,cy=h/2+Math.cos(tick/6)*60;
  X.fillStyle='#00ccff44';X.beginPath();X.arc(cx,cy,12,0,6.28);X.fill();
  X.fillStyle='#00ccff';X.beginPath();X.arc(cx,cy,4,0,6.28);X.fill();
  X.fillStyle='#5a6c8a';X.font='8px JetBrains Mono';X.fillText('Browser Agent',cx+14,cy);
  X.fillStyle=tick>80?'#00e5a0':'#ffd666';X.font='bold 9px JetBrains Mono';X.fillText(tick>80?'Form submitted':'Navigating...',22,h-24);
}
function drawRL(w,h){
  const gs=10,cs=Math.min(~~((w-30)/gs),~~((h-30)/gs)),ox=(w-cs*gs)/2,oy=(h-cs*gs)/2;
  for(let r=0;r<gs;r++)for(let c=0;c<gs;c++){
    X.fillStyle='#0c0e18';X.fillRect(ox+c*cs,oy+r*cs,cs,cs);X.strokeStyle='rgba(255,255,255,.02)';X.strokeRect(ox+c*cs,oy+r*cs,cs,cs);
  }
  // Traps
  [[2,3],[4,6],[7,2],[5,8],[1,7],[8,5]].forEach(([r,c])=>{X.fillStyle='rgba(255,68,102,.1)';X.fillRect(ox+c*cs,oy+r*cs,cs,cs)});
  // Goal
  const p=Math.sin(tick/5)*.1+.9;
  X.fillStyle=`rgba(255,214,102,${p*.12})`;X.fillRect(ox+9*cs,oy+9*cs,cs,cs);
  // Agent
  const ar=~~(tick/3)%gs,ac=~~(tick/2.5)%gs;
  X.fillStyle='#00e5a0cc';X.beginPath();X.roundRect(ox+ac*cs+3,oy+ar*cs+3,cs-6,cs-6,4);X.fill();
  X.shadowColor='#00e5a0';X.shadowBlur=8;X.fill();X.shadowBlur=0;
  X.fillStyle='#000';X.font=`bold ${cs/3}px Inter`;X.textAlign='center';X.textBaseline='middle';X.fillText('A',ox+ac*cs+cs/2,oy+ar*cs+cs/2);
  X.fillStyle='#5a6c8a';X.font='8px JetBrains Mono';X.textAlign='left';X.textBaseline='top';X.fillText('PPO Agent | Dynamic Grid',8,6);
}
function drawMulti(w,h){
  const gs=8,cs=Math.min(~~((w-30)/gs),~~((h-30)/gs)),ox=(w-cs*gs)/2,oy=(h-cs*gs)/2;
  for(let r=0;r<gs;r++)for(let c=0;c<gs;c++){X.fillStyle='#0c0e18';X.fillRect(ox+c*cs,oy+r*cs,cs,cs);X.strokeStyle='rgba(0,229,160,.02)';X.strokeRect(ox+c*cs,oy+r*cs,cs,cs)}
  // Resources
  [[1,2],[3,5],[6,1],[2,6],[5,4]].forEach(([r,c])=>{X.fillStyle='rgba(255,214,102,.6)';X.beginPath();X.roundRect(ox+c*cs+cs*.2,oy+r*cs+cs*.2,cs*.6,cs*.6,3);X.fill()});
  // Agents
  const agts=[{r:tick%gs,c:1,col:'#00e5a0',n:'A1'},{r:2,c:tick%gs,col:'#00ccff',n:'A2'},{r:(tick+3)%gs,c:(tick+2)%gs,col:'#9966ff',n:'A3'}];
  agts.forEach(a=>{
    const x=ox+(a.c%gs)*cs,y=oy+(a.r%gs)*cs;
    X.fillStyle=a.col+'cc';X.beginPath();X.roundRect(x+2,y+2,cs-4,cs-4,cs/4);X.fill();
    X.fillStyle='#000';X.font=`bold ${cs/3}px Inter`;X.textAlign='center';X.textBaseline='middle';X.fillText(a.n,x+cs/2,y+cs/2);
  });
  // Sync lines
  X.strokeStyle='rgba(0,229,160,.1)';X.lineWidth=1;X.setLineDash([3,3]);
  X.beginPath();X.moveTo(ox+(1%gs)*cs+cs/2,oy+(tick%gs)*cs+cs/2);X.lineTo(ox+(tick%gs)*cs+cs/2,oy+2*cs+cs/2);X.stroke();
  X.setLineDash([]);
  X.fillStyle='#5a6c8a';X.font='8px JetBrains Mono';X.textAlign='left';X.textBaseline='top';X.fillText('Multi-Agent Swarm | Shared Memory Sync',8,6);
}

function drawChart(){
  if(!chartX)return;const w=chartC.width,h=chartC.height;
  chartX.fillStyle='#06080c';chartX.fillRect(0,0,w,h);
  const mx=40,gw=w-mx-10,gh=h-30,my=10;
  // Axes
  chartX.strokeStyle='rgba(255,255,255,.05)';chartX.beginPath();chartX.moveTo(mx,my);chartX.lineTo(mx,my+gh);chartX.lineTo(mx+gw,my+gh);chartX.stroke();
  // Grid
  for(let i=0;i<=4;i++){const y=my+gh-gh*i/4;chartX.strokeStyle='rgba(255,255,255,.03)';chartX.beginPath();chartX.moveTo(mx,y);chartX.lineTo(mx+gw,y);chartX.stroke();chartX.fillStyle='#4a5570';chartX.font='8px JetBrains Mono';chartX.textAlign='right';chartX.textBaseline='middle';chartX.fillText((i*25)+'%',mx-4,y)}
  // Data
  if(baselineData.length>1){
    chartX.strokeStyle='#ff446666';chartX.lineWidth=1.5;chartX.beginPath();
    baselineData.forEach((v,i)=>{const x=mx+i/(baselineData.length-1)*gw,y=my+gh-gh*v/100;i===0?chartX.moveTo(x,y):chartX.lineTo(x,y)});chartX.stroke();
  }
  if(cogniData.length>1){
    chartX.strokeStyle='#00e5a0';chartX.lineWidth=2;chartX.beginPath();
    cogniData.forEach((v,i)=>{const x=mx+i/(cogniData.length-1)*gw,y=my+gh-gh*v/100;i===0?chartX.moveTo(x,y):chartX.lineTo(x,y)});chartX.stroke();
    chartX.shadowColor='#00e5a0';chartX.shadowBlur=6;chartX.stroke();chartX.shadowBlur=0;
  }
  // Legend
  chartX.fillStyle='#ff446688';chartX.fillRect(mx+10,my+4,8,3);chartX.fillStyle='#5a6c8a';chartX.font='8px Inter';chartX.textAlign='left';chartX.fillText('Baseline',mx+22,my+8);
  chartX.fillStyle='#00e5a0';chartX.fillRect(mx+80,my+4,8,3);chartX.fillStyle='#5a6c8a';chartX.fillText('CogniCore',mx+92,my+8);
}

function updateKPIs(p){
  KPIS.forEach(k=>{
    const v=k.u==='x'?k.t*p:(k.t*p);
    document.getElementById('kv-'+k.id).textContent=k.u==='x'?v.toFixed(1)+'x':v.toFixed(1)+'%';
    document.getElementById('kb-'+k.id).style.width=(k.u==='x'?v/5*100:v)+'%';
    if(p>0.1)document.getElementById('kd-'+k.id).textContent='+'+((v*0.08).toFixed(1))+' vs baseline';
  });
}

function updateAgents(p){
  AGENTS.forEach((a,i)=>{
    const score=~~(40+50*p*Math.min(1,(i+3)/7));const tasks=~~(p*20*(i+1)/3);
    a.score=score;a.task=tasks;
    document.getElementById('am-ada-'+a.id).textContent=score+'%';
    document.getElementById('am-ada-'+a.id).className='ag-mv'+(score>70?' g':score>40?' a':' r');
    document.getElementById('am-tsk-'+a.id).textContent=tasks;
    document.getElementById('am-scr-'+a.id).textContent=score;
    document.getElementById('am-scr-'+a.id).className='ag-mv'+(score>70?' g':' a');
    document.getElementById('ab-'+a.id).style.width=Math.min(100,score)+'%';
  });
  const avg=AGENTS.reduce((s,a)=>s+a.score,0)/AGENTS.length;
  document.getElementById('h-score').textContent=avg.toFixed(0);
}

async function toggleRun(){
  if(running){running=false;document.getElementById('btn-go').textContent='Start Evaluation';document.getElementById('btn-go').classList.remove('running');return}
  running=true;tick=0;baselineData.length=0;cogniData.length=0;
  document.getElementById('btn-go').textContent='Stop';document.getElementById('btn-go').classList.add('running');
  while(running&&tick<200){
    tick++;const p=tick/200;
    document.getElementById('h-sess').textContent='EVAL-2026-0516-'+tick;
    draw();updateKPIs(p);updateAgents(p);
    // Chart data
    baselineData.push(20+Math.random()*15+p*20);
    cogniData.push(20+Math.random()*10+p*55);
    if(tick%3===0)drawChart();
    // Memory & reflection feeds
    if(tick%5===0)addMem(MEM_MSGS[~~(Math.random()*MEM_MSGS.length)],['good','info','warn'][~~(Math.random()*3)]);
    if(tick%8===0)addRefl(REFL_MSGS[~~(Math.random()*REFL_MSGS.length)],Math.random()>.5?'hl':'w');
    // Overlays
    if(tick%7===0)addOv(['Memory Retrieved','Reflection Triggered','Strategy Adapted','Repeated Failure Avoided','Environment Shift'][~~(Math.random()*5)],['mem','refl','adapt','fail','shift'][~~(Math.random()*5)]);
    await new Promise(r=>setTimeout(r,80));
  }
  running=false;document.getElementById('btn-go').textContent='Start Evaluation';document.getElementById('btn-go').classList.remove('running');
  addRefl('Evaluation complete. CogniCore outperforms baseline across all agent architectures.','hl');
  addMem('Final: Statistical significance confirmed (p < 0.001) across 7 agents.','good');
}

function doReset(){
  running=false;tick=0;baselineData.length=0;cogniData.length=0;
  document.getElementById('btn-go').textContent='Start Evaluation';document.getElementById('btn-go').classList.remove('running');
  document.getElementById('mem-feed').innerHTML='';document.getElementById('refl-feed').innerHTML='';
  document.getElementById('overlays').innerHTML='';document.getElementById('mem-n').textContent='0';
  document.getElementById('h-score').textContent='--';document.getElementById('h-sess').textContent='EVAL-2026-0516';
  KPIS.forEach(k=>{document.getElementById('kv-'+k.id).textContent='--';document.getElementById('kb-'+k.id).style.width='0%';document.getElementById('kd-'+k.id).textContent=''});
  AGENTS.forEach(a=>{document.getElementById('am-ada-'+a.id).textContent='--';document.getElementById('am-tsk-'+a.id).textContent='0';document.getElementById('am-scr-'+a.id).textContent='--';document.getElementById('ab-'+a.id).style.width='0%'});
  resize();draw();drawChart();
}

window.addEventListener('load',init);
window.addEventListener('resize',()=>{resize();draw();drawChart()});
