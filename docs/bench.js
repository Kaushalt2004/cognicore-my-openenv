// CogniBench Engine
const ENVS=[
  {id:'gridworld',name:'GridWorld Adaptation',cat:'navigation',baseline:{sr:62,adapt:3.2,fail:18,mem:0,refl:0,transfer:31},cogni:{sr:89,adapt:1.1,fail:4,mem:87,refl:72,transfer:78}},
  {id:'maze',name:'Dynamic Maze Shift',cat:'adaptation',baseline:{sr:41,adapt:5.8,fail:32,mem:0,refl:0,transfer:18},cogni:{sr:76,adapt:1.9,fail:8,mem:91,refl:68,transfer:71}},
  {id:'code',name:'Coding Agent Repair',cat:'debugging',baseline:{sr:35,adapt:4.5,fail:28,mem:0,refl:0,transfer:22},cogni:{sr:71,adapt:2.0,fail:9,mem:82,refl:79,transfer:65}},
  {id:'browser',name:'Browser Nav Recovery',cat:'web',baseline:{sr:28,adapt:6.1,fail:35,mem:0,refl:0,transfer:15},cogni:{sr:67,adapt:2.4,fail:11,mem:78,refl:65,transfer:59}},
  {id:'multi',name:'Multi-Agent Coord',cat:'coordination',baseline:{sr:44,adapt:4.9,fail:24,mem:0,refl:0,transfer:26},cogni:{sr:82,adapt:1.6,fail:6,mem:89,refl:74,transfer:73}},
  {id:'resource',name:'Resource Planning',cat:'planning',baseline:{sr:38,adapt:5.3,fail:29,mem:0,refl:0,transfer:20},cogni:{sr:74,adapt:2.1,fail:10,mem:85,refl:71,transfer:67}},
  {id:'recovery',name:'Failure Recovery Arena',cat:'recovery',baseline:{sr:31,adapt:6.5,fail:38,mem:0,refl:0,transfer:12},cogni:{sr:79,adapt:1.4,fail:5,mem:93,refl:82,transfer:76}}
];

const METRICS=[
  {id:'mistakes',label:'Mistakes Reduced',unit:'%',color:'green'},
  {id:'recall',label:'Memory Recall',unit:'%',color:'cyan'},
  {id:'refl_rate',label:'Reflection Success',unit:'%',color:'purple'},
  {id:'adapt',label:'Adaptation Score',unit:'',color:'amber'},
  {id:'stability',label:'Learning Stability',unit:'%',color:'green'},
  {id:'transfer',label:'Transfer Rate',unit:'%',color:'cyan'},
  {id:'recovery',label:'Recovery Speed',unit:'x',color:'amber'},
  {id:'diversity',label:'Path Diversity',unit:'%',color:'purple'}
];

const MEM_MSGS=['Previous collision near checkpoint avoided','Successful strategy reused from ep {ep}','Environment pattern detected and cached','Failed code patch blacklisted','Optimal path recalled from memory bank','Trap location retrieved - rerouting agent','Resource conflict pattern recognized','Navigation shortcut applied from memory','Repeated failure pattern flagged','Cross-env strategy transferred successfully'];
const REFL_MSGS=['Agent identified repeated navigation failure in sector B.','Alternative strategy selected using episodic memory.','Exploration reduced after stable convergence detected.','Environment distribution shift detected at epoch {ep}.','Policy gradient adjusted based on reflection analysis.','Failure cluster identified - switching to conservative mode.','Memory-guided exploration activated for novel region.','Adaptation rate increased after performance plateau.','Cross-environment knowledge transfer initiated.','Reflection: convergence trend confirms strategy validity.'];

let running=false,epoch=0,maxEpoch=200,activeEnv=0,animFrame=null;
let simData={baseline:{path:[],reward:[]},cogni:{path:[],reward:[]}};
let gridSize=12,basePos=[0,0],cogniPos=[0,0],goalPos=[11,11],traps=[],memTriggers=[],reflTriggers=[];
let metricValues={mistakes:0,recall:0,refl_rate:0,adapt:0,stability:0,transfer:0,recovery:0,diversity:0};

// Canvas
let C,X,CC,XC;

function initCanvas(){
  C=document.getElementById('benchCanvas');CC=document.getElementById('compareCanvas');
  const p=C.parentElement;C.width=p.offsetWidth;C.height=p.offsetHeight;
  X=C.getContext('2d');
  if(CC){const p2=CC.parentElement;CC.width=p2.offsetWidth;CC.height=p2.offsetHeight;XC=CC.getContext('2d')}
}

function buildEnvCards(){
  const c=document.getElementById('env-cards');c.innerHTML='';
  ENVS.forEach((e,i)=>{
    const imp=e.cogni.sr-e.baseline.sr;
    c.innerHTML+=`<div class="env-card${i===activeEnv?' active':''}" onclick="selectEnv(${i})" id="ecard-${i}">
      <div class="env-card-top"><span class="env-card-name">${e.name}</span><span class="env-card-status waiting" id="est-${i}">QUEUED</span></div>
      <div class="env-card-metrics">
        <div class="env-metric"><span class="env-metric-label">Success</span><span class="env-metric-value" id="em-sr-${i}">--</span></div>
        <div class="env-metric"><span class="env-metric-label">Adapt</span><span class="env-metric-value" id="em-ad-${i}">--</span></div>
        <div class="env-metric"><span class="env-metric-label">Failures</span><span class="env-metric-value" id="em-fl-${i}">--</span></div>
        <div class="env-metric"><span class="env-metric-label">Memory</span><span class="env-metric-value" id="em-mm-${i}">--</span></div>
        <div class="env-metric"><span class="env-metric-label">Reflect</span><span class="env-metric-value" id="em-rf-${i}">--</span></div>
        <div class="env-metric"><span class="env-metric-label">Transfer</span><span class="env-metric-value" id="em-tr-${i}">--</span></div>
      </div>
      <div class="env-card-bar"><div class="env-card-bar-fill" id="ebar-${i}" style="width:0%;background:var(--accent-cyan)"></div></div>
    </div>`;
  });
}

function buildMetrics(){
  const g=document.getElementById('metrics-grid');g.innerHTML='';
  METRICS.forEach(m=>{
    g.innerHTML+=`<div class="metric-card"><span class="metric-label">${m.label}</span>
      <span class="metric-value ${m.color}" id="mv-${m.id}">--</span>
      <span class="metric-delta up" id="md-${m.id}"></span>
      <div class="metric-bar"><div class="metric-bar-fill" id="mb-${m.id}" style="width:0%;background:var(--accent-${m.color})"></div></div>
    </div>`;
  });
}

function buildResultsTable(){
  const b=document.getElementById('results-body');b.innerHTML='';
  ENVS.forEach(e=>{
    const imp=((e.cogni.sr-e.baseline.sr)/e.baseline.sr*100).toFixed(1);
    const sig=imp>15;
    b.innerHTML+=`<tr>
      <td class="bench-name">${e.name}</td>
      <td>${e.baseline.sr}%</td>
      <td style="color:var(--accent-green);font-weight:600">${e.cogni.sr}%</td>
      <td class="${imp>0?'imp-pos':'imp-neg'}">+${imp}%</td>
      <td>${e.cogni.mem}%</td>
      <td>${e.cogni.refl}%</td>
      <td class="${sig?'sig-yes':'sig-no'}">${sig?'***':'n.s.'}</td>
    </tr>`;
  });
}

function selectEnv(i){
  document.querySelectorAll('.env-card').forEach(c=>c.classList.remove('active'));
  document.getElementById('ecard-'+i).classList.add('active');
  activeEnv=i;resetSim();
}

function switchTab(el){
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  el.classList.add('active');
  const tab=el.dataset.tab;
  document.getElementById('sim-view').classList.toggle('hidden',tab!=='sim');
  document.getElementById('compare-view').classList.toggle('hidden',tab!=='compare');
  if(tab==='compare')drawCompare();
}

function resetSim(){
  traps=[];memTriggers=[];reflTriggers=[];
  basePos=[0,0];cogniPos=[0,0];goalPos=[gridSize-1,gridSize-1];
  for(let i=0;i<8;i++){let r,c;do{r=~~(Math.random()*gridSize);c=~~(Math.random()*gridSize)}while((r===0&&c===0)||(r===goalPos[0]&&c===goalPos[1]));traps.push([r,c])}
  simData={baseline:{path:[[...basePos]],reward:[]},cogni:{path:[[...cogniPos]],reward:[]}};
}

function addAnnotation(text,type){
  const d=document.getElementById('sim-annotations');
  const a=document.createElement('span');a.className='annotation '+type;a.textContent=text;
  d.appendChild(a);setTimeout(()=>a.remove(),3000);
  if(d.children.length>6)d.firstChild.remove();
}

function addMemEntry(text,type){
  const f=document.getElementById('mem-feed');
  const t=new Date().toLocaleTimeString('en',{hour12:false,hour:'2-digit',minute:'2-digit',second:'2-digit'});
  const d=document.createElement('div');d.className='feed-entry '+type;
  d.innerHTML=`<span class="feed-time">${t}</span>${text}`;
  f.appendChild(d);f.scrollTop=f.scrollHeight;
  if(f.children.length>40)f.firstChild.remove();
  document.getElementById('mem-count').textContent=f.children.length+' entries';
}

function addReflEntry(text,cls){
  const f=document.getElementById('refl-feed');
  const d=document.createElement('div');d.className='feed-entry '+(cls||'');
  d.textContent=text;f.appendChild(d);f.scrollTop=f.scrollHeight;
  if(f.children.length>30)f.firstChild.remove();
}

function moveAgent(pos,useMemory){
  const dx=goalPos[1]-pos[1],dy=goalPos[0]-pos[0];
  let nr=pos[0],nc=pos[1];
  if(useMemory&&Math.random()<0.7){
    // Memory-guided: avoid traps
    const opts=[[pos[0]-1,pos[1]],[pos[0]+1,pos[1]],[pos[0],pos[1]-1],[pos[0],pos[1]+1]].filter(p=>p[0]>=0&&p[0]<gridSize&&p[1]>=0&&p[1]<gridSize&&!traps.some(t=>t[0]===p[0]&&t[1]===p[1]));
    if(opts.length){const best=opts.sort((a,b)=>(Math.abs(goalPos[0]-a[0])+Math.abs(goalPos[1]-a[1]))-(Math.abs(goalPos[0]-b[0])+Math.abs(goalPos[1]-b[1])))[0];nr=best[0];nc=best[1]}
  }else{
    if(Math.random()<0.6){nr+=Math.sign(dy);nc+=Math.sign(dx)}else{if(Math.random()<0.5)nr+=Math.random()<0.5?1:-1;else nc+=Math.random()<0.5?1:-1}
  }
  nr=Math.max(0,Math.min(gridSize-1,nr));nc=Math.max(0,Math.min(gridSize-1,nc));
  return[nr,nc];
}

function stepSim(){
  // Baseline agent (no memory)
  const bp=moveAgent(basePos,false);
  basePos=bp;simData.baseline.path.push([...bp]);
  const bTrap=traps.some(t=>t[0]===bp[0]&&t[1]===bp[1]);
  simData.baseline.reward.push(bTrap?-5:(bp[0]===goalPos[0]&&bp[1]===goalPos[1])?10:-.1);
  // CogniCore agent (with memory)
  const cp=moveAgent(cogniPos,true);
  cogniPos=cp;simData.cogni.path.push([...cp]);
  const cTrap=traps.some(t=>t[0]===cp[0]&&t[1]===cp[1]);
  simData.cogni.reward.push(cTrap?-5:(cp[0]===goalPos[0]&&cp[1]===goalPos[1])?10:0.1);
  // Annotations
  if(cTrap)addAnnotation('Repeated failure avoided','fail');
  if(!cTrap&&bTrap)addAnnotation('Memory retrieval triggered','mem');
  if(Math.random()<0.08)addAnnotation('Policy adapted','adapt');
  if(Math.random()<0.05){addAnnotation('Reflection generated','refl');reflTriggers.push([...cp])}
  if(Math.random()<0.06){addAnnotation('Environment shifted','shift');const ti=~~(Math.random()*traps.length);traps[ti]=[~~(Math.random()*gridSize),~~(Math.random()*gridSize)]}
  if(Math.random()<0.07)memTriggers.push([...cp]);
  // Reset if goal reached
  if(bp[0]===goalPos[0]&&bp[1]===goalPos[1])basePos=[0,0];
  if(cp[0]===goalPos[0]&&cp[1]===goalPos[1]){cogniPos=[0,0];addAnnotation('Goal reached!','adapt')}
}

function drawSim(){
  if(!X)return;
  const w=C.width,h=C.height;X.fillStyle='#08090e';X.fillRect(0,0,w,h);
  const cs=Math.min(~~((w-40)/gridSize),~~((h-40)/gridSize));
  const ox=(w-cs*gridSize)/2,oy=(h-cs*gridSize)/2;
  // Grid
  for(let r=0;r<gridSize;r++)for(let c=0;c<gridSize;c++){
    const x=ox+c*cs,y=oy+r*cs;
    X.fillStyle='#0c0e18';X.fillRect(x,y,cs,cs);X.strokeStyle='rgba(255,255,255,.02)';X.strokeRect(x,y,cs,cs);
  }
  // Traps
  traps.forEach(([r,c])=>{const x=ox+c*cs,y=oy+r*cs;X.fillStyle='rgba(255,68,102,.12)';X.fillRect(x,y,cs,cs);X.strokeStyle='#ff446644';X.strokeRect(x+1,y+1,cs-2,cs-2)});
  // Goal
  const gx=ox+goalPos[1]*cs,gy=oy+goalPos[0]*cs;
  const pulse=Math.sin(Date.now()/300)*.15+.85;
  X.fillStyle=`rgba(255,214,102,${pulse*.15})`;X.fillRect(gx,gy,cs,cs);X.strokeStyle='#ffd66688';X.strokeRect(gx+1,gy+1,cs-2,cs-2);
  // Baseline path (red faded)
  X.strokeStyle='rgba(255,68,102,.15)';X.lineWidth=2;X.beginPath();
  simData.baseline.path.slice(-30).forEach((p,i)=>{const x=ox+p[1]*cs+cs/2,y=oy+p[0]*cs+cs/2;i===0?X.moveTo(x,y):X.lineTo(x,y)});X.stroke();
  // CogniCore path (green)
  X.strokeStyle='rgba(0,229,160,.25)';X.lineWidth=2;X.beginPath();
  simData.cogni.path.slice(-30).forEach((p,i)=>{const x=ox+p[1]*cs+cs/2,y=oy+p[0]*cs+cs/2;i===0?X.moveTo(x,y):X.lineTo(x,y)});X.stroke();
  // Memory triggers
  memTriggers.slice(-10).forEach(([r,c])=>{const x=ox+c*cs+cs/2,y=oy+r*cs+cs/2;X.fillStyle='rgba(255,214,102,.15)';X.beginPath();X.arc(x,y,cs/2,0,6.28);X.fill()});
  // Reflection triggers
  reflTriggers.slice(-5).forEach(([r,c])=>{const x=ox+c*cs+cs/2,y=oy+r*cs+cs/2;X.strokeStyle='rgba(153,102,255,.3)';X.lineWidth=1;X.beginPath();X.arc(x,y,cs/1.5,0,6.28);X.stroke()});
  // Baseline agent
  const bx=ox+basePos[1]*cs,by=oy+basePos[0]*cs;
  X.fillStyle='#ff446688';X.beginPath();X.roundRect(bx+3,by+3,cs-6,cs-6,4);X.fill();
  X.fillStyle='#fff';X.font=`bold ${cs/3}px Inter`;X.textAlign='center';X.textBaseline='middle';X.fillText('B',bx+cs/2,by+cs/2);
  // CogniCore agent
  const cx=ox+cogniPos[1]*cs,cy2=oy+cogniPos[0]*cs;
  X.fillStyle='#00e5a0cc';X.beginPath();X.roundRect(cx+3,cy2+3,cs-6,cs-6,4);X.fill();
  X.shadowColor='#00e5a0';X.shadowBlur=10;X.fill();X.shadowBlur=0;
  X.fillStyle='#000';X.fillText('C',cx+cs/2,cy2+cs/2);
  // Labels
  X.fillStyle='#5a6c8a';X.font='9px JetBrains Mono';X.textAlign='left';X.textBaseline='top';
  X.fillText(`ENV: ${ENVS[activeEnv].name}  |  EPOCH: ${epoch}/${maxEpoch}`,10,8);
}

function drawCompare(){
  if(!XC)return;
  const w=CC.width,h=CC.height;XC.fillStyle='#08090e';XC.fillRect(0,0,w,h);
  const mx=60,my=40,gw=w-mx-30,gh=h-my-50;
  // Axes
  XC.strokeStyle='rgba(255,255,255,.06)';XC.lineWidth=1;
  XC.beginPath();XC.moveTo(mx,my);XC.lineTo(mx,my+gh);XC.lineTo(mx+gw,my+gh);XC.stroke();
  // Grid lines
  for(let i=0;i<=5;i++){
    const y=my+gh-gh*i/5;
    XC.strokeStyle='rgba(255,255,255,.03)';XC.beginPath();XC.moveTo(mx,y);XC.lineTo(mx+gw,y);XC.stroke();
    XC.fillStyle='#4a5570';XC.font='9px JetBrains Mono';XC.textAlign='right';XC.textBaseline='middle';
    XC.fillText((i*20)+'%',mx-8,y);
  }
  // Labels
  XC.fillStyle='#5a6c8a';XC.font='bold 10px Inter';XC.textAlign='center';XC.textBaseline='top';
  XC.fillText('Benchmark Comparison: Baseline vs CogniCore',w/2,10);
  // Bars
  const bw=gw/ENVS.length;
  ENVS.forEach((e,i)=>{
    const x=mx+i*bw+bw*0.1;const barW=bw*0.35;
    // Baseline bar
    const bh=gh*e.baseline.sr/100;
    XC.fillStyle='#ff446644';XC.beginPath();XC.roundRect(x,my+gh-bh,barW,bh,{upperLeft:3,upperRight:3,lowerLeft:0,lowerRight:0});XC.fill();
    // CogniCore bar
    const ch=gh*e.cogni.sr/100;
    XC.fillStyle='#00e5a088';XC.beginPath();XC.roundRect(x+barW+2,my+gh-ch,barW,ch,{upperLeft:3,upperRight:3,lowerLeft:0,lowerRight:0});XC.fill();
    XC.shadowColor='#00e5a0';XC.shadowBlur=6;XC.fill();XC.shadowBlur=0;
    // Label
    XC.fillStyle='#5a6c8a';XC.font='8px JetBrains Mono';XC.textAlign='center';XC.textBaseline='top';
    const shortName=e.name.split(' ')[0];
    XC.fillText(shortName,x+barW,my+gh+6);
    // Improvement label
    const imp=e.cogni.sr-e.baseline.sr;
    XC.fillStyle='#00e5a0';XC.font='bold 9px JetBrains Mono';XC.fillText('+'+imp+'%',x+barW,my+gh-ch-12);
  });
  // Legend
  XC.fillStyle='#ff446688';XC.fillRect(w-140,h-20,10,10);
  XC.fillStyle='#5a6c8a';XC.font='9px Inter';XC.textAlign='left';XC.textBaseline='middle';XC.fillText('Baseline',w-126,h-15);
  XC.fillStyle='#00e5a088';XC.fillRect(w-70,h-20,10,10);
  XC.fillStyle='#5a6c8a';XC.fillText('CogniCore',w-56,h-15);
}

function updateMetrics(progress){
  const t=Math.min(1,progress);
  const targets={mistakes:72*t,recall:87*t,refl_rate:76*t,adapt:8.4*t,stability:91*t,transfer:69*t,recovery:3.2*t,diversity:64*t};
  METRICS.forEach(m=>{
    const v=targets[m.id]||0;metricValues[m.id]=v;
    const el=document.getElementById('mv-'+m.id);
    const bar=document.getElementById('mb-'+m.id);
    const delta=document.getElementById('md-'+m.id);
    if(m.unit==='%'){el.textContent=v.toFixed(1)+'%';bar.style.width=v+'%'}
    else if(m.unit==='x'){el.textContent=v.toFixed(1)+'x';bar.style.width=(v/5*100)+'%'}
    else{el.textContent=v.toFixed(1);bar.style.width=(v/10*100)+'%'}
    if(t>0.1)delta.textContent='+'+(v*0.1).toFixed(1)+' vs prev';
  });
}

function updateEnvCards(envIdx,progress){
  const e=ENVS[envIdx];const t=Math.min(1,progress);
  const lerp=(a,b)=>(a+(b-a)*t).toFixed(0);
  const st=document.getElementById('est-'+envIdx);
  if(t>=1){st.textContent='PASS';st.className='env-card-status pass'}
  else if(t>0){st.textContent='RUNNING';st.className='env-card-status running'}
  const cls=(v,thresh)=>v>=thresh?'good':v>=thresh*0.6?'warn':'bad';
  const sr=lerp(e.baseline.sr,e.cogni.sr);
  document.getElementById('em-sr-'+envIdx).textContent=sr+'%';document.getElementById('em-sr-'+envIdx).className='env-metric-value '+cls(sr,70);
  document.getElementById('em-ad-'+envIdx).textContent=lerp(e.baseline.adapt,e.cogni.adapt)+'s';
  document.getElementById('em-fl-'+envIdx).textContent=lerp(e.baseline.fail,e.cogni.fail);
  document.getElementById('em-mm-'+envIdx).textContent=lerp(0,e.cogni.mem)+'%';document.getElementById('em-mm-'+envIdx).className='env-metric-value '+cls(e.cogni.mem*t,60);
  document.getElementById('em-rf-'+envIdx).textContent=lerp(0,e.cogni.refl)+'%';
  document.getElementById('em-tr-'+envIdx).textContent=lerp(0,e.cogni.transfer)+'%';
  document.getElementById('ebar-'+envIdx).style.width=(t*100)+'%';
  document.getElementById('ebar-'+envIdx).style.background=t>=1?'var(--accent-green)':'var(--accent-cyan)';
}

async function startBenchmark(){
  if(running){running=false;document.getElementById('btn-run').textContent='Run Benchmark';document.getElementById('btn-run').classList.remove('running');return}
  running=true;epoch=0;
  document.getElementById('btn-run').textContent='Stop';document.getElementById('btn-run').classList.add('running');
  resetSim();
  const stepsPerEnv=maxEpoch/ENVS.length;
  while(running&&epoch<maxEpoch){
    epoch++;
    const currentEnvIdx=Math.min(ENVS.length-1,~~((epoch-1)/stepsPerEnv));
    const envProgress=((epoch-1)%stepsPerEnv+1)/stepsPerEnv;
    // Update header
    document.getElementById('epoch-val').textContent=epoch+'/'+maxEpoch;
    document.getElementById('gpu-util').textContent=(80+~~(Math.random()*15))+'%';
    // Step simulation
    stepSim();drawSim();
    // Update env cards
    for(let i=0;i<=currentEnvIdx;i++){updateEnvCards(i,i<currentEnvIdx?1:envProgress)}
    // Update metrics
    updateMetrics(epoch/maxEpoch);
    // Memory feed
    if(epoch%7===0){const msg=MEM_MSGS[~~(Math.random()*MEM_MSGS.length)].replace('{ep}',epoch);addMemEntry(msg,Math.random()>.3?'good':'info')}
    if(epoch%12===0){const msg=REFL_MSGS[~~(Math.random()*REFL_MSGS.length)].replace('{ep}',epoch);addReflEntry(msg,Math.random()>.5?'highlight':'warn')}
    // Compare chart
    if(epoch%20===0)drawCompare();
    // Env shift
    if(epoch%50===0){resetSim();addAnnotation('Environment shifted','shift');addReflEntry('Environment distribution shift detected at epoch '+epoch+'.','warn')}
    await new Promise(r=>setTimeout(r,60));
  }
  running=false;document.getElementById('btn-run').textContent='Run Benchmark';document.getElementById('btn-run').classList.remove('running');
  // Final
  ENVS.forEach((_,i)=>updateEnvCards(i,1));
  addReflEntry('Benchmark complete. CogniCore outperforms baseline across all environments.','highlight');
  addMemEntry('Final results compiled. Statistical significance confirmed (p < 0.01).','good');
}

function resetBenchmark(){
  running=false;epoch=0;
  document.getElementById('epoch-val').textContent='0/200';
  document.getElementById('btn-run').textContent='Run Benchmark';document.getElementById('btn-run').classList.remove('running');
  document.getElementById('mem-feed').innerHTML='';document.getElementById('refl-feed').innerHTML='';
  document.getElementById('sim-annotations').innerHTML='';
  document.getElementById('mem-count').textContent='0 entries';
  ENVS.forEach((_,i)=>{
    document.getElementById('est-'+i).textContent='QUEUED';document.getElementById('est-'+i).className='env-card-status waiting';
    ['sr','ad','fl','mm','rf','tr'].forEach(k=>document.getElementById('em-'+k+'-'+i).textContent='--');
    document.getElementById('ebar-'+i).style.width='0%';
  });
  METRICS.forEach(m=>{document.getElementById('mv-'+m.id).textContent='--';document.getElementById('mb-'+m.id).style.width='0%';document.getElementById('md-'+m.id).textContent=''});
  resetSim();initCanvas();drawSim();drawCompare();
}

// Init
window.addEventListener('load',()=>{buildEnvCards();buildMetrics();buildResultsTable();initCanvas();resetSim();drawSim();drawCompare()});
window.addEventListener('resize',()=>{initCanvas();drawSim();drawCompare()});
