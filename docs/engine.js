// ═══ PARTICLES ═══
const pc=document.getElementById('particles'),px=pc.getContext('2d');
pc.width=innerWidth;pc.height=innerHeight;
const pts=Array.from({length:30},()=>({x:Math.random()*pc.width,y:Math.random()*pc.height,vx:(Math.random()-.5)*.12,vy:(Math.random()-.5)*.12,r:Math.random()+.4}));
function drawP(){px.clearRect(0,0,pc.width,pc.height);pts.forEach(p=>{p.x+=p.vx;p.y+=p.vy;if(p.x<0)p.x=pc.width;if(p.x>pc.width)p.x=0;if(p.y<0)p.y=pc.height;if(p.y>pc.height)p.y=0;px.beginPath();px.arc(p.x,p.y,p.r,0,6.28);px.fillStyle='rgba(0,229,160,.06)';px.fill()});pts.forEach((a,i)=>pts.slice(i+1).forEach(b=>{const d=Math.hypot(a.x-b.x,a.y-b.y);if(d<80){px.beginPath();px.moveTo(a.x,a.y);px.lineTo(b.x,b.y);px.strokeStyle=`rgba(0,229,160,${.02*(1-d/80)})`;px.stroke()}}));requestAnimationFrame(drawP)}
drawP();

// ═══ CANVASES ═══
const CE=document.getElementById('c-env'),XE=CE.getContext('2d');
const CC=document.getElementById('c-chart'),XC=CC.getContext('2d');
function resize(){const d=devicePixelRatio||1;[CE,CC].forEach(c=>{const r=c.parentElement.getBoundingClientRect();c.width=r.width*d;c.height=r.height*d;c.getContext('2d').scale(d,d)})}
resize();window.addEventListener('resize',()=>{pc.width=innerWidth;pc.height=innerHeight;resize()});

// ═══ ENVIRONMENT ═══
let SZ=10,TRAPS=12;
class GridWorld{
  constructor(sz){
    this.sz=sz;this.walls=new Set();this.traps=new Set();
    for(let r=0;r<sz;r++)for(let c=0;c<sz;c++)if(r===0||r===sz-1||c===0||c===sz-1)this.walls.add(`${r},${c}`);
    for(let i=0;i<sz*1.5;i++){let r=2+~~(Math.random()*(sz-4)),c=2+~~(Math.random()*(sz-4));if(!(r===1&&c===1)&&!(r===sz-2&&c===sz-2))this.walls.add(`${r},${c}`)}
    for(let i=0;i<TRAPS;i++){let r,c;do{r=1+~~(Math.random()*(sz-2));c=1+~~(Math.random()*(sz-2))}while((r===1&&c===1)||(r===sz-2&&c===sz-2)||this.walls.has(`${r},${c}`)||this.traps.has(`${r},${c}`));this.traps.add(`${r},${c}`)}
    this.reset();
  }
  reset(){this.agent=[1,1];this.goal=[this.sz-2,this.sz-2];this.steps=0;this.reward=0;this.visits={};this.path=[[1,1]];this.visits['1,1']=1;return this.state()}
  state(){return`${this.agent[0]},${this.agent[1]}`}
  step(a){
    const dirs=[[-1,0],[1,0],[0,-1],[0,1]];const[dr,dc]=dirs[a];
    const nr=this.agent[0]+dr,nc=this.agent[1]+dc,k=`${nr},${nc}`;this.steps++;
    if(this.walls.has(k)||nr<0||nr>=this.sz||nc<0||nc>=this.sz)return{s:this.state(),r:-0.3,done:false,event:'wall'};
    const oldD=Math.abs(this.goal[0]-this.agent[0])+Math.abs(this.goal[1]-this.agent[1]);
    this.agent=[nr,nc];this.visits[k]=(this.visits[k]||0)+1;this.path.push([nr,nc]);
    const newD=Math.abs(this.goal[0]-nr)+Math.abs(this.goal[1]-nc);
    if(this.traps.has(k))return{s:this.state(),r:-5,done:true,event:'trap'};
    if(nr===this.goal[0]&&nc===this.goal[1])return{s:this.state(),r:20,done:true,event:'goal'};
    let r=newD<oldD?0.1:-0.15;if((this.visits[k]||0)>2)r-=0.1;this.reward+=r;
    return{s:this.state(),r,done:this.steps>this.sz*this.sz*2,event:null};
  }
}

// ═══ Q-LEARNING ═══
class QAgent{
  constructor(){this.q={};this.lr=0.2;this.gamma=0.95;this.eps=1;this.decay=0.997;this.minEps=0.02}
  key(s,a){return s+'_'+a}
  getQ(s,a){return this.q[this.key(s,a)]||0}
  setQ(s,a,v){this.q[this.key(s,a)]=v}
  act(s,memory){
    if(Math.random()<this.eps)return~~(Math.random()*4);
    let best=-1e9,ba=0;
    for(let a=0;a<4;a++){let v=this.getQ(s,a);if(memory){const mk=s+'_'+a;if(memoryBank[mk])v+=memoryBank[mk].bonus}if(v>best){best=v;ba=a}}
    return ba;
  }
  learn(s,a,r,s2,done){
    let mx=done?0:-1e9;if(!done)for(let a2=0;a2<4;a2++)mx=Math.max(mx,this.getQ(s2,a2));
    this.setQ(s,a,this.getQ(s,a)+this.lr*(r+this.gamma*mx-this.getQ(s,a)));
  }
  decay_eps(){this.eps=Math.max(this.minEps,this.eps*this.decay)}
  qSize(){return Object.keys(this.q).length}
}

// ═══ MEMORY SYSTEM ═══
let memoryBank={},memHits=0,reflCount=0,trapMemory={},successPaths=[];
const memMsgs=[];const reflMsgs=[];

function addMemory(type,msg,bonus){
  memMsgs.unshift({type,msg,time:new Date().toLocaleTimeString().slice(0,5)});
  if(memMsgs.length>50)memMsgs.pop();
  const el=document.getElementById('mem-feed');
  const cls=type==='good'?'good':type==='bad'?'bad':'info';
  const icon=type==='good'?'✅':type==='bad'?'⚠️':'💡';
  el.innerHTML=memMsgs.map(m=>`<div class="mem-item ${m.type}"><span class="time">${m.time}</span><span class="msg">${m.type==='good'?'✅':m.type==='bad'?'⚠️':'💡'} ${m.msg}</span></div>`).join('');
}

function addReflection(msg){
  reflCount++;
  reflMsgs.unshift(msg);if(reflMsgs.length>20)reflMsgs.pop();
  document.getElementById('refl-feed').innerHTML=reflMsgs.map(m=>`<div class="refl-line">${m}</div>`).join('');
}

function processMemory(s,a,result,env){
  const useMemory=document.getElementById('t-mem').classList.contains('on');
  if(!useMemory)return;
  const k=s+'_'+a;
  if(result.event==='trap'){
    trapMemory[s]=(trapMemory[s]||0)+1;
    memoryBank[k]={bonus:-3,type:'trap'};memHits++;
    addMemory('bad',`Trap at ${s}! Avoid action ${['↑','↓','←','→'][a]}`,0);
    if(trapMemory[s]>2)addMemory('bad',`Repeated trap failure at ${s} (${trapMemory[s]}×)`,0);
  }else if(result.event==='goal'){
    const pathKey=env.path.slice(-5).map(p=>p.join(',')).join('>');
    successPaths.push(pathKey);memHits++;
    addMemory('good',`Goal reached! Path stored (${env.steps} steps)`,0);
    env.path.slice(-5).forEach(([r,c])=>{for(let a2=0;a2<4;a2++){const mk=`${r},${c}_${a2}`;if(!memoryBank[mk])memoryBank[mk]={bonus:0,type:'path'};memoryBank[mk].bonus+=0.5}});
  }else if(result.r>0){
    addMemory('info',`Closer to goal from ${s}`,0);
  }
}

function processReflection(epNum,reward,wins,total){
  const useRefl=document.getElementById('t-refl').classList.contains('on');
  if(!useRefl||epNum%25!==0||epNum===0)return;
  const wr=wins/total*100;
  if(wr<20)addReflection(`Low win rate (${wr.toFixed(0)}%). Increasing exploration near known traps.`);
  else if(wr<50)addReflection(`Win rate ${wr.toFixed(0)}%. Agent learning trap avoidance patterns.`);
  else if(wr<75)addReflection(`Good progress (${wr.toFixed(0)}%). Optimizing path efficiency.`);
  else addReflection(`High performance (${wr.toFixed(0)}%). Fine-tuning exploitation strategy.`);
  
  const trapCount=Object.keys(trapMemory).length;
  if(trapCount>3)addReflection(`${trapCount} trap locations memorized. Reducing repeated mistakes.`);
  if(successPaths.length>5)addReflection(`${successPaths.length} successful paths stored. Converging on optimal route.`);
  if(agent.eps<0.1)addReflection(`Exploration minimized (ε=${agent.eps.toFixed(3)}). Exploitation phase active.`);
}

// ═══ STATE ═══
let env,agent,rewards=[],episode=0,bestR=-Infinity,wins=0,totalEp=0,stepCount=0,running=false;

function init(){
  SZ=+document.getElementById('size').value;TRAPS=~~(SZ*1.2);
  env=new GridWorld(SZ);agent=new QAgent();
  rewards=[];episode=0;bestR=-Infinity;wins=0;totalEp=0;stepCount=0;
  memoryBank={};memHits=0;reflCount=0;trapMemory={};successPaths=[];
  memMsgs.length=0;reflMsgs.length=0;
  document.getElementById('mem-feed').innerHTML='';
  document.getElementById('refl-feed').innerHTML='';
  resize();drawEnv();drawChart();updateStats(0);updatePhase();
}

// ═══ DRAW ENVIRONMENT ═══
function drawEnv(){
  const w=CE.parentElement.offsetWidth,h=CE.parentElement.offsetHeight;
  const cs=Math.min(~~((w-40)/env.sz),~~((h-40)/env.sz));
  const ox=(w-cs*env.sz)/2,oy=(h-cs*env.sz)/2;
  const view=document.getElementById('view').value;
  const showH=view==='heatmap'||view==='both',showP=view==='policy'||view==='both';
  XE.fillStyle='#04060e';XE.fillRect(0,0,w,h);
  let qMin=0,qMax=0;
  if(showH||showP){for(let r=0;r<env.sz;r++)for(let c=0;c<env.sz;c++){const k=`${r},${c}`;if(env.walls.has(k))continue;for(let a=0;a<4;a++){const v=agent.getQ(k,a);if(v<qMin)qMin=v;if(v>qMax)qMax=v}}}
  const qR=qMax-qMin||1;
  const adx=[0,0,-1,1],ady=[-1,1,0,0];

  for(let r=0;r<env.sz;r++)for(let c=0;c<env.sz;c++){
    const x=ox+c*cs,y=oy+r*cs,k=`${r},${c}`;
    if(env.walls.has(k)){XE.fillStyle='#141828';XE.fillRect(x,y,cs,cs);XE.fillStyle='#1a1e35';XE.fillRect(x+1,y+1,cs-2,cs-2)}
    else if(showH){let bq=-1e9;for(let a=0;a<4;a++)bq=Math.max(bq,agent.getQ(k,a));const t=(bq-qMin)/qR;XE.fillStyle=`rgb(${~~(200*(1-t)*.5)},${~~(220*t*.7)},${~~(40+t*40)})`;XE.fillRect(x,y,cs,cs);if(cs>24){XE.fillStyle=`rgba(255,255,255,${.2+t*.4})`;XE.font=`bold ${Math.min(cs/4,9)}px JetBrains Mono`;XE.textAlign='center';XE.textBaseline='middle';XE.fillText(bq.toFixed(1),x+cs/2,y+cs/2)}}
    else{const v=env.visits[k]||0,vi=Math.min(v*6,35);XE.fillStyle=`rgb(${8+vi},${10+vi/2},${18+vi})`;XE.fillRect(x,y,cs,cs)}
    XE.strokeStyle='rgba(255,255,255,.02)';XE.strokeRect(x,y,cs,cs);
    if(showP&&!env.walls.has(k)&&!env.traps.has(k)){let ba=0,bv=-1e9;for(let a=0;a<4;a++){const v=agent.getQ(k,a);if(v>bv){bv=v;ba=a}}if(bv!==0){const al=cs*.28,cx2=x+cs/2,cy2=y+cs/2,dx=adx[ba]*al,dy=ady[ba]*al;XE.strokeStyle='rgba(0,229,160,.4)';XE.lineWidth=1.5;XE.beginPath();XE.moveTo(cx2-dx*.5,cy2-dy*.5);XE.lineTo(cx2+dx*.5,cy2+dy*.5);XE.stroke();const an=Math.atan2(dy,dx);XE.beginPath();XE.moveTo(cx2+dx*.5,cy2+dy*.5);XE.lineTo(cx2+dx*.5-5*Math.cos(an-.5),cy2+dy*.5-5*Math.sin(an-.5));XE.lineTo(cx2+dx*.5-5*Math.cos(an+.5),cy2+dy*.5-5*Math.sin(an+.5));XE.closePath();XE.fillStyle='rgba(0,229,160,.4)';XE.fill()}}
  }
  // Path
  if(view==='agent'&&env.path.length>1){XE.strokeStyle='rgba(0,229,160,.1)';XE.lineWidth=1.5;XE.beginPath();env.path.forEach(([r,c],i)=>{const x2=ox+c*cs+cs/2,y2=oy+r*cs+cs/2;i?XE.lineTo(x2,y2):XE.moveTo(x2,y2)});XE.stroke()}
  // Traps
  env.traps.forEach(t=>{const[r,c]=t.split(',').map(Number);const x=ox+c*cs,y=oy+r*cs;XE.fillStyle='rgba(255,68,102,.65)';XE.beginPath();XE.roundRect(x+cs*.15,y+cs*.15,cs*.7,cs*.7,3);XE.fill();XE.fillStyle='#fff';XE.font=`bold ${cs/3}px Inter`;XE.textAlign='center';XE.textBaseline='middle';XE.fillText('☠',x+cs/2,y+cs/2)});
  // Goal
  const pulse=Math.sin(Date.now()/300)*.15+.85;
  const gx=ox+env.goal[1]*cs,gy=oy+env.goal[0]*cs;
  const gg=XE.createRadialGradient(gx+cs/2,gy+cs/2,0,gx+cs/2,gy+cs/2,cs*1.3);gg.addColorStop(0,`rgba(255,214,102,${pulse*.15})`);gg.addColorStop(1,'transparent');XE.fillStyle=gg;XE.fillRect(gx-cs,gy-cs,cs*3,cs*3);
  XE.fillStyle=`rgba(255,214,102,${pulse})`;XE.beginPath();XE.roundRect(gx+2,gy+2,cs-4,cs-4,cs/6);XE.fill();XE.fillStyle='#000';XE.font=`bold ${cs/2.5}px Inter`;XE.fillText('★',gx+cs/2,gy+cs/2);
  // Agent
  const ax=ox+env.agent[1]*cs,ay=oy+env.agent[0]*cs;
  const ag=XE.createRadialGradient(ax+cs/2,ay+cs/2,0,ax+cs/2,ay+cs/2,cs*1.5);ag.addColorStop(0,'rgba(0,229,160,.18)');ag.addColorStop(1,'transparent');XE.fillStyle=ag;XE.fillRect(ax-cs,ay-cs,cs*3,cs*3);
  XE.fillStyle=`rgba(0,255,180,${pulse})`;XE.beginPath();XE.roundRect(ax+2,ay+2,cs-4,cs-4,cs/5);XE.fill();XE.fillStyle='#000';XE.fillText('●',ax+cs/2,ay+cs/2);
  // HUD
  XE.fillStyle='rgba(4,6,14,.82)';XE.beginPath();XE.roundRect(6,h-44,190,38,6);XE.fill();
  XE.fillStyle='#00e5a0';XE.font='bold 10px JetBrains Mono';XE.textAlign='left';XE.textBaseline='top';
  XE.fillText(`Ep:${episode} Step:${env.steps} R:${env.reward.toFixed(1)}`,12,h-38);
  XE.fillStyle='#5a6488';XE.fillText(`ε=${agent.eps.toFixed(3)} Q:${agent.qSize()} Mem:${memHits}`,12,h-24);
}

// ═══ DRAW CHART ═══
function drawChart(){
  const w=CC.parentElement.offsetWidth,h=CC.parentElement.offsetHeight;
  XC.fillStyle='#04060e';XC.fillRect(0,0,w,h);if(!rewards.length)return;
  const pad=40,rw=w-pad-12,rh=h-pad-16;
  const mn=Math.min(...rewards,-5),mx=Math.max(...rewards,5),rng=mx-mn||1;
  XC.strokeStyle='rgba(255,255,255,.03)';XC.lineWidth=1;
  for(let i=0;i<=4;i++){const y=pad+rh*i/4;XC.beginPath();XC.moveTo(pad,y);XC.lineTo(pad+rw,y);XC.stroke();XC.fillStyle='#5a6488';XC.font='8px JetBrains Mono';XC.textAlign='right';XC.textBaseline='middle';XC.fillText((mx-rng*i/4).toFixed(0),pad-4,y)}
  const sx=rw/Math.max(rewards.length-1,1);
  rewards.forEach((r,i)=>{const x=pad+i*sx,y=pad+rh*(1-(r-mn)/rng);XC.fillStyle=r>0?'rgba(0,229,160,.1)':'rgba(255,68,102,.1)';XC.beginPath();XC.arc(x,y,1.5,0,6.28);XC.fill()});
  const win=Math.max(8,~~(rewards.length/15));
  if(rewards.length>win){
    XC.beginPath();for(let i=win;i<rewards.length;i++){const avg=rewards.slice(i-win,i).reduce((a,b)=>a+b)/win;const x=pad+i*sx,y=pad+rh*(1-(avg-mn)/rng);i===win?XC.moveTo(x,y):XC.lineTo(x,y)}
    XC.lineTo(pad+(rewards.length-1)*sx,pad+rh);XC.lineTo(pad+win*sx,pad+rh);XC.closePath();
    const grd=XC.createLinearGradient(0,pad,0,pad+rh);grd.addColorStop(0,'rgba(0,229,160,.1)');grd.addColorStop(1,'transparent');XC.fillStyle=grd;XC.fill();
    XC.beginPath();XC.strokeStyle='#00e5a0';XC.lineWidth=2;
    for(let i=win;i<rewards.length;i++){const avg=rewards.slice(i-win,i).reduce((a,b)=>a+b)/win;const x=pad+i*sx,y=pad+rh*(1-(avg-mn)/rng);i===win?XC.moveTo(x,y):XC.lineTo(x,y)}XC.stroke();
  }
  if(rewards.length>1){const bi=rewards.indexOf(Math.max(...rewards));const bx=pad+bi*sx,by=pad+rh*(1-(rewards[bi]-mn)/rng);XC.fillStyle='#ffd666';XC.beginPath();XC.arc(bx,by,4,0,6.28);XC.fill();XC.font='bold 8px JetBrains Mono';XC.textAlign='center';XC.textBaseline='bottom';XC.fillText(`Best:${rewards[bi].toFixed(1)}`,bx,by-6)}
}

// ═══ STATS ═══
function updateStats(epR){
  document.getElementById('s-ep').textContent=episode;
  document.getElementById('s-st').textContent=stepCount;
  document.getElementById('s-rw').textContent=epR.toFixed?epR.toFixed(1):'0';
  document.getElementById('s-be').textContent=bestR>-Infinity?bestR.toFixed(1):'—';
  document.getElementById('s-ep2').textContent=agent.eps.toFixed(3);
  document.getElementById('s-qs').textContent=agent.qSize();
  document.getElementById('s-wr').textContent=totalEp?~~(wins/totalEp*100)+'%':'0%';
  const ph=agent.eps>0.5?'Exploring':agent.eps>0.15?'Learning':agent.eps>0.05?'Adapting':'Mastered';
  document.getElementById('s-ph').textContent=ph;
  document.getElementById('m-wr').textContent=totalEp?~~(wins/totalEp*100)+'%':'0%';
  document.getElementById('m-avg').textContent=rewards.length?((rewards.reduce((a,b)=>a+b)/rewards.length).toFixed(1)):'0';
  document.getElementById('m-best').textContent=bestR>-Infinity?bestR.toFixed(1):'—';
  document.getElementById('m-q').textContent=agent.qSize();
  document.getElementById('m-mh').textContent=memHits;
  document.getElementById('m-rc').textContent=reflCount;
  const earlyTraps=Object.values(trapMemory).filter(v=>v>2).length;
  document.getElementById('m-mr').textContent=earlyTraps>0?`-${earlyTraps}`:wins>5?'↓'+~~(earlyTraps/Math.max(1,Object.keys(trapMemory).length)*100)+'%':'—';
}

function updatePhase(){
  const p=agent.eps>0.5?0:agent.eps>0.15?1:agent.eps>0.05?2:3;
  for(let i=0;i<4;i++){document.getElementById('ph'+i).className='phase-seg'+(i<p?' done':i===p?' active':'');document.getElementById('pl'+i).className=i===p?'active':''}
}

// ═══ TRAINING LOOP ═══
async function start(){
  if(running)return;running=true;
  document.getElementById('st').textContent='Training...';
  const speed=+document.getElementById('speed').value;const delay=speed>=10?0:speed>=3?16:50;
  while(running){
    let s=env.reset();episode++;totalEp++;let epR=0,done=false;
    while(!done&&running){
      const useMem=document.getElementById('t-mem').classList.contains('on');
      const a=agent.act(s,useMem);
      const result=env.step(a);
      agent.learn(s,a,result.r,result.s,result.done);
      processMemory(s,a,result,env);
      s=result.s;epR+=result.r;done=result.done;stepCount++;
      if(delay>0||stepCount%3===0){drawEnv();await sleep(delay)}
    }
    rewards.push(epR);if(epR>bestR)bestR=epR;if(epR>10)wins++;
    agent.decay_eps();processReflection(episode,epR,wins,totalEp);
    updateStats(epR);updatePhase();drawChart();
    if(delay===0)await sleep(1);
  }
}
function doReset(){running=false;init()}
function sleep(ms){return new Promise(r=>setTimeout(r,ms))}

// ═══ CODE DEBUGGER ENV ═══
const CODE_SNIPPETS=[
  {code:['def fib(n):','  if n<=1: return n','  return fib(n-1)+fib(n-2)','  # SLOW for large n'],bug:3,fixes:['memoize','iterate','cache'],best:1,desc:'Fibonacci recursion'},
  {code:['arr = [3,1,4,1,5]','for i in range(len(arr)):','  for j in range(len(arr)):','    if arr[j]>arr[j+1]:','      arr[j],arr[j+1]=arr[j+1],arr[j]'],bug:2,fixes:['range(len-1)','range(i,len)','add bounds check'],best:0,desc:'Bubble sort bounds'},
  {code:['def search(lst, val):','  for i in range(len(lst)):','    if lst[i] == val:','      return i','  return i  # BUG'],bug:4,fixes:['return -1','return None','raise Error'],best:0,desc:'Search return bug'},
  {code:['cache = {}','def get(key):','  return cache[key]','  # KeyError if missing'],bug:2,fixes:['.get(key,None)','try/except','if key in'],best:0,desc:'Dict key error'},
  {code:['f = open("data.txt")','data = f.read()','process(data)','# File never closed'],bug:3,fixes:['with statement','finally block','f.close()'],best:0,desc:'Resource leak'},
  {code:['users = get_users()','for u in users:','  send_email(u.email)','  # No rate limiting'],bug:3,fixes:['add sleep','batch send','async queue'],best:2,desc:'Rate limit'},
  {code:['def divide(a,b):','  return a/b','  # ZeroDivisionError'],bug:1,fixes:['check b!=0','try/except','default val'],best:0,desc:'Division by zero'},
  {code:['lst=[1,2,3]','lst.remove(4)','# ValueError'],bug:1,fixes:['if 4 in lst','try/except','discard set'],best:0,desc:'Remove missing'},
];
class CodeDebugEnv{
  constructor(){this.nActions=CODE_SNIPPETS[0].fixes.length*CODE_SNIPPETS[0].code.length;this.reset()}
  reset(){this.snippet=CODE_SNIPPETS[~~(Math.random()*CODE_SNIPPETS.length)];this.steps=0;this.reward=0;this.attempts=[];this.solved=false;this.visits={};this.path=[];return this.state()}
  state(){return`code_${CODE_SNIPPETS.indexOf(this.snippet)}_${this.steps}`}
  step(a){
    const bugLine=a%this.snippet.code.length;const fixIdx=~~(a/this.snippet.code.length)%3;this.steps++;
    this.attempts.push({line:bugLine,fix:fixIdx});
    const correct=bugLine===this.snippet.bug&&fixIdx===this.snippet.best;
    if(correct){this.solved=true;return{s:this.state(),r:15-this.steps,done:true,event:'fixed',detail:`Fixed: ${this.snippet.desc} → ${this.snippet.fixes[fixIdx]}`}}
    const closeLine=Math.abs(bugLine-this.snippet.bug)<=1;
    const r=closeLine?-0.5:-1.5;this.reward+=r;
    return{s:this.state(),r,done:this.steps>=6,event:this.steps>=6?'timeout':'wrong',detail:`Tried line ${bugLine}: ${this.snippet.fixes[fixIdx]}`};
  }
}

// ═══ MULTI-AGENT ENV ═══
class MultiAgentEnv{
  constructor(sz){this.sz=sz||8;this.reset()}
  reset(){
    this.agents=[{pos:[1,1],id:0,color:'#00e5a0'},{pos:[1,this.sz-2],id:1,color:'#4488ff'},{pos:[this.sz-2,1],id:2,color:'#9966ff'}];
    this.resources=[];for(let i=0;i<5;i++){let r,c;do{r=1+~~(Math.random()*(this.sz-2));c=1+~~(Math.random()*(this.sz-2))}while(this.agents.some(a=>a.pos[0]===r&&a.pos[1]===c));this.resources.push([r,c])}
    this.collected=0;this.steps=0;this.reward=0;this.visits={};this.path=[];this.walls=new Set();
    for(let r=0;r<this.sz;r++)for(let c=0;c<this.sz;c++)if(r===0||r===this.sz-1||c===0||c===this.sz-1)this.walls.add(`${r},${c}`);
    return this.state();
  }
  state(){return this.agents.map(a=>a.pos.join(',')).join('|')}
  step(a){
    const dirs=[[-1,0],[1,0],[0,-1],[0,1]];this.steps++;
    this.agents.forEach((ag,i)=>{const d=dirs[(a+i)%4];const nr=ag.pos[0]+d[0],nc=ag.pos[1]+d[1];if(!this.walls.has(`${nr},${nc}`)&&nr>=0&&nr<this.sz&&nc>=0&&nc<this.sz)ag.pos=[nr,nc]});
    let r=-.05;
    this.resources=this.resources.filter(res=>{const hit=this.agents.some(a=>a.pos[0]===res[0]&&a.pos[1]===res[1]);if(hit){r+=5;this.collected++}return!hit});
    if(this.resources.length===0)return{s:this.state(),r:r+10,done:true,event:'goal',detail:`All ${this.collected} resources collected!`};
    this.reward+=r;this.path.push([...this.agents[0].pos]);
    return{s:this.state(),r,done:this.steps>100,event:r>0?'collect':null,detail:r>0?`Resource collected (${this.collected}/5)`:null};
  }
}

// ═══ NPC SIM ENV ═══
class NPCSimEnv{
  constructor(){this.reset()}
  reset(){
    this.npc={pos:[4,4],mood:50,trust:50,aggression:20,memory:[]};
    this.player={pos:[2,2]};this.sz=8;this.steps=0;this.reward=0;this.visits={};this.path=[];this.walls=new Set();
    for(let r=0;r<this.sz;r++)for(let c=0;c<this.sz;c++)if(r===0||r===this.sz-1||c===0||c===this.sz-1)this.walls.add(`${r},${c}`);
    return this.state();
  }
  state(){return`npc_${this.npc.mood>50?'h':'s'}_${this.npc.trust>50?'t':'u'}_${this.steps}`}
  step(a){
    // 0=approach,1=gift,2=threaten,3=trade,4=retreat,5=talk
    const actions=['approach','gift','threaten','trade','retreat','talk'];
    const act=actions[a%6];this.steps++;
    let r=0,event=null,detail=null;
    if(act==='gift'){this.npc.mood=Math.min(100,this.npc.mood+15);this.npc.trust+=10;r=3;event='good';detail='Gift accepted. NPC trust increased.'}
    else if(act==='threaten'){this.npc.aggression+=20;this.npc.trust-=15;this.npc.mood-=20;r=-4;event='bad';detail='NPC hostility increased!'}
    else if(act==='trade'){if(this.npc.trust>40){r=8;event='good';detail=`Trade successful (trust:${this.npc.trust})`}else{r=-2;event='bad';detail='Trade rejected — low trust'}}
    else if(act==='talk'){this.npc.mood+=5;this.npc.trust+=3;r=1;detail='Conversation initiated'}
    else if(act==='approach'){if(this.npc.aggression>60){r=-3;event='bad';detail='NPC attacked! Too aggressive'}else{r=0.5;detail='Approached cautiously'}}
    else{r=-0.5;detail='Retreated'}
    this.npc.memory.push(act);this.reward+=r;this.path.push([this.npc.mood,this.npc.trust]);
    const done=this.steps>=20||this.npc.trust>=90||this.npc.aggression>=90;
    if(this.npc.trust>=90){r+=15;event='goal';detail='Full trust achieved!'}
    if(this.npc.aggression>=90){r-=10;event='trap';detail='NPC became hostile!'}
    return{s:this.state(),r,done,event,detail};
  }
}

let currentEnvType='grid';
function switchEnv(type){currentEnvType=type;running=false;setTimeout(()=>init(),50)}

// ═══ WORKFLOW AGENT ENV ═══
const WORKFLOWS=[
  {id:'data_pipeline',desc:'Data Pipeline',steps:['ingest','validate','transform','load','verify'],deps:{validate:['ingest'],transform:['validate'],load:['transform'],verify:['load']},failProb:{ingest:.1,validate:.15,transform:.1,load:.2,verify:.05}},
  {id:'deploy_app',desc:'App Deployment',steps:['build','test','stage','approve','deploy'],deps:{test:['build'],stage:['test'],approve:['stage'],deploy:['approve']},failProb:{build:.15,test:.25,stage:.1,approve:.05,deploy:.2}},
  {id:'ml_pipeline',desc:'ML Training',steps:['collect','preprocess','features','train','evaluate','tune','deploy'],deps:{preprocess:['collect'],features:['preprocess'],train:['features'],evaluate:['train'],tune:['evaluate'],deploy:['tune']},failProb:{collect:.1,preprocess:.15,features:.1,train:.3,evaluate:.1,tune:.2,deploy:.25}},
  {id:'incident',desc:'Incident Response',steps:['detect','triage','diagnose','fix','test_fix','deploy_fix'],deps:{triage:['detect'],diagnose:['triage'],fix:['diagnose'],test_fix:['fix'],deploy_fix:['test_fix']},failProb:{detect:.05,triage:.1,diagnose:.25,fix:.3,test_fix:.2,deploy_fix:.15}},
];
class WorkflowEnv{
  constructor(){this.reset()}
  reset(){
    this.wf=WORKFLOWS[~~(Math.random()*WORKFLOWS.length)];
    this.completed=new Set();this.failed=new Set();this.steps=0;this.reward=0;this.retries=3;
    this.visits={};this.path=[];this.log=[];
    return this.state();
  }
  state(){return`wf_${this.wf.id}_${this.completed.size}_${this.steps}`}
  available(){return this.wf.steps.filter(s=>!this.completed.has(s)&&!this.failed.has(s)&&(this.wf.deps[s]||[]).every(d=>this.completed.has(d)))}
  step(a){
    this.steps++;
    const avail=this.available();
    // action 0..N-1 = try available steps, N = retry, N+1 = rollback
    if(a>=avail.length+2||avail.length===0){
      // retry
      if(this.retries>0&&this.failed.size>0){this.retries--;const f=[...this.failed][0];this.failed.delete(f);this.log.push({s:f,ok:null,msg:'Retrying '+f});return{s:this.state(),r:0,done:false,event:'info',detail:'Retrying '+f}}
      return{s:this.state(),r:-0.5,done:this.steps>30,event:null,detail:'No valid action'};
    }
    if(a===avail.length){// retry action
      if(this.retries>0&&this.failed.size>0){this.retries--;const f=[...this.failed][0];this.failed.delete(f);this.log.push({s:f,ok:null,msg:'Retrying '+f});return{s:this.state(),r:0.5,done:false,event:'info',detail:'Retry: '+f}}
      return{s:this.state(),r:-0.3,done:false,event:null,detail:'Nothing to retry'};
    }
    if(a===avail.length+1){// rollback
      if(this.completed.size>0){const last=[...this.completed].pop();this.completed.delete(last);this.log.push({s:last,ok:false,msg:'Rollback '+last});return{s:this.state(),r:-0.2,done:false,event:'bad',detail:'Rolled back '+last}}
      return{s:this.state(),r:-0.3,done:false,event:null,detail:'Nothing to rollback'};
    }
    const task=avail[a%avail.length];
    const fp=this.wf.failProb[task]||0.1;
    const success=Math.random()>fp;
    if(success){
      this.completed.add(task);
      const progress=this.completed.size/this.wf.steps.length;
      const allDone=this.completed.size===this.wf.steps.length;
      this.log.push({s:task,ok:true,msg:task+' completed'});
      this.reward+=progress*3;
      return{s:this.state(),r:allDone?15:progress*3,done:allDone,event:allDone?'goal':'good',detail:allDone?'WORKFLOW COMPLETE!':task+' done ('+this.completed.size+'/'+this.wf.steps.length+')'};
    }else{
      this.failed.add(task);
      this.log.push({s:task,ok:false,msg:task+' FAILED'});
      return{s:this.state(),r:-2,done:this.steps>30,event:'bad',detail:task+' failed (p='+~~(fp*100)+'%)'};
    }
  }
}

// Override init
const _origInit=init;
function init(){
  currentEnvType=document.getElementById('envType').value;
  SZ=+document.getElementById('size').value;TRAPS=~~(SZ*1.2);
  if(currentEnvType==='code')env=new CodeDebugEnv();
  else if(currentEnvType==='multi')env=new MultiAgentEnv(SZ);
  else if(currentEnvType==='npc')env=new NPCSimEnv();
  else if(currentEnvType==='workflow')env=new WorkflowEnv();
  else env=new GridWorld(SZ);
  agent=new QAgent();
  if(currentEnvType==='code'){agent.decay=0.99}
  if(currentEnvType==='npc'){agent.decay=0.99;agent.lr=0.3}
  if(currentEnvType==='workflow'){agent.decay=0.995;agent.lr=0.25}
  rewards=[];episode=0;bestR=-Infinity;wins=0;totalEp=0;stepCount=0;
  memoryBank={};memHits=0;reflCount=0;trapMemory={};successPaths=[];
  memMsgs.length=0;reflMsgs.length=0;
  document.getElementById('mem-feed').innerHTML='';
  document.getElementById('refl-feed').innerHTML='';
  resize();drawEnvDispatch();drawChart();updateStats(0);updatePhase();
}

function drawEnvDispatch(){
  if(currentEnvType==='code')drawCodeEnv();
  else if(currentEnvType==='multi')drawMultiEnv();
  else if(currentEnvType==='npc')drawNPCEnv();
  else if(currentEnvType==='workflow')drawWorkflowEnv();
  else drawEnv();
}

function drawCodeEnv(){
  const w=CE.parentElement.offsetWidth,h=CE.parentElement.offsetHeight;
  XE.fillStyle='#04060e';XE.fillRect(0,0,w,h);
  if(!env.snippet)return;
  // Terminal-style container
  XE.fillStyle='rgba(10,14,26,.95)';XE.beginPath();XE.roundRect(12,12,w-24,h-24,10);XE.fill();
  XE.strokeStyle='rgba(0,229,160,.08)';XE.lineWidth=1;XE.stroke();
  // Title bar
  XE.fillStyle='rgba(0,229,160,.06)';XE.beginPath();XE.roundRect(12,12,w-24,28,{upperLeft:10,upperRight:10,lowerLeft:0,lowerRight:0});XE.fill();
  XE.fillStyle='#ff4466';XE.beginPath();XE.arc(26,26,4,0,6.28);XE.fill();
  XE.fillStyle='#ffd666';XE.beginPath();XE.arc(38,26,4,0,6.28);XE.fill();
  XE.fillStyle='#00e5a0';XE.beginPath();XE.arc(50,26,4,0,6.28);XE.fill();
  XE.fillStyle='#5a6c8a';XE.font='bold 10px JetBrains Mono';XE.textAlign='left';XE.textBaseline='middle';
  XE.fillText(env.snippet.desc+'.py',64,26);
  // Line numbers + code
  const lh=20,startY=52;
  env.snippet.code.forEach((line,i)=>{
    const y=startY+i*lh;const isB=i===env.snippet.bug;
    if(isB){XE.fillStyle='rgba(255,68,102,.06)';XE.fillRect(20,y-2,w-40,lh)}
    // Line number gutter
    XE.fillStyle=isB?'#ff4466':'#2a3050';XE.font='10px JetBrains Mono';XE.textAlign='right';
    XE.fillText(String(i+1),38,y+4);
    // Separator
    XE.fillStyle='rgba(255,255,255,.04)';XE.fillRect(42,y-2,1,lh);
    // Code
    XE.fillStyle=isB?'#ff6680':'#c0c8e0';XE.font='11px JetBrains Mono';XE.textAlign='left';
    XE.fillText(line,48,y+4);
    // Attempt markers
    const attemptsHere=env.attempts.filter(a=>a.line===i);
    attemptsHere.forEach((at,j)=>{XE.fillStyle='rgba(255,68,102,.6)';XE.fillText('x',w-36-j*10,y+4)});
  });
  if(env.solved){
    const by=startY+env.snippet.bug*lh;
    XE.fillStyle='rgba(0,229,160,.08)';XE.fillRect(20,by-2,w-40,lh);
    XE.fillStyle='#00e5a0';XE.font='bold 11px JetBrains Mono';XE.fillText('FIXED',w-70,by+4);
  }
  // Fix options at bottom
  const fy=h-50;
  XE.fillStyle='rgba(0,229,160,.04)';XE.beginPath();XE.roundRect(20,fy,w-40,32,6);XE.fill();
  XE.fillStyle='#5a6c8a';XE.font='9px JetBrains Mono';XE.fillText('Available fixes:',28,fy+8);
  env.snippet.fixes.forEach((f,i)=>{
    const fx=28+i*100;
    XE.fillStyle=i===env.snippet.best?'#00e5a0':'#5a6c8a';XE.font='bold 10px JetBrains Mono';
    XE.fillText(`[${i}] ${f}`,fx,fy+22);
  });
  // Status
  XE.fillStyle='#5a6c8a';XE.font='9px JetBrains Mono';
  XE.fillText(`Attempts: ${env.attempts.length}/6  |  Bug: line ?  |  Status: ${env.solved?'FIXED':'debugging...'}`,28,h-16);
}

function drawMultiEnv(){
  const w=CE.parentElement.offsetWidth,h=CE.parentElement.offsetHeight;
  const cs=Math.min(~~((w-40)/env.sz),~~((h-60)/env.sz));
  const ox=(w-cs*env.sz)/2,oy=(h-cs*env.sz)/2+8;
  XE.fillStyle='#04060e';XE.fillRect(0,0,w,h);
  // Title
  XE.fillStyle='#5a6c8a';XE.font='bold 10px JetBrains Mono';XE.textAlign='center';XE.textBaseline='top';
  XE.fillText('MULTI-AGENT COORDINATION',w/2,6);
  // Grid
  for(let r=0;r<env.sz;r++)for(let c=0;c<env.sz;c++){
    const k=`${r},${c}`,x=ox+c*cs,y=oy+r*cs;
    if(env.walls.has(k)){XE.fillStyle='#141828';XE.fillRect(x,y,cs,cs)}
    else{XE.fillStyle='#080e18';XE.fillRect(x,y,cs,cs)}
    XE.strokeStyle='rgba(0,229,160,.03)';XE.strokeRect(x,y,cs,cs);
  }
  // Resources with pulse
  const pulse=Math.sin(Date.now()/300)*.15+.85;
  env.resources.forEach(([r,c])=>{
    const x=ox+c*cs,y=oy+r*cs;
    const glow=XE.createRadialGradient(x+cs/2,y+cs/2,0,x+cs/2,y+cs/2,cs);
    glow.addColorStop(0,`rgba(255,214,102,${pulse*.12})`);glow.addColorStop(1,'transparent');
    XE.fillStyle=glow;XE.fillRect(x-cs/2,y-cs/2,cs*2,cs*2);
    XE.fillStyle=`rgba(255,214,102,${pulse*.8})`;XE.beginPath();XE.roundRect(x+cs*.15,y+cs*.15,cs*.7,cs*.7,4);XE.fill();
    XE.fillStyle='#000';XE.font=`bold ${cs/3}px Inter`;XE.textAlign='center';XE.textBaseline='middle';XE.fillText('R',x+cs/2,y+cs/2);
  });
  // Agents with trails
  env.agents.forEach(ag=>{
    const x=ox+ag.pos[1]*cs,y=oy+ag.pos[0]*cs;
    const glow=XE.createRadialGradient(x+cs/2,y+cs/2,0,x+cs/2,y+cs/2,cs*1.2);
    glow.addColorStop(0,ag.color+'22');glow.addColorStop(1,'transparent');
    XE.fillStyle=glow;XE.fillRect(x-cs/2,y-cs/2,cs*2,cs*2);
    XE.fillStyle=ag.color+'dd';XE.beginPath();XE.roundRect(x+2,y+2,cs-4,cs-4,cs/4);XE.fill();
    XE.fillStyle='#fff';XE.font=`bold ${Math.max(cs/4,8)}px JetBrains Mono`;XE.textAlign='center';XE.textBaseline='middle';
    XE.fillText(`A${ag.id}`,x+cs/2,y+cs/2);
  });
  // Status bar
  XE.fillStyle='rgba(4,6,14,.85)';XE.beginPath();XE.roundRect(6,h-34,w-12,28,6);XE.fill();
  XE.fillStyle='#00e5a0';XE.font='bold 10px JetBrains Mono';XE.textAlign='left';XE.textBaseline='middle';
  XE.fillText(`Resources: ${env.collected}/5`,14,h-20);
  XE.fillStyle='#4488ff';XE.fillText(`Agents: ${env.agents.length}`,140,h-20);
  XE.fillStyle='#5a6c8a';XE.fillText(`Steps: ${env.steps}/100`,240,h-20);
}

function drawNPCEnv(){
  const w=CE.parentElement.offsetWidth,h=CE.parentElement.offsetHeight;
  XE.fillStyle='#04060e';XE.fillRect(0,0,w,h);
  XE.fillStyle='rgba(10,14,26,.95)';XE.beginPath();XE.roundRect(12,12,w-24,h-24,10);XE.fill();
  XE.strokeStyle='rgba(0,229,160,.04)';XE.stroke();
  const npc=env.npc;
  // Title
  XE.fillStyle='#5a6c8a';XE.font='bold 10px JetBrains Mono';XE.textAlign='center';XE.textBaseline='top';
  XE.fillText('NPC SIMULATION - SOCIAL DYNAMICS',w/2,20);
  // NPC portrait
  const cy=h/2-30;
  const faceColor=npc.mood>60?'#00e5a0':npc.mood>30?'#ffd666':'#ff4466';
  // Outer glow ring
  const pulse=Math.sin(Date.now()/400)*.1+.9;
  XE.strokeStyle=faceColor+'44';XE.lineWidth=1;XE.beginPath();XE.arc(w/2,cy,58*pulse,0,6.28);XE.stroke();
  // Inner circle
  XE.fillStyle=faceColor+'18';XE.beginPath();XE.arc(w/2,cy,45,0,6.28);XE.fill();
  XE.strokeStyle=faceColor+'88';XE.lineWidth=2;XE.stroke();
  // Face emoji
  const face=npc.aggression>60?'(>_<)':npc.mood>60?'(^_^)':npc.mood>30?'(-_-)':'(T_T)';
  XE.fillStyle=faceColor;XE.font='bold 16px JetBrains Mono';XE.textAlign='center';XE.textBaseline='middle';
  XE.fillText(face,w/2,cy);
  // NPC name
  XE.fillStyle='#c0c8e0';XE.font='bold 11px Inter';XE.fillText('NPC Agent',w/2,cy+60);
  // Stat bars with numeric values
  const barW=w-100,barH=8,barX=50,barY=cy+80;
  [['MOOD',npc.mood,'#00e5a0'],['TRUST',npc.trust,'#4488ff'],['AGGRESSION',npc.aggression,'#ff4466']].forEach(([l,v,c],i)=>{
    const y=barY+i*32;
    XE.fillStyle='#5a6c8a';XE.font='bold 8px JetBrains Mono';XE.textAlign='left';XE.textBaseline='bottom';
    XE.fillText(l,barX,y-3);
    XE.fillStyle='#c0c8e0';XE.textAlign='right';XE.fillText(~~v+'/100',barX+barW,y-3);
    XE.fillStyle='#0a0e1a';XE.beginPath();XE.roundRect(barX,y,barW,barH,4);XE.fill();
    const grad=XE.createLinearGradient(barX,0,barX+barW*v/100,0);
    grad.addColorStop(0,c);grad.addColorStop(1,c+'88');
    XE.fillStyle=grad;XE.beginPath();XE.roundRect(barX,y,barW*Math.max(v,1)/100,barH,4);XE.fill();
    // Glow
    XE.shadowColor=c;XE.shadowBlur=6;XE.fillRect(barX,y,barW*v/100,barH);XE.shadowBlur=0;
  });
  // Action history
  XE.fillStyle='rgba(0,229,160,.04)';XE.beginPath();XE.roundRect(20,h-90,w-40,74,6);XE.fill();
  XE.fillStyle='#5a6c8a';XE.font='bold 8px JetBrains Mono';XE.textAlign='left';XE.textBaseline='top';
  XE.fillText('INTERACTION LOG',28,h-84);
  XE.font='9px JetBrains Mono';XE.fillStyle='#8090b0';
  npc.memory.slice(-4).forEach((m,i)=>{
    const col=m==='gift'||m==='trade'||m==='talk'?'#00e5a0':m==='threaten'?'#ff4466':'#5a6c8a';
    XE.fillStyle=col;XE.fillText('> '+m,28,h-70+i*14);
  });
}

function drawWorkflowEnv(){
  const w=CE.parentElement.offsetWidth,h=CE.parentElement.offsetHeight;
  XE.fillStyle='#04060e';XE.fillRect(0,0,w,h);
  XE.fillStyle='rgba(10,14,26,.95)';XE.beginPath();XE.roundRect(12,12,w-24,h-24,10);XE.fill();
  XE.strokeStyle='rgba(0,229,160,.04)';XE.stroke();
  if(!env.wf)return;
  // Title
  XE.fillStyle='#00e5a0';XE.font='bold 11px JetBrains Mono';XE.textAlign='center';XE.textBaseline='top';
  XE.fillText('WORKFLOW: '+env.wf.desc.toUpperCase(),w/2,20);
  // Pipeline visualization
  const steps=env.wf.steps;const n=steps.length;
  const nodeW=Math.min(70,(w-60)/n-8);const nodeH=32;
  const startX=(w-(n*(nodeW+8)-8))/2;const pipeY=60;
  steps.forEach((s,i)=>{
    const x=startX+i*(nodeW+8);
    const done=env.completed.has(s);const fail=env.failed.has(s);
    const avail=env.available().includes(s);
    // Connection line
    if(i>0){
      const prevDone=env.completed.has(steps[i-1]);
      XE.strokeStyle=prevDone?'#00e5a066':'#1a2844';XE.lineWidth=2;
      XE.beginPath();XE.moveTo(x-6,pipeY+nodeH/2);XE.lineTo(x-2,pipeY+nodeH/2);XE.stroke();
      // Arrow
      XE.fillStyle=prevDone?'#00e5a066':'#1a2844';
      XE.beginPath();XE.moveTo(x-2,pipeY+nodeH/2-3);XE.lineTo(x+2,pipeY+nodeH/2);XE.lineTo(x-2,pipeY+nodeH/2+3);XE.fill();
    }
    // Node
    const col=done?'#00e5a0':fail?'#ff4466':avail?'#ffd666':'#1a2844';
    XE.fillStyle=col+'18';XE.beginPath();XE.roundRect(x,pipeY,nodeW,nodeH,6);XE.fill();
    XE.strokeStyle=col+'88';XE.lineWidth=1;XE.stroke();
    if(done||avail){XE.shadowColor=col;XE.shadowBlur=8;XE.strokeRect(x,pipeY,nodeW,nodeH);XE.shadowBlur=0}
    // Label
    XE.fillStyle=done?'#00e5a0':fail?'#ff4466':avail?'#ffd666':'#5a6c8a';
    XE.font=`bold ${Math.min(8,nodeW/s.length*1.2)}px JetBrains Mono`;XE.textAlign='center';XE.textBaseline='middle';
    XE.fillText(s.length>8?s.slice(0,7)+'..':s,x+nodeW/2,pipeY+nodeH/2);
    // Status icon
    XE.font='10px Inter';
    if(done)XE.fillText('v',x+nodeW/2,pipeY-6);
    if(fail)XE.fillText('x',x+nodeW/2,pipeY-6);
  });
  // Progress bar
  const prog=env.completed.size/n;
  const pbY=pipeY+nodeH+16;
  XE.fillStyle='#0a0e1a';XE.beginPath();XE.roundRect(30,pbY,w-60,6,3);XE.fill();
  const pgGrad=XE.createLinearGradient(30,0,30+(w-60)*prog,0);
  pgGrad.addColorStop(0,'#00e5a0');pgGrad.addColorStop(1,'#00ccff');
  XE.fillStyle=pgGrad;XE.beginPath();XE.roundRect(30,pbY,Math.max((w-60)*prog,4),6,3);XE.fill();
  XE.fillStyle='#5a6c8a';XE.font='8px JetBrains Mono';XE.textAlign='center';XE.textBaseline='top';
  XE.fillText(`${env.completed.size}/${n} (${~~(prog*100)}%)`,w/2,pbY+10);
  // Execution log
  const logY=pbY+30;
  XE.fillStyle='rgba(0,0,0,.3)';XE.beginPath();XE.roundRect(20,logY,w-40,h-logY-20,6);XE.fill();
  XE.strokeStyle='rgba(0,229,160,.04)';XE.stroke();
  XE.fillStyle='#5a6c8a';XE.font='bold 8px JetBrains Mono';XE.textAlign='left';XE.textBaseline='top';
  XE.fillText('EXECUTION LOG',28,logY+6);
  XE.font='9px JetBrains Mono';
  env.log.slice(-8).forEach((entry,i)=>{
    const y=logY+20+i*14;
    XE.fillStyle=entry.ok===true?'#00e5a0':entry.ok===false?'#ff4466':'#ffd666';
    XE.fillText((entry.ok===true?'[OK] ':entry.ok===false?'[FAIL] ':'[..] ')+entry.msg,28,y);
  });
  // Footer stats
  XE.fillStyle='#5a6c8a';XE.font='9px JetBrains Mono';XE.textAlign='left';
  XE.fillText(`Retries: ${env.retries}/3  |  Failed: ${env.failed.size}  |  Steps: ${env.steps}`,28,h-16);
}

// Override training loop
async function start(){
  if(running)return;running=true;
  document.getElementById('st').textContent='Training...';
  const speed=+document.getElementById('speed').value;const delay=speed>=10?0:speed>=3?16:50;
  const nActions=currentEnvType==='code'?15:currentEnvType==='npc'?6:currentEnvType==='workflow'?8:4;
  while(running){
    let s=env.reset();episode++;totalEp++;let epR=0,done=false;
    while(!done&&running){
      const useMem=document.getElementById('t-mem').classList.contains('on');
      const a=agent.act(s,useMem);
      const result=env.step(a%nActions);
      agent.learn(s,a%nActions,result.r,result.s,result.done);
      processMemoryGeneric(s,a%nActions,result);
      s=result.s;epR+=result.r;done=result.done;stepCount++;
      if(delay>0||stepCount%3===0){drawEnvDispatch();await sleep(delay)}
    }
    rewards.push(epR);if(epR>bestR)bestR=epR;if(epR>10)wins++;
    agent.decay_eps();processReflection(episode,epR,wins,totalEp);
    updateStats(epR);updatePhase();drawChart();
    if(delay===0)await sleep(1);
  }
}

function processMemoryGeneric(s,a,result){
  const useMemory=document.getElementById('t-mem').classList.contains('on');
  if(!useMemory)return;
  if(result.event==='fixed'||result.event==='goal'){memHits++;addMemory('good',result.detail||'Success!');successPaths.push(s)}
  else if(result.event==='trap'||result.event==='bad'){memHits++;trapMemory[s]=(trapMemory[s]||0)+1;addMemory('bad',result.detail||'Failed at '+s)}
  else if(result.event==='collect'||result.event==='good'){memHits++;addMemory('info',result.detail||'Progress')}
  else if(result.event==='info'){addMemory('info',result.detail||'Info')}
  else if(result.detail&&Math.random()<0.05)addMemory('info',result.detail);
}

function doReset(){running=false;init()}
init();

