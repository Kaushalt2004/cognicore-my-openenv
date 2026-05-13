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
init();
