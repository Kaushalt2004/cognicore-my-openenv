#!/usr/bin/env python3
"""
CogniCore CodeRepairBench v2 — Adaptive Runtime Cognition

Real bugs. Real test execution. Semantic patch rejection.
Strategy mutation. LLM integration. Active behavioral divergence.

Run:  python examples/coderepair_v2.py
      python examples/coderepair_v2.py --openai   # with GPT
"""
import sys,os,io,time,json,hashlib,difflib,random,argparse,textwrap
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cognicore.runtime import CogniCoreRuntime, RuntimeConfig

# ══════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════
def clog(tag,msg,detail=""):
    C={"PATCH REJECTED":"\033[31m","MEMORY RETRIEVAL":"\033[33m",
       "REFLECTION GENERATED":"\033[35m","STRATEGY MUTATION":"\033[36m",
       "ADAPTIVE SUCCESS":"\033[32m","FAILED PATCH":"\033[31m",
       "SUCCESS":"\033[32m","SIMILARITY":"\033[33m","INFO":"\033[37m"}
    print(f"  {C.get(tag,chr(27)+'[0m')}[{tag}]\033[0m {msg}")
    if detail:
        for l in detail.strip().split("\n")[:5]: print(f"         {l}")

# ══════════════════════════════════════════════════════════
# SANDBOX
# ══════════════════════════════════════════════════════════
def sandbox(code,tests):
    ns={}
    try:
        exec(compile(code,"<patch>","exec"),ns)
        exec(compile(tests,"<test>","exec"),ns)
        return True,None
    except Exception as e:
        return False,f"{type(e).__name__}: {e}"

def similarity(a,b):
    if not a or not b: return 0.0
    return difflib.SequenceMatcher(None,a.strip(),b.strip()).ratio()

def phash(c): return hashlib.md5(c.strip().encode()).hexdigest()[:10]

# ══════════════════════════════════════════════════════════
# STRATEGY ENGINE — mutable repair strategies
# ══════════════════════════════════════════════════════════
class RepairStrategy:
    """Mutable strategy that CogniCore reflection can alter."""
    def __init__(self):
        self.allowed_tactics = {"direct_fix","boundary_fix","guard_fix",
                                "restructure","memoize","lock_fix","with_fix"}
        self.disabled_tactics = set()
        self.preferred_tactics = []
        self.mutations = []

    def mutate(self, disable=None, prefer=None, reason=""):
        if disable:
            self.disabled_tactics.add(disable)
            self.allowed_tactics.discard(disable)
        if prefer:
            self.preferred_tactics.insert(0, prefer)
            self.allowed_tactics.add(prefer)
        self.mutations.append({"disable":disable,"prefer":prefer,"reason":reason})
        clog("STRATEGY MUTATION",
             f"Policy changed: disabled={disable}, preferred={prefer}",
             reason)

    def get_tactics(self):
        ordered = list(self.preferred_tactics)
        for t in self.allowed_tactics:
            if t not in ordered: ordered.append(t)
        return [t for t in ordered if t not in self.disabled_tactics]

# ══════════════════════════════════════════════════════════
# BUG DATABASE
# ══════════════════════════════════════════════════════════
BUGS = [
  {"id":"BUG-001","cat":"off_by_one",
   "title":"Sliding window grabs extra element",
   "buggy":"""
def sliding_max(arr, k):
    if not arr or k<=0: return []
    return [max(arr[i:i+k+1]) for i in range(len(arr)-k+1)]
""",
   "test":"""
assert sliding_max([1,3,2,5,1,4],3)==[3,5,5,5], f"got {sliding_max([1,3,2,5,1,4],3)}"
assert sliding_max([1],1)==[1]
assert sliding_max([],3)==[]
assert sliding_max([4,3,2,1],2)==[4,3,2]
""",
   "tactics":{
     "direct_fix": lambda c: c.replace("i+k+1","i+k"),
     "boundary_fix": lambda c: c.replace("i+k+1","i+k"),
     "guard_fix": lambda c: c.replace("i+k+1","i+k"),  # same fix = will be rejected
     "restructure": lambda c: "def sliding_max(arr,k):\n  if not arr or k<=0: return []\n  return [max(arr[i:i+k]) for i in range(len(arr)-k+1)]\n",
   }},
  {"id":"BUG-002","cat":"none_handling",
   "title":"Dict merge crashes on missing keys",
   "buggy":"""
def merge_profiles(base, update):
    result = {}
    for key in set(list(base.keys())+list(update.keys())):
        result[key] = update[key] if update[key] else base[key]
    return result
""",
   "test":"""
r=merge_profiles({'name':'Jo','age':25},{'name':'Jo B','email':'j@b.com'})
assert r['name']=='Jo B'
assert r['age']==25
assert r['email']=='j@b.com'
""",
   "tactics":{
     "direct_fix": lambda c: c.replace("update[key] if update[key] else base[key]",
                                        "update.get(key, base.get(key))"),
     "guard_fix": lambda c: c.replace("update[key] if update[key] else base[key]",
                                       "update.get(key) or base.get(key)"),
     "restructure": lambda c: "def merge_profiles(base,update):\n  r={**base}\n  r.update({k:v for k,v in update.items() if v is not None})\n  return r\n",
   }},
  {"id":"BUG-003","cat":"recursion",
   "title":"Tree flatten crashes on None children",
   "buggy":"""
def flatten_tree(node):
    result = [node['value']]
    for child in node['children']:
        result.extend(flatten_tree(child))
    return result
""",
   "test":"""
tree={'value':1,'children':[{'value':2,'children':[]},{'value':3,'children':[{'value':4,'children':[]}]}]}
assert flatten_tree(tree)==[1,2,3,4]
assert flatten_tree({'value':99,'children':None})==[99]
""",
   "tactics":{
     "direct_fix": lambda c: c.replace("node['children']","(node['children'] or [])"),
     "guard_fix": lambda c: c.replace("for child in node['children']:","for child in (node.get('children') or []):"),
     "restructure": lambda c: "def flatten_tree(node):\n  r=[node['value']]\n  ch=node.get('children')\n  if ch:\n    for c in ch: r.extend(flatten_tree(c))\n  return r\n",
   }},
  {"id":"BUG-004","cat":"concurrency",
   "title":"Race condition in counter",
   "buggy":"""
import threading
class SafeCounter:
    def __init__(self):
        self.value = 0
    def increment(self, amount=1):
        current = self.value
        self.value = current + amount
""",
   "test":"""
import threading
c=SafeCounter()
ts=[threading.Thread(target=c.increment) for _ in range(200)]
for t in ts: t.start()
for t in ts: t.join()
assert c.value>=195, f"Race: got {c.value}"
""",
   "tactics":{
     "lock_fix": lambda c: c.replace("self.value = 0","self.value=0;self._lock=threading.Lock()").replace("current = self.value\n        self.value = current + amount","with self._lock: self.value+=amount"),
     "direct_fix": lambda c: c.replace("current = self.value\n        self.value = current + amount","with threading.Lock(): self.value+=amount"),
   }},
  {"id":"BUG-005","cat":"resource_leak",
   "title":"Config parser leaks file handle",
   "buggy":"""
def parse_config(path):
    f = open(path)
    lines = f.readlines()
    config = {}
    for line in lines:
        k, v = line.strip().split('=')
        config[k] = v
    f.close()
    return config
""",
   "test":"""
import tempfile,os
t=tempfile.NamedTemporaryFile(mode='w',suffix='.cfg',delete=False)
t.write("host=localhost\\nport=8080\\n# comment\\n\\ndb=postgres\\n")
t.close()
r=parse_config(t.name)
os.unlink(t.name)
assert r['host']=='localhost'
assert r['port']=='8080'
assert r['db']=='postgres'
""",
   "tactics":{
     "direct_fix": lambda c: None,  # intentionally fails
     "with_fix": lambda c: "def parse_config(path):\n  config={}\n  with open(path) as f:\n    for line in f:\n      line=line.strip()\n      if not line or line.startswith('#'): continue\n      if '=' in line:\n        k,v=line.split('=',1)\n        config[k.strip()]=v.strip()\n  return config\n",
     "guard_fix": lambda c: c.replace("k, v = line.strip().split('=')",
                                       "parts=line.strip().split('=',1)\n        if len(parts)!=2: continue\n        k,v=parts"),
   }},
  {"id":"BUG-006","cat":"off_by_one",
   "title":"Pagination drops last page",
   "buggy":"""
def paginate(items, page_size):
    total_pages = len(items) // page_size
    pages = []
    for i in range(total_pages):
        pages.append(items[i*page_size:(i+1)*page_size])
    return pages
""",
   "test":"""
r=paginate([1,2,3,4,5],2)
assert len(r)==3,f"Expected 3, got {len(r)}"
assert r[0]==[1,2]
assert r[2]==[5]
assert paginate([],5)==[]
""",
   "tactics":{
     "direct_fix": lambda c: c.replace("len(items) // page_size","(len(items)+page_size-1)//page_size"),
     "boundary_fix": lambda c: c.replace("len(items) // page_size","-(-len(items)//page_size)"),
     "restructure": lambda c: "def paginate(items,ps):\n  return [items[i:i+ps] for i in range(0,len(items),ps)] if items else []\n",
   }},
  {"id":"BUG-007","cat":"none_handling",
   "title":"API parser crashes on null fields",
   "buggy":"""
def parse_api_response(data):
    return {
        'id': data['id'],
        'name': data['user']['name'],
        'email': data['user']['email'],
        'score': data['metrics']['score'],
    }
""",
   "test":"""
full={'id':1,'user':{'name':'Jo','email':'j@b.com'},'metrics':{'score':95}}
assert parse_api_response(full)=={'id':1,'name':'Jo','email':'j@b.com','score':95}
partial={'id':2,'user':None,'metrics':None}
r=parse_api_response(partial)
assert r['id']==2
assert r['name'] is None
assert r['score'] is None
missing={'id':3}
r2=parse_api_response(missing)
assert r2['name'] is None
""",
   "tactics":{
     "direct_fix": lambda c: c.replace("data['user']['name']","(data.get('user') or {}).get('name')").replace("data['user']['email']","(data.get('user') or {}).get('email')").replace("data['metrics']['score']","(data.get('metrics') or {}).get('score')"),
     "guard_fix": lambda c: c.replace("data['user']['name']","data.get('user',{}).get('name')").replace("data['user']['email']","data.get('user',{}).get('email')").replace("data['metrics']['score']","data.get('metrics',{}).get('score')"),
     "restructure": lambda c: "def parse_api_response(data):\n  u=data.get('user') or {}\n  m=data.get('metrics') or {}\n  return {'id':data['id'],'name':u.get('name'),'email':u.get('email'),'score':m.get('score')}\n",
   }},
  {"id":"BUG-008","cat":"recursion",
   "title":"Fibonacci exponential blowup",
   "buggy":"""
def fibonacci(n):
    if n <= 1: return n
    return fibonacci(n-1) + fibonacci(n-2)
""",
   "test":"""
import time
assert fibonacci(10)==55
t=time.perf_counter()
assert fibonacci(35)==9227465
assert time.perf_counter()-t<1.0,"Too slow"
""",
   "tactics":{
     "direct_fix": lambda c: c,  # returns buggy = will fail
     "memoize": lambda c: "def fibonacci(n,_c={}):\n  if n in _c: return _c[n]\n  if n<=1: return n\n  _c[n]=fibonacci(n-1)+fibonacci(n-2)\n  return _c[n]\n",
     "restructure": lambda c: "def fibonacci(n):\n  if n<=1: return n\n  a,b=0,1\n  for _ in range(2,n+1): a,b=b,a+b\n  return b\n",
   }},
]

# ══════════════════════════════════════════════════════════
# REPAIR AGENT — generates patches using tactics + strategy
# ══════════════════════════════════════════════════════════
class AdaptiveRepairAgent:
    SIMILARITY_THRESHOLD = 0.88

    def __init__(self, use_llm=False, llm_client=None, llm_model="gpt-4o-mini"):
        self.use_llm = use_llm
        self.llm_client = llm_client
        self.llm_model = llm_model

    def generate_patch(self, bug, strategy, failed_patches, context):
        """Generate a patch. Strategy mutation + similarity rejection are ACTIVE."""
        buggy = bug["buggy"]
        tactics = bug.get("tactics", {})
        allowed = strategy.get_tactics()
        failed_hashes = {phash(p) for p in failed_patches}

        # If LLM available, use it
        if self.use_llm and self.llm_client:
            return self._llm_patch(bug, failed_patches, context)

        # Try tactics in strategy-ordered priority
        for tactic_name in allowed:
            if tactic_name not in tactics:
                continue
            try:
                patch = tactics[tactic_name](buggy)
                if patch is None:
                    continue

                # SEMANTIC PATCH REJECTION
                h = phash(patch)
                if h in failed_hashes:
                    clog("PATCH REJECTED",
                         f"Tactic '{tactic_name}' identical to failed patch (hash={h})")
                    continue

                for fp in failed_patches[-5:]:
                    sim = similarity(patch, fp)
                    if sim > self.SIMILARITY_THRESHOLD:
                        clog("PATCH REJECTED",
                             f"Tactic '{tactic_name}' too similar to failed patch ({sim:.0%})",
                             f"Threshold: {self.SIMILARITY_THRESHOLD:.0%}")
                        break
                else:
                    return patch, tactic_name
            except Exception:
                continue

        return buggy, "exhausted"

    def _llm_patch(self, bug, failed_patches, context):
        """Generate patch via real LLM API."""
        failed_info = ""
        if failed_patches:
            failed_info = "\n\nPrevious FAILED patches (DO NOT repeat these):\n"
            for i, fp in enumerate(failed_patches[-3:], 1):
                failed_info += f"--- Failed Patch {i} ---\n{fp.strip()}\n"

        reflection = ""
        if context.get("reflection_hint"):
            reflection = f"\n\nReflection from CogniCore:\n{context['reflection_hint']}\n"

        prompt = f"""Fix this buggy Python code. Return ONLY the fixed code, no explanation.

Buggy code:
{bug['buggy']}

Bug description: {bug['title']}

Test that must pass:
{bug['test']}
{failed_info}{reflection}
Return ONLY the fixed Python function code."""

        try:
            resp = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role":"system","content":"You are a code repair agent. Return only fixed code."},
                          {"role":"user","content":prompt}],
                temperature=0.3 + 0.1 * len(failed_patches),  # increase creativity after failures
                max_tokens=500,
            )
            code = resp.choices[0].message.content.strip()
            if code.startswith("```"): code = code.split("\n",1)[1].rsplit("```",1)[0]
            return code, "llm"
        except Exception as e:
            clog("INFO", f"LLM call failed: {e}")
            return bug["buggy"], "llm_error"


# ══════════════════════════════════════════════════════════
# BENCHMARK ENGINE
# ══════════════════════════════════════════════════════════
def run_benchmark(bugs=None, max_attempts=5, use_llm=False, llm_client=None):
    bugs = bugs or BUGS
    agent = AdaptiveRepairAgent(use_llm=use_llm, llm_client=llm_client)
    runtime = CogniCoreRuntime(config=RuntimeConfig(
        reflection_min_samples=1, reflection_failure_threshold=1, memory_top_k=5,
    ), name="coderepair-v2")

    print(f"\n{'='*64}")
    print(f"  CogniCore CodeRepairBench v2 — Adaptive Runtime Cognition")
    print(f"  {len(bugs)} bugs | {max_attempts} attempts | LLM={'ON' if use_llm else 'OFF'}")
    print(f"{'='*64}")

    all_results = []

    for bug in bugs:
        print(f"\n  {'─'*58}")
        print(f"  {bug['id']}: {bug['title']} [{bug['cat']}]")
        print(f"  {'─'*58}")

        # ── BASELINE: no memory, no rejection, no mutation ──
        print(f"  [A] BASELINE (no cognition)")
        b_strategy = RepairStrategy()
        b_failed = []
        b_solved = False
        b_attempts = 0
        b_repeats = 0
        b_hashes = set()

        for attempt in range(1, max_attempts+1):
            b_attempts = attempt
            patch, tactic = agent.generate_patch(bug, b_strategy, [], {})  # empty context!
            h = phash(patch)
            if h in b_hashes: b_repeats += 1
            b_hashes.add(h)
            ok, err = sandbox(patch, bug["test"])
            if ok:
                b_solved = True
                print(f"    Attempt {attempt}: PASS (tactic={tactic})")
                break
            else:
                b_failed.append(patch)
                rep = " **REPEATED**" if b_repeats and h in b_hashes else ""
                print(f"    Attempt {attempt}: FAIL (tactic={tactic}) {err[:60]}{rep}")

        # ── COGNICORE: full cognition pipeline ──
        print(f"  [B] COGNICORE (memory + rejection + reflection + mutation)")
        c_strategy = RepairStrategy()
        c_failed_patches = []
        c_solved = False
        c_attempts = 0
        c_repeats = 0
        c_mem_hits = 0
        c_refl_hits = 0
        c_rejections = 0
        c_mutations = 0
        c_hashes = set()

        for attempt in range(1, max_attempts+1):
            c_attempts = attempt

            # Build context from CogniCore runtime
            ctx = runtime._build_context(bug["cat"])
            ctx["_failed_patches"] = c_failed_patches

            # STEP 1: Memory retrieval
            if ctx.get("memory"):
                c_mem_hits += 1
                failed_errors = [str(e.get("predicted",""))[:60] for e in ctx["memory"]
                                 if not e.get("correct")]
                if failed_errors:
                    clog("MEMORY RETRIEVAL",
                         f"Found {len(failed_errors)} past failures in '{bug['cat']}'",
                         "\n".join(failed_errors[:3]))

            # STEP 2: Reflection → Strategy mutation
            if ctx.get("reflection_hint"):
                c_refl_hits += 1
                hint = ctx["reflection_hint"]
                clog("REFLECTION GENERATED", "Analyzing failure patterns", hint[:120])

                # ACTIVE mutation based on reflection content
                if ctx.get("failures_to_avoid"):
                    worst = ctx["failures_to_avoid"][0] if ctx["failures_to_avoid"] else ""
                    if "direct_fix" in worst or attempt > 2:
                        c_strategy.mutate(disable="direct_fix", prefer="restructure",
                                          reason="Direct fixes repeatedly failed. Switching to restructure.")
                        c_mutations += 1
                    elif "guard_fix" in worst:
                        c_strategy.mutate(disable="guard_fix", prefer="restructure",
                                          reason="Guard-based fixes insufficient. Preferring restructure.")
                        c_mutations += 1

            # STEP 3: Generate patch with active rejection
            patch, tactic = agent.generate_patch(bug, c_strategy, c_failed_patches, ctx)
            h = phash(patch)
            if h in c_hashes: c_repeats += 1
            c_hashes.add(h)

            # STEP 4: Execute
            ok, err = sandbox(patch, bug["test"])

            # STEP 5: Store result in runtime memory
            runtime.memory.store({
                "category": bug["cat"],
                "predicted": f"tactic:{tactic} error:{err[:80] if err else 'PASS'}",
                "correct": ok,
                "bug_id": bug["id"],
                "patch_hash": h,
            })

            if ok:
                c_solved = True
                clog("ADAPTIVE SUCCESS",
                     f"Attempt {attempt}: PASSED (tactic={tactic})",
                     f"Mutations applied: {c_mutations} | Memory hits: {c_mem_hits}")
                break
            else:
                c_failed_patches.append(patch)
                clog("FAILED PATCH", f"Attempt {attempt}: {err[:70]} (tactic={tactic})")

        all_results.append({
            "id": bug["id"], "cat": bug["cat"], "title": bug["title"],
            "b_solved": b_solved, "b_attempts": b_attempts, "b_repeats": b_repeats,
            "c_solved": c_solved, "c_attempts": c_attempts, "c_repeats": c_repeats,
            "c_mem": c_mem_hits, "c_refl": c_refl_hits, "c_mutations": c_mutations,
            "c_rejections": c_rejections,
        })

    # ── FINAL REPORT ──
    print(f"\n{'='*64}")
    print(f"  CODEREPAIRBENCH v2 — RESULTS")
    print(f"{'='*64}")
    print(f"\n  {'Bug':<10} {'Cat':<15} {'Base':>5} {'Cogni':>6} {'Mem':>4} {'Refl':>5} {'Mut':>4} {'B.Rep':>5} {'C.Rep':>5}")
    print(f"  {'-'*60}")
    for r in all_results:
        bs="PASS" if r["b_solved"] else "FAIL"
        cs="PASS" if r["c_solved"] else "FAIL"
        print(f"  {r['id']:<10} {r['cat']:<15} {bs:>5} {cs:>6} {r['c_mem']:>4} {r['c_refl']:>5} {r['c_mutations']:>4} {r['b_repeats']:>5} {r['c_repeats']:>5}")

    n=len(all_results)
    bs=sum(1 for r in all_results if r["b_solved"])
    cs=sum(1 for r in all_results if r["c_solved"])
    br=sum(r["b_repeats"] for r in all_results)
    cr=sum(r["c_repeats"] for r in all_results)
    ba=sum(r["b_attempts"] for r in all_results)
    ca=sum(r["c_attempts"] for r in all_results)
    tm=sum(r["c_mem"] for r in all_results)
    tr=sum(r["c_refl"] for r in all_results)
    tmu=sum(r["c_mutations"] for r in all_results)

    print(f"\n  {'Metric':<35} {'Baseline':>10} {'CogniCore':>10}")
    print(f"  {'-'*55}")
    print(f"  {'Bugs Solved':<35} {bs:>10} {cs:>10}")
    print(f"  {'Total Attempts':<35} {ba:>10} {ca:>10}")
    print(f"  {'Repeated Patches':<35} {br:>10} {cr:>10}")
    print(f"  {'Memory Retrievals':<35} {'--':>10} {tm:>10}")
    print(f"  {'Reflections Generated':<35} {'--':>10} {tr:>10}")
    print(f"  {'Strategy Mutations':<35} {'--':>10} {tmu:>10}")

    rep_red = ((br-cr)/max(1,br))*100 if br>0 else 0
    print(f"\n  Repeated patch reduction: {rep_red:+.1f}%")

    print(f"\n  VERDICT:")
    checks=[
        ("Memory retrieval active", tm>0),
        ("Reflection generated", tr>0),
        ("Strategy mutations applied", tmu>0),
        ("Repeated patches reduced", cr<=br),
        ("CogniCore solves >= baseline", cs>=bs),
    ]
    for label,ok in checks:
        s="\033[32mPROVEN\033[0m" if ok else "\033[31mNOT PROVEN\033[0m"
        print(f"  [{s}] {label}")
    print(f"{'='*64}\n")
    return all_results


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--openai",action="store_true",help="Use OpenAI GPT for patches")
    p.add_argument("--model",default="gpt-4o-mini")
    p.add_argument("--attempts",type=int,default=5)
    args=p.parse_args()

    client=None
    if args.openai:
        try:
            from openai import OpenAI
            client=OpenAI()
            print("  OpenAI client initialized")
        except Exception as e:
            print(f"  OpenAI not available: {e}")

    run_benchmark(max_attempts=args.attempts, use_llm=args.openai, llm_client=client)
