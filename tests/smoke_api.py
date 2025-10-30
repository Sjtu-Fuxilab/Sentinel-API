# tests/smoke_api.py — Sentinel-API smoke tests (in-process or HTTP if BASE_URL is set)
import os, sys, json, time, threading, queue, traceback
from pathlib import Path
def _need(m,p=None):
    try: return __import__(m)
    except ImportError:
        import subprocess; subprocess.check_call([sys.executable,"-m","pip","install","-q",p or m]); return __import__(m)
requests = _need("requests"); httpx = _need("httpx")
try:
    from fastapi.testclient import TestClient  # type: ignore
except Exception:
    _need("starlette"); from fastapi.testclient import TestClient  # type: ignore

BASE_URL = os.environ.get("BASE_URL")  # e.g. http://127.0.0.1:8000 for remote mode
PREDICT_PAYLOAD = {"age":65,"gender":"M","hr_mean":85,"sbp_mean":120}
CONCURRENT_REQUESTS = int(os.environ.get("SMOKE_CONCURRENCY","10"))
OUT_JSON = "api_test_report.json"
OUT_JUNIT = "api_test_report.junit.xml"

def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f: json.dump(obj, f, indent=2, ensure_ascii=False)

def write_junit_xml(path, results):
    import xml.etree.ElementTree as ET
    steps = results["steps"]; failures = sum(1 for s in steps if s["status"] not in ("passed","skipped"))
    ts = ET.Element("testsuite", {"name":"sentinel-api-smoke","tests":str(len(steps)),"failures":str(failures),"time":f'{results.get("elapsed_sec",0):.3f}'})
    for s in steps:
        tc = ET.SubElement(ts, "testcase", {"classname":"smoke","name":s["name"],"time":f'{s.get("elapsed_sec",0):.3f}'})
        if s["status"] not in ("passed","skipped"):
            fl = ET.SubElement(tc, "failure", {"message":s.get("error","failure")}); fl.text = s.get("details","")
    ET.ElementTree(ts).write(path, encoding="utf-8", xml_declaration=True)

def step(fn):
    def wrapper(name, ctx):
        t0 = time.time(); rec = {"name":name,"status":"passed","elapsed_sec":0.0}
        try: fn(ctx, rec)
        except Exception as e:
            rec["status"]="failed"; rec["error"]=f"{type(e).__name__}: {e}"; rec["details"]=traceback.format_exc()
        finally:
            rec["elapsed_sec"]=time.time()-t0; ctx["steps"].append(rec)
    return wrapper

# transport selection
client = None
if BASE_URL:
    mode = f"http ({BASE_URL})"
else:
    # import app for in-process tests
    here = Path.cwd(); probe = here; found=False
    for _ in range(6):
        if (probe / "api" / "main.py").exists():
            sys.path.insert(0, str(probe)); found=True; break
        probe = probe.parent
    if not found: sys.path.insert(0, str(here))
    from api.main import app  # type: ignore
    client = TestClient(app); mode = "in-process TestClient"

@step
def test_root(ctx, rec):
    if BASE_URL:
        r = requests.get(f"{BASE_URL.rstrip('/')}/"); r.raise_for_status(); js = r.json()
    else:
        r = client.get("/"); assert r.status_code==200, r.text; js = r.json()
    assert "message" in js; rec["response"]=js

@step
def test_health(ctx, rec):
    if BASE_URL:
        r = requests.get(f"{BASE_URL.rstrip('/')}/health"); r.raise_for_status(); js=r.json()
    else:
        r = client.get("/health"); assert r.status_code==200, r.text; js=r.json()
    assert isinstance(js,dict) and any(k in js for k in ("status","ok","healthy")); rec["response"]=js

@step
def test_version_optional(ctx, rec):
    try:
        if BASE_URL:
            r = requests.get(f"{BASE_URL.rstrip('/')}/version", timeout=5)
            if r.status_code==404: rec["status"]="skipped"; rec["details"]="no /version"; return
            r.raise_for_status(); js=r.json()
        else:
            r = client.get("/version"); 
            if r.status_code==404: rec["status"]="skipped"; rec["details"]="no /version"; return
            assert r.status_code==200, r.text; js=r.json()
        assert isinstance(js,dict); rec["response"]=js
    except Exception as e:
        rec["status"]="skipped"; rec["details"]=f"/version unavailable: {e}"

@step
def test_predict(ctx, rec):
    if BASE_URL:
        r = requests.post(f"{BASE_URL.rstrip('/')}/predict", json=PREDICT_PAYLOAD); r.raise_for_status(); js=r.json()
    else:
        r = client.post("/predict", json=PREDICT_PAYLOAD); assert r.status_code==200, r.text; js=r.json()
    assert isinstance(js,dict); 
    if "probability" in js: assert 0.0 <= float(js["probability"]) <= 1.0
    if "confidence" in js:  assert 0.0 <= float(js["confidence"])  <= 1.0
    if "risk_category" in js: assert isinstance(js["risk_category"], str)
    rec["response"]=js

@step
def test_concurrency(ctx, rec):
    lat=[],[]
    latencies, errs = lat
    q=queue.Queue(); [q.put(1) for _ in range(CONCURRENT_REQUESTS)]
    def worker():
        while True:
            try: q.get_nowait()
            except: return
            t=time.time()
            try:
                if BASE_URL:
                    rr=requests.post(f"{BASE_URL.rstrip('/')}/predict", json=PREDICT_PAYLOAD, timeout=10); rr.raise_for_status()
                else:
                    rr=client.post("/predict", json=PREDICT_PAYLOAD); assert rr.status_code==200, rr.text
                latencies.append(time.time()-t)
            except Exception as e:
                errs.append(str(e))
            finally:
                q.task_done()
    threads=[threading.Thread(target=worker) for _ in range(min(CONCURRENT_REQUESTS,8))]
    [t.start() for t in threads]; [t.join(timeout=60) for t in threads]
    rec["latencies_sec"]=latencies; rec["errors"]=errs
    assert not errs, f"errors: {errs}"
    if latencies:
        s=sorted(latencies); rec["p50_ms"]=round(1000*s[len(s)//2],1); rec["p95_ms"]=round(1000*s[int(0.95*(len(s)-1))],1)

def main():
    ctx={"steps":[]}; t0=time.time()
    for name, fn in [("GET /",test_root),("GET /health",test_health),("GET /version (optional)",test_version_optional),("POST /predict",test_predict),(f"POST /predict x{CONCURRENT_REQUESTS}", test_concurrency)]:
        fn(name, ctx)
    elapsed=time.time()-t0
    failed=sum(1 for s in ctx["steps"] if s["status"] not in ("passed","skipped"))
    report={"mode":mode,"elapsed_sec":round(elapsed,3),"timestamp":time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),"payload_sample":PREDICT_PAYLOAD,"steps":ctx["steps"],"summary":{"total":len(ctx["steps"]),"passed":sum(1 for s in ctx["steps"] if s["status"]=="passed"),"skipped":sum(1 for s in ctx["steps"] if s["status"]=="skipped"),"failed":failed}}
    write_json(OUT_JSON, report); write_junit_xml(OUT_JUNIT, report)
    for s in ctx["steps"]:
        mark = "✅" if s["status"]=="passed" else ("⏭️" if s["status"]=="skipped" else "❌")
        extra = f" (p50={s.get('p50_ms','-')} ms, p95={s.get('p95_ms','-')} ms)" if "p50_ms" in s else ""
        print(f"{mark} {s['name']} — {s['status']}{extra}")
    print(f"\nArtifacts:\n  JSON : {OUT_JSON}\n  JUnit: {OUT_JUNIT}")
    if failed: sys.exit(1)

if __name__ == "__main__":
    main()
