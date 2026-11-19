"""Uploads files through the running backend's real API (as the UI does) and prints the verdict.
Usage: .venv-win\\Scripts\\python.exe experiments\\api_smoke_test.py file1 [file2 ...]
Creates/uses a throwaway local account smoketest@example.com."""
import sys, time, requests

B = "http://127.0.0.1:8001"
s = requests.Session()
cred = {"email": "smoketest@example.com", "name": "Smoke Test", "password": "smoketest123"}
if s.post(f"{B}/auth/login", json=cred).status_code != 200:
    s.post(f"{B}/auth/register", json=cred).raise_for_status()

for f in sys.argv[1:]:
    with open(f, "rb") as fh:
        job = s.post(f"{B}/jobs", files={"file": (f.split("\\")[-1].split("/")[-1], fh)}).json()
    jid = job.get("job_id") or job.get("id")
    for _ in range(300):
        j = s.get(f"{B}/jobs/{jid}").json()
        if j.get("status") in ("done", "error"):
            break
        time.sleep(1)
    v = j.get("verdict") or {}
    loc = (j.get("forensics") or {}).get("localization") or {}
    print(f"\n=== {f}\n  status={j.get('status')}  verdict={v.get('classification')}  score={v.get('final_score')}  "
          f"confidence={v.get('confidence_level')}  top={v.get('top_contributing_signal')}")
    if loc:
        print(f"  TruFor score={loc.get('score')} reliability={loc.get('reliability')} regions={[r['box'] for r in loc.get('regions', [])]}")
    for st in j.get("steps", []):
        print(f"  [{st['title']}] {st['detail'][:170]}")
    print("  explanation:", (j.get("explanation") or "")[:400])
