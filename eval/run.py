"""Scores a whole run against ground truth, offline (directly against vLLM) or from a live pipeline's
database, and writes the results.

    python -m eval.run offline [--out results/offline]     normalise with vLLM and with the keyword
                                                          baseline, fuse, write and check reports
    python -m eval.run live [--out results/live]          score what the running pipeline stored
    python -m eval.run recheck                             re-run the (deterministic) report checker on
                                                          the reports of the last offline run
"""
import argparse
import json
import statistics
import time
from pathlib import Path

from eval.score import score
from fusion import llm
from fusion.fuse import fuse
from fusion.normalize import normalize
from fusion.report import verify, write

SOURCES = ["sensor", "patrol", "comms", "imagery", "osint"]


def load_records(scenario):
    return [json.loads(l) for s in SOURCES for l in open(Path(scenario) / "sources" / f"{s}.jsonl")]


def report_stats(reports):
    total = sum(r["total"] for r in reports)
    sup = sum(r["supported"] for r in reports)
    problems = {}
    for r in reports:
        for s in r["sentences"]:
            for p in s.get("problems", []):
                k = p.split(" is ")[0].split(" ")[0] if " is not" in p else p.split(":")[0]
                problems[k] = problems.get(k, 0) + 1
    return {"reports": len(reports), "sentences": total, "supported": sup,
            "supported_share": round(sup / max(1, total), 3),
            "reports_fully_supported": sum(r["supported"] == r["total"] and r["total"] > 0 for r in reports),
            "unsupported_by_kind": problems}


def offline(scenario, out):
    truth = json.load(open(Path(scenario) / "truth.json"))
    recs = load_records(scenario)
    res = {}
    for mode in ("rules", "llm"):
        t0 = time.time()
        obs = [normalize(r, use_llm=(mode == "llm")) for r in recs]
        t_norm = time.time() - t0
        before = {o["obs_id"]: o.get("time") for o in obs}
        emb = {}
        if mode == "llm":
            texts = [(o["obs_id"], o["text"]) for o in obs if o["text"]]
            for i in range(0, len(texts), 32):
                vecs = llm.embed([t for _, t in texts[i:i + 32]])
                emb.update({k: v for (k, _), v in zip(texts[i:i + 32], vecs)})
        events, found = fuse(obs, emb)
        s = score(obs, events, truth, before)
        s["time_findings"] = found
        s["normalise_seconds"] = round(t_norm, 1)
        res[mode] = s
        if mode == "llm":
            by_id = {o["obs_id"]: o for o in obs}
            reports, lat = [], []
            for e in events:
                t1 = time.time()
                reports.append({"event_id": e["event_id"], **verify(write(e, by_id), e, by_id)})
                lat.append(time.time() - t1)
            res["reports"] = report_stats(reports)
            res["reports"]["seconds_median"] = round(statistics.median(lat), 2)
            res["llm_usage"] = dict(llm.stats)
            json.dump({"observations": obs, "events": events, "reports": reports}, open(out / "state.json", "w"), indent=1)
    json.dump(res, open(out / "results.json", "w"), indent=1)
    return res


def live(out):
    from fusion import store
    truth = json.load(open("data/scenario/truth.json"))
    with store.conn() as c:
        obs = [r[0] for r in c.execute("SELECT doc FROM observations").fetchall()]
        events = [r[0] for r in c.execute("SELECT doc FROM events").fetchall()]
        reports = [r[0] for r in c.execute("SELECT report FROM reports WHERE status <> 'superseded'").fetchall()]
        log = c.execute("SELECT detail FROM pipeline_log WHERE stage='fuse' ORDER BY t DESC LIMIT 1").fetchone()
    # fusion state (duplicates, adjusted times) lives in the events; re-derive it for scoring
    _, found = fuse(obs)
    res = score(obs, events, truth)
    res["reports"] = report_stats(reports)
    res["time_findings"] = found
    res["last_fusion"] = log[0] if log else None
    json.dump(res, open(out / "results.json", "w"), indent=1)
    return res


def recheck(out):
    st = json.load(open(out / "state.json"))
    by_id = {o["obs_id"]: o for o in st["observations"]}
    ev = {e["event_id"]: e for e in st["events"]}
    reps = [{"event_id": r["event_id"], **verify({"sentences": [{"text": s["text"], "cites": s["cites"]} for s in r["sentences"]],
                                                    "assessment": r["assessment"]}, ev[r["event_id"]], by_id)} for r in st["reports"]]
    res = json.load(open(out / "results.json"))
    secs = res["reports"].get("seconds_median")
    res["reports"] = report_stats(reps)
    res["reports"]["seconds_median"] = secs
    st["reports"] = reps
    json.dump(st, open(out / "state.json", "w"), indent=1)
    json.dump(res, open(out / "results.json", "w"), indent=1)
    return res["reports"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["offline", "live", "recheck"])
    ap.add_argument("--scenario", default="data/scenario")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out or f"results/{'offline' if a.mode == 'recheck' else a.mode}")
    out.mkdir(parents=True, exist_ok=True)
    r = {"offline": lambda: offline(a.scenario, out), "live": lambda: live(out), "recheck": lambda: recheck(out)}[a.mode]()
    print(json.dumps(r, indent=1)[:4000])
