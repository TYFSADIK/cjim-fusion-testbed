"""The streaming pipeline.

  python -m fusion.pipeline produce [--speed 720]   replay the scenario's five sources into Redpanda
                                                    (one topic each, raw.<source>) in arrival order;
                                                    --speed 720 plays 72 hours in 6 minutes
  python -m fusion.pipeline normalize               consume raw.*, normalise (vLLM for free text),
                                                    embed (vLLM), store in PostGIS, OpenSearch, Qdrant
  python -m fusion.pipeline fuse                    every few seconds, when observations have changed:
                                                    dedup, reconcile times, associate, score confidence,
                                                    and (re)write the report of every event whose
                                                    evidence changed
"""
import argparse
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path

from confluent_kafka import Consumer, Producer

from fusion import llm, store
from fusion.fuse import fuse
from fusion.normalize import normalize
from fusion.report import verify, write

KAFKA = os.environ.get("KAFKA", "127.0.0.1:19092")
SOURCES = ["sensor", "patrol", "comms", "imagery", "osint"]
SCENARIO = Path(os.environ.get("SCENARIO", "data/scenario"))


def produce(speed):
    p = Producer({"bootstrap.servers": KAFKA})
    recs = [json.loads(l) for s in SOURCES for l in open(SCENARIO / "sources" / f"{s}.jsonl")]
    recs.sort(key=lambda r: r["received_at"])
    t_first = datetime.fromisoformat(recs[0]["received_at"].replace("Z", "+00:00"))
    wall0 = time.time()
    for r in recs:
        due = (datetime.fromisoformat(r["received_at"].replace("Z", "+00:00")) - t_first).total_seconds() / speed
        time.sleep(max(0.0, due - (time.time() - wall0)))
        p.produce(f"raw.{r['source']}", key=r["record_id"], value=json.dumps(r))
        p.poll(0)
    p.flush()
    print(f"produced {len(recs)} records in {time.time() - wall0:.0f} s")


def wait_for(check, what):
    while not check():
        print(f"waiting for {what}...", flush=True)
        time.sleep(5)


def normalize_loop():
    wait_for(lambda: llm.ready(), "vLLM")
    wait_for(lambda: llm.ready(llm.EMBED_URL), "vLLM embeddings")
    for _ in range(30):
        try:
            store.init()
            break
        except Exception as e:
            print("waiting for storage:", e, flush=True)
            time.sleep(5)
    store.ensure_collection(len(llm.embed(["probe"])[0]))
    c = Consumer({"bootstrap.servers": KAFKA, "group.id": "normalizer", "auto.offset.reset": "earliest",
                  "enable.auto.commit": False})
    c.subscribe([f"raw.{s}" for s in SOURCES])
    n = 0
    while True:
        m = c.poll(1.0)
        if m is None or m.error():
            continue
        rec = json.loads(m.value())
        t0 = time.time()
        o = normalize(rec)
        vec = llm.embed([o["text"]])[0] if o["text"] else None
        store.save_observation(o, vec)
        c.commit(m)
        n += 1
        print(f"{n} {o['obs_id']} {o['method']} {o.get('event_type')} {o.get('place')} {time.time() - t0:.2f}s", flush=True)


def members_hash(e, obs_by_id):
    return hashlib.sha1(json.dumps([(m, obs_by_id[m].get("time")) for m in sorted(e["members"])]).encode()).hexdigest()


def fuse_loop(period=5.0, once=False):
    for _ in range(60):
        try:
            store.init()
            break
        except Exception as e:
            print("waiting for storage:", e, flush=True)
            time.sleep(5)
    last = None
    while True:
        obs = store.load_observations()
        sig = (len(obs), obs[-1]["obs_id"] if obs else None)
        if obs and sig != last:
            last = sig
            t0 = time.time()
            # fusion always starts again from the stored observations, so every step can be redone
            emb = store.load_vectors([o["obs_id"] for o in obs])
            events, found = fuse(obs, emb)
            store.save_events(events, obs)
            by_id = {o["obs_id"]: o for o in obs}
            have = store.report_hashes()
            wrote = 0
            for e in events:
                h = members_hash(e, by_id)
                if have.get(e["event_id"]) == h:
                    continue      # evidence unchanged: keep the report and its review
                try:
                    rep = verify(write(e, by_id), e, by_id)
                except Exception as ex:
                    rep = {"sentences": [], "assessment": f"report not written: {ex}", "supported": 0, "total": 0, "status": "failed"}
                store.save_report(e["event_id"], h, rep)
                wrote += 1
            gone = store.supersede_reports(e["event_id"] for e in events)
            store.log("fuse", {"observations": len(obs), "events": len(events), "reports_written": wrote, "superseded": gone,
                               "time_findings": found, "seconds": round(time.time() - t0, 1)})
            print(f"fused {len(obs)} observations into {len(events)} events; {wrote} reports written, {gone} superseded; {found}", flush=True)
        if once:
            return
        time.sleep(period)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["produce", "normalize", "fuse"])
    ap.add_argument("--speed", type=float, default=720)
    a = ap.parse_args()
    {"produce": lambda: produce(a.speed), "normalize": normalize_loop, "fuse": fuse_loop}[a.cmd]()
