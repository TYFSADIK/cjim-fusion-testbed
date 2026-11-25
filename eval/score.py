"""Scores the fusion against the scenario's ground truth. Every number in the results comes from here.

  extraction     for free-text records about a real event: event type, place, entities right
  duplicates     re-sent copies and corrections recognised (precision, recall)
  time           error of each record's time against when it really happened, before and after
                 reconciliation (median and 90th percentile, minutes)
  association    for every pair of records about real events: same fused event exactly when they
                 describe the same real event (pairwise precision, recall, F1)
  events         fused events that match exactly one real event; real events found
  confidence     the band of events from real happenings against the fabricated rumour
"""
import json
import statistics
from collections import Counter
from datetime import datetime
from itertools import combinations


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def score(obs, events, truth, times_before=None):
    t_events = {e["id"]: e for e in truth["events"]}
    rec = {f"{r['source']}:{r['record_id']}": r for r in truth["records"]}
    out = {}

    # extraction, free text only, records about real events
    ex = [o for o in obs if o["source"] != "sensor" and rec[o["obs_id"]]["event_id"] and rec[o["obs_id"]]["kind"] != "correction"]
    right = Counter()
    for o in ex:
        te = t_events[rec[o["obs_id"]]["event_id"]]
        right["type"] += o["event_type"] == te["type"]
        right["place"] += o.get("place") == te["place"]
        right["entities"] += set(o["entities"]) == set(te["entities"])
    out["extraction"] = {"records": len(ex), **{k: round(right[k] / len(ex), 3) for k in ("type", "place", "entities")}}
    noise = [o for o in obs if rec[o["obs_id"]]["kind"] == "noise"]
    out["extraction"]["noise_recognised"] = round(sum(o["event_type"] is None for o in noise) / max(1, len(noise)), 3)

    # duplicates and corrections
    truly = {k for k, r in rec.items() if r["kind"] in ("duplicate", "correction")}
    marked = {o["obs_id"] for o in obs if o.get("duplicate_of")}
    out["duplicates"] = {"true": len(truly), "marked": len(marked), "precision": round(len(truly & marked) / max(1, len(marked)), 3),
                         "recall": round(len(truly & marked) / max(1, len(truly)), 3)}

    # time error, records about real events (not copies)
    def errs(times):
        e = {}
        for o in obs:
            r = rec[o["obs_id"]]
            if r["event_id"] and r["kind"] == "genuine" and times.get(o["obs_id"]) and o.get("time_basis") != "vague":
                e.setdefault(o["source"], []).append(abs((ts(times[o["obs_id"]]) - ts(r["true_time"])).total_seconds()) / 60)
        return {s: {"median_min": round(statistics.median(v), 1), "p90_min": round(pct(v, 0.9), 1), "n": len(v)} for s, v in sorted(e.items())}
    out["time_after"] = errs({o["obs_id"]: o.get("time") for o in obs})
    if times_before:
        out["time_before"] = errs(times_before)

    # association: pairs of records about real events (copies counted with their original)
    where = {}
    for e in events:
        for m in e["members"] + e["duplicates"]:
            where[m] = e["event_id"]
    real = [k for k, r in rec.items() if r["event_id"]]
    tp = fp = fn = 0
    for a, b in combinations(real, 2):
        same_truth = rec[a]["event_id"] == rec[b]["event_id"]
        same_fused = a in where and where.get(a) == where.get(b)
        tp += same_truth and same_fused
        fp += same_fused and not same_truth
        fn += same_truth and not same_fused
    p, r = tp / max(1, tp + fp), tp / max(1, tp + fn)
    out["association"] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / max(1e-9, p + r), 3)}

    # events: each fused event's majority real event; pure if all its members are from that event
    pure = 0
    found = set()
    rumour_events = []
    for e in events:
        ids = [rec[m]["event_id"] for m in e["members"]]
        maj, n = Counter(ids).most_common(1)[0]
        if maj and n == len(ids):
            pure += 1
        if maj:
            found.add(maj)
        if any(rec[m]["kind"] == "false" for m in e["members"]):
            rumour_events.append(e)
    observed = {r["event_id"] for r in rec.values() if r["event_id"]}
    out["events"] = {"fused": len(events), "pure": pure, "real_observed": len(observed), "real_found": len(found & observed)}

    # confidence bands: real events against the rumour
    bands = Counter()
    for e in events:
        ids = [rec[m]["event_id"] for m in e["members"]]
        if Counter(ids).most_common(1)[0][0]:
            bands[e["band"]] += 1
    out["confidence"] = {"real_events_by_band": dict(bands),
                         "rumour": [{"band": e["band"], "confidence": e["confidence"], "sources": e["sources"]} for e in rumour_events]}
    return out


if __name__ == "__main__":
    import sys
    d = json.load(open(sys.argv[1]))
    print(json.dumps(score(d["observations"], d["events"], json.load(open(sys.argv[2]))), indent=1))
