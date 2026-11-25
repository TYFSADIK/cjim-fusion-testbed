"""From normalised observations to fused events, each step recorded on the observations it changes.

  dedup        re-sent copies merge into the original; a correction supersedes the time it corrects
  align_times  per source, the delay between "when it happened" and "when it arrived" is learned;
               a source whose stated times arrive before they happened has a fast clock (corrected by
               the median), and a single report whose delay is off by the local-time offset was written
               in local time but labelled Z (corrected, and flagged)
  associate    observations join an event when time, place, type and entities agree, with meaning
               (embedding similarity) as evidence where place or type is missing
  confidence   independent sources combine: 1 - product of (1 - weight of each source's best report),
               where the weight comes from its NATO grading; copies from one source count once
"""
import hashlib
import statistics
from datetime import datetime, timedelta, timezone

from fusion.reference import CREDIBILITY_WEIGHT, RELIABILITY_WEIGHT, km

LOCAL_OFFSET = timedelta(hours=4)
TIME_MARGIN = timedelta(minutes=60)      # how far outside an event's span an observation may fall
LONG_MARGIN = timedelta(hours=3)         # ...when place and type already agree (a long loiter)
MAX_SPAN = timedelta(hours=8)            # no single event lasts longer
UNTYPED_SEM = 0.75                       # embedding similarity an untyped report needs to join on place alone
COMPATIBLE = {frozenset({"vessel_loiter", "vessel_transit"})}   # a radar cannot always tell these apart


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def iso(t):
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if t else None


def shift(o, delta, why):
    for k in ("time", "time_lo", "time_hi"):
        if o.get(k):
            o[k] = iso(ts(o[k]) + delta)
    o.setdefault("adjustments", []).append(why)


# ------------------------------------------------------------------ duplicates and corrections

def dedup(obs):
    by_id = {o["obs_id"]: o for o in obs}
    seen = {}
    for o in sorted(obs, key=lambda o: o["received_at"]):
        if o.get("correction_of") and o["correction_of"] in by_id:
            orig = by_id[o["correction_of"]]
            if o.get("time"):
                for k in ("time", "time_lo", "time_hi", "time_basis"):
                    orig[k] = o[k]
                orig.setdefault("adjustments", []).append(f"time corrected by {o['obs_id']}")
            o["duplicate_of"] = orig["obs_id"]
            continue
        # the same text with the same stated time: a re-sent copy (the same words on another day are not)
        key = (o["source"], " ".join(o["text"].lower().split()), o.get("time"))
        if o["text"] and key in seen:
            o["duplicate_of"] = seen[key]
        else:
            seen[key] = o["obs_id"]
    return obs


# ------------------------------------------------------------------ time

def align_times(obs):
    """Learns each source's reporting delay and fixes what does not fit. Returns what was found."""
    found = {}
    for src in sorted({o["source"] for o in obs}):
        mine = [o for o in obs if o["source"] == src and o.get("time") and o["time_basis"] != "vague" and not o.get("duplicate_of")]
        delays = [(ts(o["received_at"]) - ts(o["time"])).total_seconds() for o in mine]
        if len(delays) < 5:
            continue
        med = statistics.median(delays)
        if med < -30:      # stated times after arrival: the source's clock runs fast
            for o in mine:
                shift(o, timedelta(seconds=med), f"{src} clock {-med / 60:.1f} min fast")
            found[src] = {"clock_offset_s": round(-med, 1)}
            delays = [d - med for d in delays]
            med = 0
        mad = statistics.median(abs(d - med) for d in delays) * 1.4826 or 60
        fixed = 0
        for o, d in zip(mine, delays):
            # a delay off by about the local-time offset, which the offset puts back in range
            if o["time_basis"] == "labelled_z" and abs(d - med) > 5 * mad and abs(d - LOCAL_OFFSET.total_seconds() - med) < 5 * mad:
                shift(o, LOCAL_OFFSET, "labelled Z but written in local time")
                o["time_basis"] = "local"
                fixed += 1
        if fixed:
            found.setdefault(src, {})["local_time_labelled_z"] = fixed
    return found


# ------------------------------------------------------------------ association

def type_ok(a, b):
    return a is None or b is None or a == b or frozenset({a, b}) in COMPATIBLE


def place_ok(c, o):
    if o.get("lat") is None or c["lat"] is None:
        return None                                    # unknown: neither for nor against
    if o.get("place") and c["place"]:
        return o["place"] == c["place"]
    return km(o["lat"], o["lon"], c["lat"], c["lon"]) <= max(3.0, (o.get("radius_km") or 1) + 2)


def ents_ok(c, o):
    if not o["entities"] or not c["entities"]:
        return None
    return bool(set(o["entities"]) & c["entities"])


def cos(a, b):
    return sum(x * y for x, y in zip(a, b)) if a and b else 0.0


def score(c, o, emb):
    lo, hi = ts(o["time_lo"]), ts(o["time_hi"])
    same_where_what = o.get("place") and o.get("place") == c["place"] and o["event_type"] and o["event_type"] == c["type"]
    margin = LONG_MARGIN if same_where_what else TIME_MARGIN
    if hi < c["start"] - margin or lo > c["end"] + margin:
        return None
    if max(c["end"], hi) - min(c["start"], lo) > MAX_SPAN and o["time_basis"] != "vague":
        return None
    if not type_ok(c["type"], o["event_type"]):
        return None
    p, e = place_ok(c, o), ents_ok(c, o)
    if p is False or e is False:
        return None
    sem = max((cos(emb.get(o["obs_id"]), emb.get(m)) for m in c["members"][-5:]), default=0.0)
    s = 0.4 * (p is True) + 0.25 * (o["event_type"] is not None and o["event_type"] == c["type"]) + 0.2 * (e is True) + 0.3 * sem
    # with nothing concrete in common, meaning alone is not enough
    if p is not True and e is not True:
        return None
    # a report whose type is unknown corroborates an event only through a shared vessel, aircraft or
    # unit, or when it says nearly the same thing; place alone would let any untyped report confirm
    # whatever else happened there
    if (o["event_type"] is None) != (c["type"] is None) and e is not True and sem < UNTYPED_SEM:
        return None
    return s


def new_cluster(o):
    return {"members": [o["obs_id"]], "start": ts(o["time_lo"]) if o["time_basis"] == "vague" else ts(o["time"]),
            "end": ts(o["time_hi"]) if o["time_basis"] == "vague" else ts(o["time"]),
            "type": o["event_type"], "place": o.get("place"), "lat": o.get("lat"), "lon": o.get("lon"),
            "entities": set(o["entities"]), "precise": o["time_basis"] != "vague"}


def absorb(c, o):
    c["members"].append(o["obs_id"])
    if o["time_basis"] != "vague":
        t = ts(o["time"])
        if not c["precise"]:
            c["start"] = c["end"] = t
            c["precise"] = True
        c["start"], c["end"] = min(c["start"], t), max(c["end"], t)
    c["type"] = c["type"] or o["event_type"]
    if c["place"] is None and o.get("place"):
        c["place"], c["lat"], c["lon"] = o["place"], o["lat"], o["lon"]
    c["entities"] |= set(o["entities"])


def associate(obs, emb=None, threshold=0.45):
    """Greedy clustering in order of time: precise observations first, so vague ones join events
    whose time is already pinned down."""
    emb = emb or {}
    live = [o for o in obs if not o.get("duplicate_of") and o.get("time")]
    order = sorted(live, key=lambda o: (o["time_basis"] == "vague", o["time"]))
    clusters = []
    for o in order:
        best, bs = None, threshold
        for c in clusters:
            s = score(c, o, emb)
            if s is not None and s >= bs:
                best, bs = c, s
        if best:
            absorb(best, o)
        else:
            clusters.append(new_cluster(o))
    return clusters


# ------------------------------------------------------------------ confidence

def weight(o):
    return RELIABILITY_WEIGHT[o["reliability"]] * CREDIBILITY_WEIGHT[o["credibility"]]


def confidence(members):
    best = {}
    for o in members:
        best[o["source"]] = max(best.get(o["source"], 0), weight(o))
    p = 1.0
    for w in best.values():
        p *= 1 - w
    c = 1 - p
    return round(c, 3), sorted(best), "high" if c >= 0.85 else "medium" if c >= 0.6 else "low"


def events_from(clusters, obs):
    by_id = {o["obs_id"]: o for o in obs}
    out = []
    for i, c in enumerate(sorted(clusters, key=lambda c: c["start"])):
        members = [by_id[m] for m in c["members"]]
        if c["type"] is None and len(members) == 1:
            continue                                   # routine traffic: kept as an observation only
        conf, sources, band = confidence(members)
        counts = [o["count"] for o in members if o.get("count") is not None]
        # an event's ID comes from its earliest observation, so it survives re-fusion and reviews stay attached
        first = min(c["members"], key=lambda m: (by_id[m]["received_at"], m))
        out.append({"event_id": "EV-" + hashlib.sha1(first.encode()).hexdigest()[:8], "type": c["type"], "place": c["place"], "lat": c["lat"], "lon": c["lon"],
                    "start": iso(c["start"]), "end": iso(c["end"]), "entities": sorted(c["entities"]),
                    "count": statistics.median(counts) if counts else None,
                    "count_conflict": len(set(counts)) > 1, "confidence": conf, "band": band, "sources": sources,
                    "members": c["members"],
                    "duplicates": sorted(o["obs_id"] for o in obs if o.get("duplicate_of") in c["members"])})
    return out


def fuse(obs, emb=None):
    dedup(obs)
    found = align_times(obs)
    clusters = associate(obs, emb)
    return events_from(clusters, obs), found
