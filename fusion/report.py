"""Evidence-linked summaries: vLLM writes each fused event as sentences, every sentence citing the
observations it rests on; a deterministic checker then verifies each sentence against what it cites.

A sentence fails the check if it cites nothing, cites an observation outside the event, or states a
time, a count, a place or a named vessel, aircraft or unit that none of its cited observations
contain. Failing sentences are flagged for the analyst. Nothing is final until an analyst approves it.
"""
import json
import re
from datetime import datetime, timedelta, timezone

from fusion import llm
from fusion.reference import GAZETTEER, KNOWN

REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "sentences": {"type": "array", "minItems": 1, "maxItems": 5, "items": {
            "type": "object",
            "properties": {"text": {"type": "string"}, "cites": {"type": "array", "items": {"type": "string"}, "minItems": 1}},
            "required": ["text", "cites"], "additionalProperties": False}},
        "assessment": {"type": "string"},
    },
    "required": ["sentences", "assessment"],
    "additionalProperties": False,
}

SYSTEM = """You are an intelligence analyst writing a short, factual event summary for a fictional exercise.
Use only the observations given. Write 2 to 4 sentences. Every sentence must cite, in "cites", the
observation IDs that support it, exactly as given at the start of each observation line. Give times in
UTC as HHMMZ, copied from the observation you cite (the "16 Nov 0309Z" at the start of a line means
day 16, time 0309Z).
Do not add facts that are not in the observations. In "assessment", state in one sentence how well
the reporting is corroborated, using the source gradings given."""


def evidence_block(event, obs_by_id):
    lines = []
    for m in event["members"]:
        o = obs_by_id[m]
        t = datetime.fromisoformat(o["time"].replace("Z", "+00:00")).strftime("%d %b %H%MZ") if o.get("time") else "time unknown"
        facts = ", ".join(x for x in [o.get("event_type") or "", o.get("place") or "",
                                      " ".join(KNOWN[e][1][0] for e in o["entities"]),
                                      f"count {o['count']}" if o.get("count") is not None else ""] if x)
        lines.append(f'{m} [{o["source"]}, graded {o["reliability"]}{o["credibility"]}] {t}: "{o["text"]}" ({facts})')
    return "\n".join(lines)


def schema_for(event):
    """The report schema with citations limited to this event's own observations, so vLLM's
    structured output cannot cite anything else."""
    sch = json.loads(json.dumps(REPORT_SCHEMA))
    sch["properties"]["sentences"]["items"]["properties"]["cites"]["items"] = {"type": "string", "enum": list(event["members"])}
    return sch


def write(event, obs_by_id):
    user = (f"Event type: {event['type']}. Place: {event['place']}. Confidence {event['confidence']} ({event['band']}), "
            f"{len(event['sources'])} independent source{'s' if len(event['sources']) != 1 else ''}: {', '.join(event['sources'])} "
            f"(several reports from one source are not independent).\nObservations:\n{evidence_block(event, obs_by_id)}")
    return llm.chat_json(SYSTEM, user, schema_for(event), max_tokens=500)


# ------------------------------------------------------------------ verification

def _times(o):
    """Every way a cited observation's time may legitimately be written: HHMM in UTC (as normalised),
    and the time as the source wrote it."""
    out = set()
    for k in ("time", "time_lo", "time_hi"):
        if o.get(k):
            t = datetime.fromisoformat(o[k].replace("Z", "+00:00"))
            for d in (-2, -1, 0, 1, 2):
                out.add((t + timedelta(minutes=d)).strftime("%H%M"))
    out.update(re.findall(r"(?<!\d)(\d{4})(?!\d)", o.get("text", "")))   # "1626Z", "2309 local"
    return out


def _places(o):
    names = set()
    if o.get("place"):
        names.add(o["place"].lower())
        names.update(a.lower() for a in GAZETTEER[o["place"]][2])
    return names


def _entity_names(o):
    return {n.lower() for e in o["entities"] for n in KNOWN[e][1]}


def check_sentence(s, event, obs_by_id):
    """The problems with one sentence, as a list of short explanations (empty: supported)."""
    problems = []
    cites = s.get("cites") or []
    if not cites:
        return ["cites nothing"]
    outside = [c for c in cites if c not in event["members"]]
    if outside:
        problems.append(f"cites observations outside this event: {', '.join(outside)}")
    cited = [obs_by_id[c] for c in cites if c in obs_by_id and c in event["members"]]
    if not cited:
        return problems or ["cites nothing that exists"]
    text = s["text"]
    low = text.lower()
    times = set().union(*(_times(o) for o in cited))
    for hhmm in re.findall(r"\b(\d{4})\s*Z\b", text) + re.findall(r"\b(\d{2}):(\d{2})\b", text):
        h = hhmm if isinstance(hhmm, str) else "".join(hhmm)
        if h not in times:
            problems.append(f"time {h} is not in the cited observations")
    counts = {str(o["count"]) for o in cited if o.get("count") is not None} | set(re.findall(r"\b\d+\b", " ".join(o["text"] for o in cited)))
    for n in re.findall(r"\b(\d{1,3})\s+(?:vehicles|containers|trucks|lifts|loads|crates|boxes)\b", low):
        if n not in counts:
            problems.append(f"count {n} is not in the cited observations")
    places = set().union(*(_places(o) for o in cited))
    for name, (_, _, aliases) in GAZETTEER.items():
        if name.lower() in low and name.lower() not in places:
            problems.append(f"place {name} is not in the cited observations")
    # corroboration: a sentence may not claim more independent sources than the event has
    # ("two independent posts" from one source), but may say there are too few ("lack of independent sources")
    n_src = len(event.get("sources") or {obs_by_id[m].get("source") for m in event["members"] if m in obs_by_id})
    for m in re.finditer(r"\b(two|three|four|five|several|multiple|\d+)\s+(?:separate\s+|different\s+)?independent\b", low):
        if re.search(r"\b(lack|lacks|lacking|no|not|without|limited|absence|few|fewer|only)\b", low[max(0, m.start() - 30):m.start()]):
            continue
        n = {"two": 2, "three": 3, "four": 4, "five": 5, "several": 2, "multiple": 2}.get(m.group(1)) or int(m.group(1))
        if n > n_src:
            problems.append(f"claims {m.group(1)} independent sources; the event has {n_src}")
    names = set().union(*(_entity_names(o) for o in cited))
    cited_text = " ".join(o["text"].lower() for o in cited)
    for eid, (_, enames, _) in KNOWN.items():
        for n in enames:
            # whole words only, so a short name like "Red 1" counts and "Tern" does not match "eastern"
            if re.search(rf"\b{re.escape(n.lower())}\b", low) and n.lower() not in names \
                    and not any(n.lower() in x for x in names) and n.lower() not in cited_text:
                problems.append(f"{n} is not in the cited observations")
                break
    return problems


def verify(report, event, obs_by_id):
    out = []
    for s in report["sentences"]:
        p = check_sentence(s, event, obs_by_id)
        out.append({**s, "supported": not p, "problems": p})
    return {"sentences": out, "assessment": report["assessment"],
            "supported": sum(x["supported"] for x in out), "total": len(out),
            "status": "ready for review" if all(x["supported"] for x in out) else "needs attention"}
