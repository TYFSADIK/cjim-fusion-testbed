"""Every source's record becomes one observation in a common schema:

  obs_id, source, record_id, received_at, grading (reliability, credibility)
  time (UTC best estimate), time_lo, time_hi (uncertainty), time_basis
  place (gazetteer name), lat, lon, radius_km
  event_type, entities (known IDs), count, text, method (how the fields were obtained)
  duplicate_of / correction_of (filled by dedup)

Structured fields are taken as they are. Free text goes to vLLM, which extracts the event type,
entity mentions, place mention and count under a JSON schema; deterministic code then resolves
mentions to the reference data and converts each source's time format to UTC."""
import re
from datetime import datetime, timedelta, timezone

from fusion import llm
from fusion.reference import (AIS, CALLSIGNS, EVENT_TYPES, GAZETTEER, GRADING, find_place_in, nearest_place,
                              resolve_entity, resolve_place)

LOCAL = timezone(timedelta(hours=-4))
MONTHS = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
PARTS = {"early morning": (0, 9), "morning": (5, 12), "afternoon": (9, 17), "evening": (17, 24)}

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        # a plain restatement first: a small model classifies better after saying what happened
        "what_happened": {"type": "string", "maxLength": 160},
        "event_type": {"type": "string", "enum": EVENT_TYPES + ["none"]},
        "entities": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "place": {"type": ["string", "null"]},
        "count": {"type": ["integer", "null"]},
    },
    "required": ["what_happened", "event_type", "entities", "place", "count"],
    "additionalProperties": False,
}
EXTRACT_SYSTEM = """You extract facts from short reports in a fictional military exercise. Return JSON only.
what_happened: one short plain sentence saying what was seen, in your own words.
event_type, one of:
  vessel_loiter: a ship staying in one area: holding position, loitering, circling, drifting, sitting off a place
  vessel_transit: a ship passing through or going by a place without stopping
  convoy_move: a group of vehicles or trucks moving or leaving a place
  air_activity: an aircraft or helicopter flying, orbiting, passing or heard overhead
  drone_sighting: a drone, UAS, quadcopter, or any small flying object with propellers or blinking lights
  gps_interference: GPS, GNSS or navigation positions lost, jammed, wrong or wandering
  power_outage: electricity lost: power out, lights went dark, running on generators or diesel
  port_activity: cargo being loaded or unloaded at a port: containers, crates, boxes, lifts, cranes
  vessel_incident: a ship aground, in a collision, on fire or leaking
  none: only for routine items with no activity to report: radio checks, weather, shift changes, scheduled ferries, chatter
entities: names of vessels, aircraft or units exactly as written (not places, not people, not generic words like "a ship")
place: the location exactly as written, or null
count: the number of vehicles, trucks, containers, crates, boxes, loads or lifts if a number is stated, else null

Examples:
"Green 9 rolled out of Dunmore in 7 trucks" -> {"what_happened": "A unit left Dunmore in seven trucks", "event_type": "convoy_move", "entities": ["Green 9"], "place": "Dunmore", "count": 7}
"nav displays going haywire around Westcott" -> {"what_happened": "Navigation equipment is failing near Westcott", "event_type": "gps_interference", "entities": [], "place": "Westcott", "count": null}
"Radio check, all stations loud and clear" -> {"what_happened": "A routine radio check", "event_type": "none", "entities": [], "place": null, "count": null}"""


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def iso(t):
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if t else None


def base(rec):
    r, c = GRADING[rec["source"]]
    return {"obs_id": f"{rec['source']}:{rec['record_id']}", "source": rec["source"], "record_id": rec["record_id"],
            "received_at": rec["received_at"], "reliability": r, "credibility": c, "time": None, "time_lo": None,
            "time_hi": None, "time_basis": None, "place": None, "lat": None, "lon": None, "radius_km": None,
            "event_type": None, "entities": [], "count": None, "text": "", "method": "", "duplicate_of": None,
            "correction_of": None, "raw": rec["payload"]}


def set_time(o, t, minutes, basis):
    o["time"], o["time_lo"], o["time_hi"] = iso(t), iso(t - timedelta(minutes=minutes)), iso(t + timedelta(minutes=minutes))
    o["time_basis"] = basis


def set_place(o, name, radius=3.0):
    if name:
        o["place"] = name
        o["lat"], o["lon"] = GAZETTEER[name][0], GAZETTEER[name][1]
        o["radius_km"] = radius


def patrol_time(text, received):
    """'At approx 2309 local on 15 Nov' or a correction's 'time should read 2339 local on 15 Nov'.
    Patrols report in local time (UTC-4); a 'Z' suffix is kept as written and checked later."""
    m = re.search(r"(\d{4})\s*(Z|local)\s+on\s+(\d{1,2})\s+([A-Z][a-z]{2})", text)
    if not m:
        return None, None
    hhmm, zone, day, mon = m.groups()
    t = datetime(received.year, MONTHS[mon], int(day), int(hhmm[:2]), int(hhmm[2:]),
                 tzinfo=timezone.utc if zone == "Z" else LOCAL)
    return t, zone


def osint_window(text, received):
    """'(Tuesday afternoon)' becomes that part of the most recent such day, local time."""
    m = re.search(r"\((\w+day)\s+(early morning|morning|afternoon|evening)\)", text)
    if not m:
        return None
    day, part = m.groups()
    loc = received.astimezone(LOCAL)
    back = (loc.weekday() - DAYS.index(day)) % 7
    d = (loc - timedelta(days=back)).date()
    h0, h1 = PARTS[part]
    lo = datetime(d.year, d.month, d.day, h0, tzinfo=LOCAL)
    hi = datetime(d.year, d.month, d.day, 23, 59, tzinfo=LOCAL) if h1 == 24 else datetime(d.year, d.month, d.day, h1, tzinfo=LOCAL)
    return lo, hi


def extract(o, text, use_llm=True):
    """Event type, entities, place and count from free text: vLLM, then reference resolution."""
    if use_llm:
        x = llm.chat_json(EXTRACT_SYSTEM, text, EXTRACT_SCHEMA, max_tokens=160)
        o["method"] = "llm"
    else:
        x = rules_extract(text)
        o["method"] = "rules"
    o["event_type"] = None if x["event_type"] == "none" else x["event_type"]
    ents = {resolve_entity(e) for e in x.get("entities") or []}
    ents.discard(None)
    if not ents:   # a known name the model did not return
        e = resolve_entity(text)
        if e:
            ents.add(e)
    o["entities"] = sorted(ents)
    o["count"] = x.get("count")
    set_place(o, resolve_place(x.get("place")) or find_place_in(text))
    o["llm_raw"] = x


KEYWORDS = [("drone_sighting", ["drone", "uas", "quadcopter"]), ("gps_interference", ["gps", "gnss", "navigation interference"]),
            ("power_outage", ["power", "generator"]), ("port_activity", ["container", "lifts", "cargo", "loads"]),
            ("convoy_move", ["convoy", "column", "vehicles"]), ("vessel_loiter", ["loiter", "holding position", "circling"]),
            ("vessel_transit", ["transit", "passed through", "making way"]), ("air_activity", ["low pass", "orbiting", "operating over"]),
            ("vessel_incident", ["aground", "collision", "leaking", "on fire"])]


def rules_extract(text):
    """A keyword baseline, to measure what the language model adds."""
    t = text.lower()
    et = next((k for k, words in KEYWORDS if any(w in t for w in words)), "none")
    m = re.search(r"\b(\d{1,2})\s+(vehicles|containers|lifts|loads)", t)
    return {"event_type": et, "entities": [], "place": None, "count": int(m.group(1)) if m else None}


def normalize(rec, use_llm=True):
    src, p = rec["source"], rec["payload"]
    o = base(rec)
    received = parse_ts(rec["received_at"])
    if src == "sensor":
        set_time(o, parse_ts(p["time"]), 1, "exact")
        o["lat"], o["lon"], o["radius_km"] = p["lat"], p["lon"], 1.0
        o["place"] = nearest_place(p["lat"], p["lon"])
        if p["class"] == "surface":
            o["entities"] = [AIS[p["ais"]]] if p.get("ais") in AIS else ["V-UNKNOWN"]
            o["event_type"] = "vessel_loiter" if p["speed_kt"] < 4 else "vessel_transit"
        else:
            o["entities"] = [CALLSIGNS[p["callsign"].upper()]] if p.get("callsign", "").upper() in CALLSIGNS else []
            o["event_type"] = "air_activity"
        o["text"] = f"Track {p['track_id']}: {p['class']} contact at {p['speed_kt']} kt"
        o["method"] = "structured"
    elif src == "imagery":
        o["text"] = p["note"]
        extract(o, p["note"], use_llm)
        set_time(o, parse_ts(p["capture_time"]), 1, "exact")
        o["lat"], o["lon"], o["radius_km"] = p["lat"], p["lon"], 0.5
        o["place"] = nearest_place(p["lat"], p["lon"]) or o["place"]
        o["count"] = p.get("count") if p.get("count") is not None else o["count"]
        o["method"] = f"structured+{o['method']}"
    elif src == "comms":
        ts, callsign, msg = p["line"].split(",", 2)
        o["text"] = msg
        extract(o, msg, use_llm)
        set_time(o, parse_ts(ts), 2, "logged")      # this clock's offset is estimated later
    elif src == "patrol":
        text = p["text"]
        o["text"] = text
        m = re.search(r"CORRECTION to ([A-Z]{3}-\d{5})", text)
        if m:
            o["correction_of"] = f"patrol:{m.group(1)}"
        extract(o, re.sub(r"^.*?:\s*", "", text, count=1) if m else text, use_llm)
        t, zone = patrol_time(text, received)
        if t:
            set_time(o, t, 15, "local" if zone == "local" else "labelled_z")
    elif src == "osint":
        text = p["text"]
        o["text"] = text
        extract(o, text, use_llm)
        w = osint_window(text, received)
        if w:
            lo, hi = w
            mid = lo + (hi - lo) / 2
            o["time"], o["time_lo"], o["time_hi"], o["time_basis"] = iso(mid), iso(lo), iso(hi), "vague"
        else:
            set_time(o, received - timedelta(hours=3), 180, "received_minus")
        o["radius_km"] = 8.0 if o["place"] else None
    return o
