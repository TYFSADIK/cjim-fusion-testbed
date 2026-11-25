"""The scenario: a fictional 72-hour exercise, NORTHERN LANTERN, observed by five independent sources.

Ground-truth events happen at real coordinates in the Gulf of St. Lawrence under invented place
names; no real people, units or vessels appear. Each source observes some of them, in its own
imperfect way, and the generator records exactly which record belongs to which event, so every
step of the fusion can be scored.

  sensor   structured tracks from a coastal radar and an AIS-like feed: precise, frequent, own IDs
  patrol   free-text patrol reports: local time (UTC-4), sometimes mislabelled "Z", place names
  comms    a communications log as CSV: UTC, but the logging clock runs 7 minutes fast
  imagery  imagery analysis notes: accurate capture time, reported hours later, counts
  osint    open-source snippets: vague times, approximate places, and one fabricated rumour

Re-sent duplicates, corrections that change a time or count, and unrelated chatter are mixed in.

    python -m scenario.generate [--seed 7] [--out data/scenario]
"""
import argparse
import json
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

T0 = datetime(2026, 11, 16, 0, 0, tzinfo=timezone.utc)
HOURS = 72
LOCAL = timezone(timedelta(hours=-4), "AST")

# invented names at real coordinates (Gulf of St. Lawrence, Cape Breton, Magdalen Islands)
PLACES = {
    "Port Avery":       {"lat": 46.137, "lon": -60.195, "aliases": ["Avery harbour", "PT AVERY"]},
    "Kestrel Point":    {"lat": 46.248, "lon": -60.380, "aliases": ["Kestrel Pt", "KESTREL"]},
    "Gannet Bank":      {"lat": 47.050, "lon": -60.700, "aliases": ["the Gannet shoal", "GANNET BK"]},
    "Brannock Narrows": {"lat": 45.980, "lon": -60.900, "aliases": ["the Narrows", "BRANNOCK"]},
    "Fairhaven Depot":  {"lat": 46.030, "lon": -60.520, "aliases": ["Fairhaven", "FH DEPOT"]},
    "Ridley Airfield":  {"lat": 46.165, "lon": -60.050, "aliases": ["Ridley strip", "RIDLEY AFD"]},
    "Highcliff Radar":  {"lat": 46.820, "lon": -60.400, "aliases": ["Highcliff site", "HC RADAR"]},
    "Merrow Island":    {"lat": 47.380, "lon": -61.850, "aliases": ["Merrow", "MERROW IS"]},
    "Tolland Ferry":    {"lat": 45.640, "lon": -61.410, "aliases": ["the Tolland crossing", "TOLLAND"]},
    "Cobb Reach":       {"lat": 46.600, "lon": -61.100, "aliases": ["Cobb waters", "COBB RCH"]},
    "Ashby Substation": {"lat": 46.090, "lon": -60.300, "aliases": ["Ashby sub", "ASHBY SS"]},
    "North Spit":       {"lat": 46.900, "lon": -60.250, "aliases": ["the Spit", "N SPIT"]},
}

# fictional entities, each with the names different sources use for it
ENTITIES = {
    "V-ORCA":    {"kind": "vessel", "names": ["MV Orca Dawn", "ORCA DAWN", "fishing vessel Orca"], "ais": "316000101"},
    "V-TERN":    {"kind": "vessel", "names": ["MV Grey Tern", "GREY TERN", "the Tern"], "ais": "316000102"},
    "V-HALCYON": {"kind": "vessel", "names": ["research vessel Halcyon", "HALCYON", "RV Halcyon"], "ais": "316000103"},
    "V-UNKNOWN": {"kind": "vessel", "names": ["unidentified vessel", "dark vessel", "UNK VESSEL"], "ais": None},
    "A-LANTERN21": {"kind": "aircraft", "names": ["LANTERN 21", "LNTRN21", "patrol aircraft Lantern 21"]},
    "A-DRONE":   {"kind": "aircraft", "names": ["small drone", "UAS", "quadcopter"]},
    "U-BLUE3":   {"kind": "unit", "names": ["Blue 3", "BLUE-3", "3rd Blue troop"]},
    "U-RED1":    {"kind": "unit", "names": ["Red 1", "RED-1", "opposing force element Red One"]},
}

EVENT_TYPES = {
    "vessel_loiter":   {"entities": ["V-ORCA", "V-TERN", "V-UNKNOWN"], "places": ["Gannet Bank", "Cobb Reach", "North Spit", "Merrow Island"], "dur": (2, 6)},
    "vessel_transit":  {"entities": ["V-ORCA", "V-TERN", "V-HALCYON"], "places": ["Brannock Narrows", "Tolland Ferry", "Cobb Reach"], "dur": (1, 3)},
    "convoy_move":     {"entities": ["U-BLUE3", "U-RED1"], "places": ["Fairhaven Depot", "Port Avery", "Kestrel Point"], "dur": (1, 2)},
    "air_activity":    {"entities": ["A-LANTERN21"], "places": ["Ridley Airfield", "Highcliff Radar", "Gannet Bank"], "dur": (1, 3)},
    "drone_sighting":  {"entities": ["A-DRONE"], "places": ["Ashby Substation", "Ridley Airfield", "Port Avery"], "dur": (0.25, 0.75)},
    "gps_interference": {"entities": [], "places": ["Highcliff Radar", "Kestrel Point", "North Spit"], "dur": (0.5, 2)},
    "power_outage":    {"entities": [], "places": ["Ashby Substation", "Fairhaven Depot"], "dur": (1, 4)},
    "port_activity":   {"entities": ["V-HALCYON", "V-ORCA"], "places": ["Port Avery", "Kestrel Point"], "dur": (2, 5)},
}

# NATO-style source grading: reliability A (completely reliable) to F, credibility 1 (confirmed) to 6
SOURCES = {
    "sensor":  {"reliability": "A", "credibility": 2, "name": "Coastal radar and AIS feed"},
    "patrol":  {"reliability": "B", "credibility": 2, "name": "Ground patrol reports"},
    "comms":   {"reliability": "B", "credibility": 3, "name": "Communications log"},
    "imagery": {"reliability": "A", "credibility": 2, "name": "Imagery analysis notes"},
    "osint":   {"reliability": "D", "credibility": 4, "name": "Open-source monitoring"},
}
COMMS_CLOCK_FAST = timedelta(minutes=7)


def iso(t):
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def jitter(rng, lat, lon, metres):
    d = rng.gauss(0, metres) / 111_320
    a = rng.uniform(0, 2 * math.pi)
    return round(lat + d * math.cos(a), 5), round(lon + d * math.sin(a) / math.cos(math.radians(lat)), 5)


def clashes(events, et, place, ents, start, end):
    """A physically impossible or indistinguishable pair: the same vessel, aircraft or unit in two
    events at once, or two events of one type at one place within two hours of each other."""
    for e in events:
        s0 = datetime.fromisoformat(e["start"].replace("Z", "+00:00"))
        e0 = datetime.fromisoformat(e["end"].replace("Z", "+00:00"))
        if ents and set(ents) & set(e["entities"]) and start < e0 + timedelta(hours=1) and s0 < end + timedelta(hours=1):
            return True
        if e["type"] == et and e["place"] == place and start < e0 + timedelta(hours=2) and s0 < end + timedelta(hours=2):
            return True
    return False


def make_events(rng, n):
    events = []
    while len(events) < n:
        i = len(events)
        et = rng.choice(list(EVENT_TYPES))
        spec = EVENT_TYPES[et]
        start = T0 + timedelta(minutes=rng.randint(60, HOURS * 60 - 360))
        dur = timedelta(hours=rng.uniform(*spec["dur"]))
        place = rng.choice(spec["places"])
        ents = [rng.choice(spec["entities"])] if spec["entities"] else []
        if clashes(events, et, place, ents, start, start + dur):
            continue
        events.append({"id": f"E{i + 1:03d}", "type": et, "start": iso(start), "end": iso(start + dur),
                       "place": place, "lat": PLACES[place]["lat"], "lon": PLACES[place]["lon"],
                       "entities": ents, "count": rng.randint(2, 9) if et in ("convoy_move", "port_activity") else None})
    return events


# Plain wording and indirect wording, as people really write. Indirect phrases avoid the obvious
# keywords, so a keyword matcher misses them and an extractor has to understand the sentence.
PHRASES = {
    "vessel_loiter": ["{ent} loitering near {place}", "{ent} holding position off {place}", "{ent} circling slowly at {place}",
                      "{ent} has been sitting off {place} for hours, not going anywhere", "{ent} drifting back and forth near {place}, engines idling"],
    "vessel_transit": ["{ent} transiting {place}", "{ent} passed through {place} heading north", "{ent} making way through {place}",
                       "{ent} came through {place} at speed and kept going", "{ent} went by {place} northbound without stopping"],
    "convoy_move": ["convoy of {count} vehicles ({ent}) moving out of {place}", "{ent} column, {count} vehicles, departing {place}",
                    "{count} vehicles with {ent} on the road from {place}", "{ent} rolled out of {place} in {count} trucks",
                    "a line of {count} trucks belonging to {ent} left {place}"],
    "air_activity": ["{ent} operating over {place}", "{ent} low pass over {place}", "{ent} orbiting {place}",
                     "{ent} heard overhead at {place} for a long time", "{ent} kept coming back over {place}"],
    "drone_sighting": ["{ent} seen hovering over {place}", "{ent} spotted near {place}, operator not seen", "{ent} overflying {place}",
                       "something small with blinking lights buzzing over {place}", "a buzzing object with propellers above {place}"],
    "gps_interference": ["GPS signal lost near {place}", "navigation interference reported at {place}", "GNSS jamming suspected around {place}",
                         "nav displays going haywire around {place}", "position fix wandering kilometres off near {place}"],
    "power_outage": ["power out at {place}", "loss of mains power at {place}", "{place} on backup generators after outage",
                     "lights went dark across {place}", "{place} running on its own diesel, grid supply gone"],
    "port_activity": ["{count} containers being loaded at {place} onto {ent}", "{ent} alongside at {place}, {count} lifts observed",
                      "cargo operations at {place} with {ent}, about {count} loads", "cranes busy at {place} putting {count} boxes aboard {ent}",
                      "{ent} tied up at {place} taking on {count} crates"],
}


def phrase(rng, ev, name_style):
    ent = ""
    if ev["entities"]:
        names = ENTITIES[ev["entities"][0]]["names"]
        ent = names[name_style % len(names)]
    place = ev["place"] if rng.random() < 0.6 else rng.choice(PLACES[ev["place"]]["aliases"])
    return rng.choice(PHRASES[ev["type"]]).format(ent=ent, place=place, count=ev["count"] or "several")


def generate(seed=7, n_events=60):
    rng = random.Random(seed)
    events = make_events(rng, n_events)
    records, truth = {s: [] for s in SOURCES}, []

    def add(src, rec, event_id, true_time, kind, duplicate_of=None):
        records[src].append(rec)
        truth.append({"source": src, "record_id": rec["record_id"], "event_id": event_id, "true_time": iso(true_time) if true_time else None,
                      "kind": kind, "duplicate_of": duplicate_of})

    counters = {s: 0 for s in SOURCES}

    def rid(src):
        counters[src] += 1
        return f"{src.upper()[:3]}-{counters[src]:05d}"

    for ev in events:
        start = datetime.fromisoformat(ev["start"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(ev["end"].replace("Z", "+00:00"))
        mid = start + (end - start) / 2
        ent = ev["entities"][0] if ev["entities"] else None
        kind = ENTITIES[ent]["kind"] if ent else None

        # sensor: tracks for vessels and aircraft, 1 to 3 reports over the event
        if kind in ("vessel", "aircraft") and ev["type"] != "drone_sighting" and rng.random() < 0.85:
            track = f"TRK-{rng.randint(1000, 9999)}"
            for k in range(rng.randint(1, 3)):
                t = start + (end - start) * rng.uniform(0, 1)
                lat, lon = jitter(rng, ev["lat"], ev["lon"], 600)
                rec = {"record_id": rid("sensor"), "received_at": iso(t + timedelta(seconds=rng.randint(5, 60))),
                       "payload": {"track_id": track, "time": iso(t), "lat": lat, "lon": lon,
                                   "class": "surface" if kind == "vessel" else "air",
                                   "ais": ENTITIES[ent].get("ais") if kind == "vessel" else None,
                                   "callsign": ENTITIES[ent]["names"][1] if kind == "aircraft" else None,
                                   "speed_kt": round(rng.uniform(0, 4) if ev["type"] == "vessel_loiter" else rng.uniform(6, 18), 1)}}
                add("sensor", rec, ev["id"], t, "genuine")

        # patrol: free text in local time; 10% wrongly labelled Z; 12% re-sent; 8% corrected
        if ev["type"] in ("convoy_move", "drone_sighting", "power_outage", "port_activity", "vessel_loiter", "gps_interference") and rng.random() < 0.7:
            t = start + (end - start) * rng.uniform(0, 0.6)
            approx = t + timedelta(minutes=rng.choice([-10, -5, 0, 0, 5, 10]))
            will_correct = rng.random() < 0.09
            # a report that will be corrected states the wrong time; the correction states the right one
            stated = approx - timedelta(minutes=30) if will_correct else approx
            local = stated.astimezone(LOCAL)
            mislabel = not will_correct and rng.random() < 0.10
            when = f"{local:%H%M}{'Z' if mislabel else ' local'}"
            text = f"PATROL REPORT. At approx {when} on {local:%d %b}, {phrase(rng, ev, 0)}. Observed by foot patrol, no contact."
            rec = {"record_id": rid("patrol"), "received_at": iso(t + timedelta(minutes=rng.randint(15, 90))), "payload": {"text": text}}
            add("patrol", rec, ev["id"], approx, "genuine")
            if will_correct:
                fl = approx.astimezone(LOCAL)
                ctext = f"CORRECTION to {rec['record_id']}: time should read {fl:%H%M} local on {fl:%d %b}. {phrase(rng, ev, 0)}."
                corr = {"record_id": rid("patrol"), "received_at": iso(t + timedelta(minutes=rng.randint(120, 240))), "payload": {"text": ctext}}
                add("patrol", corr, ev["id"], approx, "correction", duplicate_of=rec["record_id"])
            elif rng.random() < 0.12:
                dup = {"record_id": rid("patrol"), "received_at": iso(t + timedelta(minutes=rng.randint(100, 200))), "payload": {"text": text}}
                add("patrol", dup, ev["id"], approx, "duplicate", duplicate_of=rec["record_id"])

        # comms: CSV line, UTC but the logging clock runs 7 minutes fast
        if rng.random() < 0.55:
            t = start + (end - start) * rng.uniform(0, 0.8)
            callsign = rng.choice(["HOTEL 2", "VICTOR 9", "KILO 4"])
            line = f"{iso(t + COMMS_CLOCK_FAST)},{callsign},{phrase(rng, ev, 1).upper()}"
            # the receiving system's clock is right: so the log appears to record messages after they arrived
            rec = {"record_id": rid("comms"), "received_at": iso(t + timedelta(seconds=20)), "payload": {"line": line}}
            add("comms", rec, ev["id"], t, "genuine")

        # imagery: structured note, capture time exact, delivered hours later, with counts
        if ev["type"] in ("vessel_loiter", "port_activity", "convoy_move", "power_outage") and rng.random() < 0.5:
            t = start + (end - start) * rng.uniform(0.1, 0.9)
            lat, lon = jitter(rng, ev["lat"], ev["lon"], 150)
            rec = {"record_id": rid("imagery"), "received_at": iso(t + timedelta(hours=rng.uniform(3, 9))),
                   "payload": {"capture_time": iso(t), "lat": lat, "lon": lon, "count": ev["count"],
                               "note": f"Imagery shows {phrase(rng, ev, 2)}."}}
            add("imagery", rec, ev["id"], t, "genuine")

        # osint: vague time, sometimes wrong count
        if rng.random() < 0.35:
            t = start + (end - start) * rng.uniform(0, 1)
            local = t.astimezone(LOCAL)
            part = "early morning" if local.hour < 9 else "afternoon" if local.hour < 17 else "evening"
            text = f"Local social media post ({local:%A} {part}): residents say {phrase(rng, ev, 2)}."
            rec = {"record_id": rid("osint"), "received_at": iso(t + timedelta(hours=rng.uniform(1, 6))), "payload": {"text": text}}
            add("osint", rec, ev["id"], t, "genuine")

    # a fabricated rumour, from open source only: an event that never happened
    t = T0 + timedelta(hours=40)
    local = t.astimezone(LOCAL)
    for k in range(2):
        rec = {"record_id": rid("osint"), "received_at": iso(t + timedelta(hours=1 + k)),
               "payload": {"text": f"Local social media post ({local:%A} afternoon): residents say a navy ship ran aground at North Spit and is leaking fuel{'.' if k == 0 else ', photos to follow.'}"}}
        add("osint", rec, None, t, "false")

    # unrelated chatter
    for _ in range(25):
        t = T0 + timedelta(minutes=rng.randint(0, HOURS * 60))
        src = rng.choice(["patrol", "comms", "osint"])
        msg = rng.choice(["radio check, all stations", "nothing significant to report", "weather deteriorating, winds 30 knots",
                          "patrol returned to base", "routine fuel delivery completed"])
        if src == "comms":
            rec = {"record_id": rid("comms"), "received_at": iso(t + timedelta(seconds=20)), "payload": {"line": f"{iso(t + COMMS_CLOCK_FAST)},ZERO,{msg.upper()}"}}
        elif src == "patrol":
            lt = t.astimezone(LOCAL)
            rec = {"record_id": rid("patrol"), "received_at": iso(t + timedelta(minutes=20)), "payload": {"text": f"PATROL REPORT. At {lt:%H%M} local on {lt:%d %b}, {msg}."}}
        else:
            rec = {"record_id": rid("osint"), "received_at": iso(t + timedelta(hours=1)), "payload": {"text": f"Local social media post: {msg}."}}
        add(src, rec, None, t, "noise")

    for src in records:
        records[src].sort(key=lambda r: r["received_at"])
    return events, records, truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--events", type=int, default=60)
    ap.add_argument("--out", default="data/scenario")
    a = ap.parse_args()
    events, records, truth = generate(a.seed, a.events)
    out = Path(a.out)
    (out / "sources").mkdir(parents=True, exist_ok=True)
    for src, recs in records.items():
        with open(out / "sources" / f"{src}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps({"source": src, **r}) + "\n")
    json.dump({"seed": a.seed, "start": iso(T0), "hours": HOURS, "sources": SOURCES, "places": PLACES, "entities": ENTITIES,
               "events": events, "records": truth}, open(out / "truth.json", "w"), indent=1)
    n = {s: len(r) for s, r in records.items()}
    kinds = {}
    for t in truth:
        kinds[t["kind"]] = kinds.get(t["kind"], 0) + 1
    print(f"{len(events)} events; records {n}, total {sum(n.values())}; {kinds}")


if __name__ == "__main__":
    main()
