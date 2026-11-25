"""Reference data an analyst cell would hold: the gazetteer, the names known for vessels, aircraft
and units, and the grading of each source. The fusion pipeline uses only this, never the
scenario's ground truth."""
import re
import difflib
import math

GAZETTEER = {
    "Port Avery": (46.137, -60.195, ["Avery harbour", "PT AVERY"]),
    "Kestrel Point": (46.248, -60.380, ["Kestrel Pt", "KESTREL"]),
    "Gannet Bank": (47.050, -60.700, ["the Gannet shoal", "Gannet shoal", "GANNET BK"]),
    "Brannock Narrows": (45.980, -60.900, ["the Narrows", "BRANNOCK"]),
    "Fairhaven Depot": (46.030, -60.520, ["Fairhaven", "FH DEPOT"]),
    "Ridley Airfield": (46.165, -60.050, ["Ridley strip", "RIDLEY AFD"]),
    "Highcliff Radar": (46.820, -60.400, ["Highcliff site", "HC RADAR"]),
    "Merrow Island": (47.380, -61.850, ["Merrow", "MERROW IS"]),
    "Tolland Ferry": (45.640, -61.410, ["the Tolland crossing", "Tolland crossing", "TOLLAND"]),
    "Cobb Reach": (46.600, -61.100, ["Cobb waters", "COBB RCH"]),
    "Ashby Substation": (46.090, -60.300, ["Ashby sub", "ASHBY SS"]),
    "North Spit": (46.900, -60.250, ["the Spit", "N SPIT"]),
}

KNOWN = {
    "V-ORCA": ("vessel", ["MV Orca Dawn", "Orca Dawn", "fishing vessel Orca", "Orca"], "316000101"),
    "V-TERN": ("vessel", ["MV Grey Tern", "Grey Tern", "the Tern", "Tern"], "316000102"),
    "V-HALCYON": ("vessel", ["research vessel Halcyon", "RV Halcyon", "Halcyon"], "316000103"),
    "V-UNKNOWN": ("vessel", ["unidentified vessel", "dark vessel", "UNK VESSEL", "unknown vessel"], None),
    "A-LANTERN21": ("aircraft", ["LANTERN 21", "LNTRN21", "patrol aircraft Lantern 21", "Lantern 21"], None),
    "A-DRONE": ("aircraft", ["small drone", "UAS", "quadcopter", "drone"], None),
    "U-BLUE3": ("unit", ["Blue 3", "BLUE-3", "3rd Blue troop"], None),
    "U-RED1": ("unit", ["Red 1", "RED-1", "opposing force element Red One", "Red One"], None),
}
AIS = {v[2]: k for k, v in KNOWN.items() if v[2]}
CALLSIGNS = {n.upper(): k for k, v in KNOWN.items() if v[0] == "aircraft" for n in v[1]}

# NATO-style grading: reliability A (completely reliable) to F; credibility 1 (confirmed) to 6
GRADING = {"sensor": ("A", 2), "patrol": ("B", 2), "comms": ("B", 3), "imagery": ("A", 2), "osint": ("D", 4)}
# how much one report from a source of that grading supports an event being real
RELIABILITY_WEIGHT = {"A": 0.9, "B": 0.8, "C": 0.65, "D": 0.45, "E": 0.25, "F": 0.35}
CREDIBILITY_WEIGHT = {1: 1.0, 2: 0.95, 3: 0.85, 4: 0.7, 5: 0.4, 6: 0.6}

EVENT_TYPES = ["vessel_loiter", "vessel_transit", "convoy_move", "air_activity", "drone_sighting",
               "gps_interference", "power_outage", "port_activity", "vessel_incident"]


def _norm(s):
    return " ".join(s.lower().replace("-", " ").split())


def _contains(text, name):
    """A name inside a text as whole words: "Tern" is in "the Tern turned", not in "afternoon"."""
    return re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text) is not None


def resolve_place(text):
    """A place mention to its gazetteer name, by exact name, alias, or close spelling."""
    if not text:
        return None
    t = _norm(text)
    for name, (_, _, aliases) in GAZETTEER.items():
        if t == _norm(name) or any(t == _norm(a) for a in aliases):
            return name
    for name, (_, _, aliases) in GAZETTEER.items():   # mention inside a longer phrase
        for n in [name] + aliases:
            if len(n) > 4 and _contains(t, _norm(n)):
                return name
    best, score = None, 0.0
    for name, (_, _, aliases) in GAZETTEER.items():
        for n in [name] + aliases:
            r = difflib.SequenceMatcher(None, t, _norm(n)).ratio()
            if r > score:
                best, score = name, r
    return best if score >= 0.8 else None


def find_place_in(text):
    """Any gazetteer place named anywhere in a text (longest match first)."""
    t = _norm(text)
    hits = []
    for name, (_, _, aliases) in GAZETTEER.items():
        for n in [name] + aliases:
            if len(n) > 4 and _contains(t, _norm(n)):
                hits.append((len(n), name))
    return max(hits)[1] if hits else None


def resolve_entity(text):
    if not text:
        return None
    t = _norm(text)
    for eid, (_, names, _) in KNOWN.items():
        if any(t == _norm(n) for n in names):
            return eid
    for eid, (_, names, _) in sorted(KNOWN.items(), key=lambda kv: -max(len(n) for n in kv[1][1])):
        if any(len(n) > 3 and _contains(t, _norm(n)) for n in names):
            return eid
    return None


def nearest_place(lat, lon, max_km=20):
    best, dist = None, 1e9
    for name, (plat, plon, _) in GAZETTEER.items():
        d = km(lat, lon, plat, plon)
        if d < dist:
            best, dist = name, d
    return best if dist <= max_km else None


def km(lat1, lon1, lat2, lon2):
    p = math.pi / 180
    a = 0.5 - math.cos((lat2 - lat1) * p) / 2 + math.cos(lat1 * p) * math.cos(lat2 * p) * (1 - math.cos((lon2 - lon1) * p)) / 2
    return 12742 * math.asin(math.sqrt(a))
