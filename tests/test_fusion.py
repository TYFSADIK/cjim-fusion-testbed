"""Unit tests: time parsing, time reconciliation, deduplication, association gates, confidence, and the
report checker. No vLLM, no services: normalisation runs with its keyword baseline here."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.fuse import align_times, associate, confidence, dedup, fuse  # noqa: E402
from fusion.normalize import normalize, osint_window, patrol_time, parse_ts  # noqa: E402
from fusion.report import check_sentence  # noqa: E402
from scenario.generate import generate  # noqa: E402


def recs(text, src="patrol", rid="PAT-00001", received="2026-11-16T05:00:00Z"):
    return {"source": src, "record_id": rid, "received_at": received, "payload": {"text": text}}


def test_patrol_local_time_to_utc():
    t, zone = patrol_time("At approx 2309 local on 15 Nov, x", parse_ts("2026-11-16T05:00:00Z"))
    assert zone == "local" and t.isoformat().startswith("2026-11-15T23:09:00-04:00")


def test_osint_vague_window():
    lo, hi = osint_window("post (Monday early morning): x", parse_ts("2026-11-16T13:00:00Z"))
    assert lo.hour == 0 and hi.hour == 9 and lo.day == 16


def test_scenario_is_deterministic_and_scored_records_exist():
    a, b = generate(7), generate(7)
    assert a[0] == b[0] and len(a[2]) > 150


def test_comms_clock_and_labelled_z_are_found():
    events, records, truth = generate(7)
    obs = [normalize({"source": s, **r}, use_llm=False) for s in records for r in records[s]]
    dedup(obs)
    found = align_times(obs)
    assert 380 < found["comms"]["clock_offset_s"] < 430          # 7 min fast, less 20 s of delivery
    assert found["patrol"]["local_time_labelled_z"] >= 1


def test_copies_merge_and_a_correction_supersedes_the_time():
    o1 = normalize(recs("PATROL REPORT. At approx 0100 local on 16 Nov, power out at Ashby sub."), use_llm=False)
    o2 = normalize(recs("PATROL REPORT. At approx 0100 local on 16 Nov, power out at Ashby sub.", rid="PAT-00002",
                        received="2026-11-16T06:00:00Z"), use_llm=False)
    o3 = normalize(recs("CORRECTION to PAT-00001: time should read 0130 local on 16 Nov. power out at Ashby sub.",
                        rid="PAT-00003", received="2026-11-16T07:00:00Z"), use_llm=False)
    dedup([o1, o2, o3])
    assert o2["duplicate_of"] == o1["obs_id"] and o3["duplicate_of"] == o1["obs_id"]
    assert o1["time"] == "2026-11-16T05:30:00Z"


def test_different_entities_never_associate():
    a = normalize(recs("PATROL REPORT. At approx 0100 local on 16 Nov, MV Orca Dawn loitering near Gannet Bank."), use_llm=False)
    b = normalize(recs("PATROL REPORT. At approx 0110 local on 16 Nov, MV Grey Tern loitering near Gannet Bank.", rid="PAT-00002"),
                  use_llm=False)
    assert len(associate([a, b])) == 2


def test_single_low_grade_source_is_low_confidence():
    o = {"source": "osint", "reliability": "D", "credibility": 4}
    conf, srcs, band = confidence([o, dict(o)])
    assert band == "low" and srcs == ["osint"]
    conf2, _, band2 = confidence([o, {"source": "sensor", "reliability": "A", "credibility": 2}])
    assert conf2 > conf and band2 == "high"


def test_report_checker_flags_unsupported_facts():
    obs = {"patrol:PAT-1": {"obs_id": "patrol:PAT-1", "time": "2026-11-16T05:09:00Z", "text": "9 vehicles departing Port Avery",
                            "place": "Port Avery", "entities": ["U-BLUE3"], "count": 9}}
    ev = {"members": ["patrol:PAT-1"]}
    ok = {"text": "Blue 3 left Port Avery at 0509Z with 9 vehicles.", "cites": ["patrol:PAT-1"]}
    assert check_sentence(ok, ev, obs) == []
    # a time quoted as the source wrote it ("0105Z" in the text) is supported too
    obs["patrol:PAT-1"]["text"] = "At approx 0105Z, 9 vehicles departing Port Avery"
    assert check_sentence({"text": "Reported at 0105Z.", "cites": ["patrol:PAT-1"]}, ev, obs) == []
    bad = {"text": "Red 1 left Kestrel Point at 0700Z with 12 vehicles.", "cites": ["patrol:PAT-1"]}
    problems = check_sentence(bad, ev, obs)
    assert len(problems) == 4, problems
    assert check_sentence({"text": "x", "cites": ["osint:OSI-9"]}, ev, obs)


def test_baseline_end_to_end_scores():
    events, records, truth_ = generate(7)
    obs = [normalize({"source": s, **r}, use_llm=False) for s in records for r in records[s]]
    fused, _ = fuse(obs)
    from eval.score import score
    truth = {"events": events, "records": truth_}
    s = score(obs, fused, truth)
    assert s["duplicates"]["precision"] == 1.0 and s["duplicates"]["recall"] == 1.0
    assert s["association"]["precision"] > 0.9
    assert s["confidence"]["rumour"] and all(r["band"] == "low" for r in s["confidence"]["rumour"])


def test_untyped_report_does_not_corroborate_on_place_alone():
    a = normalize(recs("PATROL REPORT. At approx 0100 local on 16 Nov, power out at Ashby sub."), use_llm=False)
    b = normalize(recs("PATROL REPORT. At approx 0110 local on 16 Nov, something odd near Ashby sub.", rid="PAT-00002"),
                  use_llm=False)
    assert a["event_type"] == "power_outage" and b["event_type"] is None
    assert len(associate([a, b])) == 2
    # ...unless it says nearly the same thing
    assert len(associate([a, b], emb={a["obs_id"]: [1.0, 0.0], b["obs_id"]: [1.0, 0.0]})) == 1


def test_report_schema_limits_citations_to_the_event():
    from fusion.report import schema_for
    s = schema_for({"members": ["patrol:PAT-1", "comms:COM-2"]})
    assert s["properties"]["sentences"]["items"]["properties"]["cites"]["items"]["enum"] == ["patrol:PAT-1", "comms:COM-2"]


def test_names_match_whole_words_only():
    from fusion.reference import find_place_in, resolve_entity
    assert resolve_entity("post (Tuesday afternoon): a navy ship ran aground") is None     # "afTERNoon"
    assert resolve_entity("the Tern turned north") == "V-TERN"
    assert find_place_in("convoy left Port Avery at dawn") == "Port Avery"


def test_checker_flags_overstated_corroboration():
    obs = {"osint:O-1": {"obs_id": "osint:O-1", "source": "osint", "time": "2026-11-17T17:00:00Z", "text": "a ship ran aground",
                         "place": None, "entities": [], "count": None},
           "osint:O-2": {"obs_id": "osint:O-2", "source": "osint", "time": "2026-11-17T17:00:00Z", "text": "a ship ran aground",
                         "place": None, "entities": [], "count": None}}
    ev = {"members": ["osint:O-1", "osint:O-2"], "sources": ["osint"]}
    bad = {"text": "Two independent posts say a ship ran aground.", "cites": ["osint:O-1", "osint:O-2"]}
    assert any("independent" in p for p in check_sentence(bad, ev, obs))
    ok = {"text": "Two posts from one source say a ship ran aground.", "cites": ["osint:O-1", "osint:O-2"]}
    assert check_sentence(ok, ev, obs) == []
