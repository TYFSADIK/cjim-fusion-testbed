"""Where everything is kept, and how it is searched.

  PostGIS     observations, fused events, reports and analyst reviews; the system of record,
              with geometry for map and distance queries
  OpenSearch  full-text search over observations and approved report text
  Qdrant      meaning-based search over observations (vLLM embeddings)
"""
import json
import os
import uuid

import httpx
import psycopg

PG = os.environ.get("PG", "host=127.0.0.1 port=5433 dbname=cjim user=cjim password=cjim")
OS = os.environ.get("OPENSEARCH", "http://127.0.0.1:9200")
QD = os.environ.get("QDRANT", "http://127.0.0.1:6333")
http = httpx.Client(timeout=30)

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE TABLE IF NOT EXISTS observations (
  obs_id text PRIMARY KEY, source text, received_at timestamptz, time timestamptz, event_type text,
  place text, geom geometry(Point, 4326), doc jsonb, normalised_at timestamptz DEFAULT now());
CREATE TABLE IF NOT EXISTS events (
  event_id text PRIMARY KEY, type text, place text, start_t timestamptz, end_t timestamptz, confidence real, band text,
  geom geometry(Point, 4326), members text[], doc jsonb, updated_at timestamptz DEFAULT now());
CREATE TABLE IF NOT EXISTS reports (
  event_id text PRIMARY KEY, members_hash text, report jsonb, status text, reviewer text, review_note text,
  reviewed_at timestamptz, written_at timestamptz DEFAULT now());
CREATE TABLE IF NOT EXISTS pipeline_log (t timestamptz DEFAULT now(), stage text, detail jsonb);
CREATE INDEX IF NOT EXISTS events_geom ON events USING gist (geom);
CREATE INDEX IF NOT EXISTS obs_geom ON observations USING gist (geom);
"""


def conn():
    return psycopg.connect(PG, autocommit=True)


def init():
    with conn() as c:
        c.execute("SELECT pg_advisory_lock(42)")    # several workers start at once
        c.execute(SCHEMA)
        c.execute("SELECT pg_advisory_unlock(42)")
    http.put(f"{OS}/observations", json={"mappings": {"properties": {
        "text": {"type": "text"}, "source": {"type": "keyword"}, "event_type": {"type": "keyword"},
        "place": {"type": "keyword"}, "time": {"type": "date"}, "event_id": {"type": "keyword"}}}})
    http.put(f"{OS}/reports", json={"mappings": {"properties": {"text": {"type": "text"}, "event_id": {"type": "keyword"},
                                                                "status": {"type": "keyword"}}}})


def ensure_collection(dim):
    r = http.get(f"{QD}/collections/observations")
    if r.status_code == 404:
        http.put(f"{QD}/collections/observations", json={"vectors": {"size": dim, "distance": "Cosine"}}).raise_for_status()


def save_observation(o, vector=None):
    with conn() as c:
        c.execute("""INSERT INTO observations (obs_id, source, received_at, time, event_type, place, geom, doc)
                     VALUES (%s, %s, %s, %s, %s, %s, CASE WHEN %s::float IS NULL THEN NULL ELSE ST_SetSRID(ST_MakePoint(%s, %s), 4326) END, %s)
                     ON CONFLICT (obs_id) DO UPDATE SET doc = excluded.doc, time = excluded.time, event_type = excluded.event_type,
                       place = excluded.place, geom = excluded.geom""",
                  (o["obs_id"], o["source"], o["received_at"], o.get("time"), o.get("event_type"), o.get("place"),
                   o.get("lon"), o.get("lon"), o.get("lat"), json.dumps(o)))
    http.put(f"{OS}/observations/_doc/{o['obs_id'].replace(':', '_')}",
             json={k: o.get(k) for k in ("text", "source", "event_type", "place", "time")} | {"obs_id": o["obs_id"]})
    if vector:
        pid = str(uuid.uuid5(uuid.NAMESPACE_URL, o["obs_id"]))   # the same ID in every process
        http.put(f"{QD}/collections/observations/points", json={"points": [
            {"id": pid, "vector": vector, "payload": {"obs_id": o["obs_id"], "source": o["source"]}}]}).raise_for_status()


def load_observations():
    with conn() as c:
        return [r[0] for r in c.execute("SELECT doc FROM observations ORDER BY received_at").fetchall()]


def load_vectors(obs_ids):
    out = {}
    if not obs_ids:
        return out
    r = http.post(f"{QD}/collections/observations/points/scroll",
                  json={"limit": 10000, "with_vector": True, "with_payload": True})
    if r.status_code != 200:
        return out
    for p in r.json()["result"]["points"]:
        out[p["payload"]["obs_id"]] = p["vector"]
    return out


def save_events(events, obs):
    by_id = {o["obs_id"]: o for o in obs}
    with conn() as c:
        c.execute("DELETE FROM events")
        for e in events:
            c.execute("""INSERT INTO events (event_id, type, place, start_t, end_t, confidence, band, geom, members, doc)
                         VALUES (%s,%s,%s,%s,%s,%s,%s, CASE WHEN %s::float IS NULL THEN NULL ELSE ST_SetSRID(ST_MakePoint(%s,%s),4326) END, %s, %s)""",
                      (e["event_id"], e["type"], e["place"], e["start"], e["end"], e["confidence"], e["band"],
                       e["lon"], e["lon"], e["lat"], e["members"], json.dumps(e)))
    for e in events:
        for m in e["members"] + e["duplicates"]:
            if m in by_id:
                http.post(f"{OS}/observations/_update/{m.replace(':', '_')}", json={"doc": {"event_id": e["event_id"]}})


def save_report(event_id, members_hash, report):
    with conn() as c:
        c.execute("""INSERT INTO reports (event_id, members_hash, report, status) VALUES (%s, %s, %s, %s)
                     ON CONFLICT (event_id) DO UPDATE SET members_hash = excluded.members_hash, report = excluded.report,
                       status = excluded.status, reviewer = NULL, review_note = NULL, reviewed_at = NULL, written_at = now()""",
                  (event_id, members_hash, json.dumps(report), report["status"]))


def supersede_reports(current_ids):
    """Reports of events that no longer exist after re-fusion (their evidence moved to other events)
    leave the review queue; they are kept, with any review, for the record."""
    with conn() as c:
        return c.execute("""UPDATE reports SET status = 'superseded' WHERE NOT (event_id = ANY(%s)) AND status <> 'superseded'""",
                         (list(current_ids),)).rowcount


def report_hashes():
    with conn() as c:
        return dict(c.execute("SELECT event_id, members_hash FROM reports").fetchall())


def log(stage, detail):
    with conn() as c:
        c.execute("INSERT INTO pipeline_log (stage, detail) VALUES (%s, %s)", (stage, json.dumps(detail)))
