"""The analyst's dashboard: events with confidence and provenance, evidence-linked reports to review,
and search by words (OpenSearch) or by meaning (Qdrant, with vLLM embeddings).

GET  /api/events                 fused events (filter: type, band)
GET  /api/events/{id}            an event, its report, every observation behind it and the raw record
POST /api/events/{id}/review     {"decision": "approve"|"reject", "note": "...", "reviewer": "..."}
GET  /api/search?q=&mode=        mode "text" (OpenSearch) or "meaning" (Qdrant)
GET  /api/stats                  pipeline counts, time findings, review status
"""
import json
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse

from fusion import llm, store

app = FastAPI(title="CJIM fusion")
HERE = Path(__file__).resolve().parent


def rows(sql, *args):
    with store.conn() as c:
        return c.execute(sql, args).fetchall()


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/api/events")
def events(type: str | None = None, band: str | None = None):
    q = """SELECT e.doc, r.status, r.report->>'supported', r.report->>'total' FROM events e
           LEFT JOIN reports r USING (event_id) WHERE (%s::text IS NULL OR e.type = %s) AND (%s::text IS NULL OR e.band = %s)
           ORDER BY e.start_t"""
    out = []
    for doc, status, sup, tot in rows(q, type, type, band, band):
        out.append({**{k: doc[k] for k in ("event_id", "type", "place", "lat", "lon", "start", "end", "entities", "count",
                                           "confidence", "band", "sources")},
                    "observations": len(doc["members"]), "report_status": status,
                    "supported": int(sup) if sup else None, "sentences": int(tot) if tot else None})
    return out


@app.get("/api/events/{event_id}")
def event(event_id: str):
    r = rows("SELECT doc FROM events WHERE event_id = %s", event_id)
    if not r:
        raise HTTPException(404, "no such event")
    doc = r[0][0]
    obs = {o[0]["obs_id"]: o[0] for o in rows("SELECT doc FROM observations WHERE obs_id = ANY(%s)", doc["members"] + doc["duplicates"])}
    rep = rows("SELECT report, status, reviewer, review_note, reviewed_at FROM reports WHERE event_id = %s", event_id)
    return {"event": doc, "observations": [obs[m] for m in doc["members"] + doc["duplicates"] if m in obs],
            "report": rep[0][0] if rep else None, "status": rep[0][1] if rep else None,
            "review": {"reviewer": rep[0][2], "note": rep[0][3], "at": str(rep[0][4]) if rep[0][4] else None} if rep else None}


@app.post("/api/events/{event_id}/review")
def review(event_id: str, body: dict = Body(...)):
    decision = body.get("decision")
    if decision not in ("approve", "reject"):
        raise HTTPException(400, "decision must be approve or reject")
    status = "approved" if decision == "approve" else "rejected"
    with store.conn() as c:
        n = c.execute("""UPDATE reports SET status = %s, reviewer = %s, review_note = %s, reviewed_at = now()
                         WHERE event_id = %s RETURNING report""",
                      (status, body.get("reviewer", "analyst"), body.get("note", ""), event_id)).fetchone()
    if not n:
        raise HTTPException(404, "no report for this event")
    if status == "approved":
        text = " ".join(s["text"] for s in n[0]["sentences"])
        store.http.put(f"{store.OS}/reports/_doc/{event_id}", json={"event_id": event_id, "text": text, "status": status})
    return {"event_id": event_id, "status": status}


@app.get("/api/search")
def search(q: str, mode: str = "text", k: int = 15):
    if mode == "meaning":
        vec = llm.embed([q])[0]
        r = store.http.post(f"{store.QD}/collections/observations/points/search",
                            json={"vector": vec, "limit": k, "with_payload": True}).json()
        hits = [(p["payload"]["obs_id"], round(p["score"], 3)) for p in r.get("result", [])]
    else:
        r = store.http.post(f"{store.OS}/observations/_search",
                            json={"size": k, "query": {"multi_match": {"query": q, "fields": ["text^2", "place", "event_type"]}}}).json()
        hits = [(h["_source"]["obs_id"], round(h["_score"], 3)) for h in r.get("hits", {}).get("hits", [])]
    docs = {d[0]["obs_id"]: d[0] for d in rows("SELECT doc FROM observations WHERE obs_id = ANY(%s)", [h[0] for h in hits])}
    member_of = {}
    for doc, in rows("SELECT doc FROM events"):
        for m in doc["members"] + doc["duplicates"]:
            member_of[m] = doc["event_id"]
    return [{"obs_id": h, "score": s, "event_id": member_of.get(h), "source": docs[h]["source"], "text": docs[h]["text"],
             "time": docs[h].get("time"), "place": docs[h].get("place")} for h, s in hits if h in docs]


@app.get("/api/stats")
def stats():
    n_obs = rows("SELECT count(*), count(*) FILTER (WHERE doc->>'method' LIKE '%%llm%%') FROM observations")[0]
    n_ev = rows("SELECT count(*), count(*) FILTER (WHERE band='high'), count(*) FILTER (WHERE band='medium'), count(*) FILTER (WHERE band='low') FROM events")[0]
    rev = dict(rows("SELECT status, count(*) FROM reports WHERE status <> 'superseded' GROUP BY status"))
    last = rows("SELECT detail FROM pipeline_log WHERE stage = 'fuse' ORDER BY t DESC LIMIT 1")
    return {"observations": n_obs[0], "read_by_llm": n_obs[1], "events": n_ev[0],
            "bands": {"high": n_ev[1], "medium": n_ev[2], "low": n_ev[3]}, "reports": rev,
            "last_fusion": last[0][0] if last else None}
