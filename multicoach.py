# ambrose_migrated.py — single‑user, inline RAG, Neon‑optimized, safe migration
# - Keeps your existing memory/profiles tables intact
# - Migrates legacy RAG JSON (rag_data.json_data) into a proper table (rag_entries) with real[] embeddings
# - Inlines all RAG logic (no rag_utils import)
# - Uses one upserted memory row (as before) — no history cap
# - Retrieval supports optional neighbor expansion for better thread context
# - Idempotent migrations; does not delete legacy data

import os
import json
import base64
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import tiktoken
import streamlit as st
import openai
from sqlalchemy import text

# ──────────────────────────────────────────────────────────────────────────────
# DB connection
# ──────────────────────────────────────────────────────────────────────────────
conn = st.connection("neon_db", type="sql")

# ──────────────────────────────────────────────────────────────────────────────
# OpenAI client
# ──────────────────────────────────────────────────────────────────────────────
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OPENAI_API_KEY must be set in your environment variables or secrets.")
    st.stop()
client = openai.OpenAI(api_key=openai_api_key)

# ──────────────────────────────────────────────────────────────────────────────
# App constants
# ──────────────────────────────────────────────────────────────────────────────
MAX_TOKENS = 8000
PROFILE_UPDATE_THRESHOLD = 7500
UI_DISPLAY_MESSAGES = 100
MIN_ASSISTANT_LEN_FOR_RAG = 50  # skip tiny acks from being embedded
NEIGHBOR_PREV = 1               # neighbor expansion window (previous pairs)
NEIGHBOR_NEXT = 0               # set >0 if you also want next pairs
NEIGHBOR_MAX_AGE_HOURS = 12     # ignore neighbors if far apart in time
SAFETY_MARGIN = 512

# ──────────────────────────────────────────────────────────────────────────────
# Tokenizer helpers
# ──────────────────────────────────────────────────────────────────────────────

def _enc():
    try:
        return tiktoken.encoding_for_model("gpt-5")
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")
ENC = _enc()

def tok_len(s: str | None) -> int:
    return len(ENC.encode(s or ""))

# ──────────────────────────────────────────────────────────────────────────────
# Legacy schema init (unchanged tables)
# + New tables / migrations (idempotent)
# ──────────────────────────────────────────────────────────────────────────────

def initialize_database():
    """Create legacy tables if they don't exist; add new ones safely."""
    with conn.session as s:
        # legacy tables (as before)
        s.execute(text("""
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY,
                session_data TEXT NOT NULL,
                last_saved TEXT NOT NULL
            );
        """))
        s.execute(text("""
            CREATE TABLE IF NOT EXISTS profiles (
                id SERIAL PRIMARY KEY,
                profile_text TEXT NOT NULL,
                timestamp TEXT NOT NULL
            );
        """))
        s.execute(text("""
            CREATE TABLE IF NOT EXISTS rag_data (
                id INTEGER PRIMARY KEY,
                json_data TEXT NOT NULL
            );
        """))
        # migrations registry
        s.execute(text("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                name text PRIMARY KEY,
                applied_at timestamptz NOT NULL DEFAULT now()
            );
        """))
        # new RAG table with real[] embeddings
        s.execute(text("""
            CREATE TABLE IF NOT EXISTS rag_entries (
                id bigserial PRIMARY KEY,
                doc_id text UNIQUE NOT NULL,
                text_blob text NOT NULL,
                embedding real[] NOT NULL,
                ts timestamptz NOT NULL
            );
        """))
        # helpful indexes
        s.execute(text("CREATE INDEX IF NOT EXISTS ix_rag_entries_ts ON rag_entries(ts);"))
        s.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ux_memory_id ON memory(id);") )
        s.commit()


def migrate_rag_json_to_table():
    """One-time copy of entries from rag_data.json_data → rag_entries.
    Safe to re-run; skips already-inserted doc_ids. Does NOT delete rag_data.
    """
    MIGRATION_NAME = "2025-08-09-migrate-rag-json-to-table"
    with conn.session as s:
        # if already marked done, we still allow re-run but don't re-mark
        row = s.execute(text("SELECT 1 FROM schema_migrations WHERE name=:n"), {"n": MIGRATION_NAME}).fetchone()
        # read legacy JSON (if any)
        src = s.execute(text("SELECT json_data FROM rag_data WHERE id = 1")) .fetchone()
        if not src or not src[0]:
            if not row:
                # nothing to migrate, just mark
                s.execute(text("INSERT INTO schema_migrations(name) VALUES (:n) ON CONFLICT DO NOTHING"), {"n": MIGRATION_NAME})
                s.commit()
            return
        try:
            data = json.loads(src[0])
        except Exception:
            data = {"entries": []}

        entries = data.get("entries", [])
        imported = 0
        for e in entries:
            try:
                # prefer stable id from source if present
                doc_id = e.get("id") or f"legacy-{e.get('metadata',{}).get('timestamp','')}-{imported}"
                exists = s.execute(text("SELECT 1 FROM rag_entries WHERE doc_id=:d"), {"d": doc_id}).fetchone()
                if exists:
                    continue
                text_val = e.get("text", "")
                emb = e.get("embedding") or []
                # cast to float32 list to save space in real[]
                emb32 = np.array(emb, dtype=np.float32).tolist()
                ts = e.get("metadata", {}).get("timestamp") or datetime.now(timezone.utc).isoformat()
                try:
                    ts_dt = datetime.fromisoformat(ts.replace("Z","+00:00"))
                except Exception:
                    ts_dt = datetime.now(timezone.utc)
                payload = json.dumps({"text": text_val, "metadata": e.get("metadata", {})}, ensure_ascii=False)
                s.execute(text("""
                    INSERT INTO rag_entries(doc_id, text_blob, embedding, ts)
                    VALUES (:d, :b, :e, :ts)
                """), {"d": doc_id, "b": payload, "e": emb32, "ts": ts_dt})
                imported += 1
            except Exception:
                # skip any malformed rows; keep going
                continue
        if imported > 0:
            st.toast(f"Migrated {imported} RAG entries from legacy JSON → table.")
        # mark migration
        s.execute(text("INSERT INTO schema_migrations(name) VALUES (:n) ON CONFLICT DO NOTHING"), {"n": MIGRATION_NAME})
        s.commit()

# ──────────────────────────────────────────────────────────────────────────────
# Timestamptz migration for legacy TEXT columns
# ──────────────────────────────────────────────────────────────────────────────

def migrate_text_timestamps_to_timestamptz():
    """Convert legacy TEXT timestamps to TIMESTAMPTZ once (idempotent)."""
    MIG = "2025-08-09-text-ts-to-timestamptz"
    with conn.session as s:
        s.execute(text("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                name text PRIMARY KEY,
                applied_at timestamptz NOT NULL DEFAULT now()
            );
        """))
        done = s.execute(text("SELECT 1 FROM schema_migrations WHERE name=:n"), {"n": MIG}).fetchone()
        if not done:
            # profiles.timestamp (TEXT -> TIMESTAMPTZ)
            typ = s.execute(text("""
                SELECT data_type FROM information_schema.columns
                WHERE table_name='profiles' AND column_name='timestamp'
            """)).fetchone()
            if typ and typ[0] == 'text':
                s.execute(text("""
                    ALTER TABLE profiles
                    ALTER COLUMN "timestamp" TYPE timestamptz
                    USING ("timestamp"::timestamptz);
                """))
            # memory.last_saved (TEXT -> TIMESTAMPTZ)
            typ2 = s.execute(text("""
                SELECT data_type FROM information_schema.columns
                WHERE table_name='memory' AND column_name='last_saved'
            """)).fetchone()
            if typ2 and typ2[0] == 'text':
                s.execute(text("""
                    ALTER TABLE memory
                    ALTER COLUMN last_saved TYPE timestamptz
                    USING (last_saved::timestamptz);
                """))
            s.execute(text("INSERT INTO schema_migrations(name) VALUES (:n) ON CONFLICT DO NOTHING"), {"n": MIG})
        s.commit()

# ──────────────────────────────────────────────────────────────────────────────
# Profiles (legacy-compatible)
# ──────────────────────────────────────────────────────────────────────────────

def get_latest_profile() -> tuple[str | None, str | None]:
    with conn.session as s:
        res = s.execute(text("SELECT profile_text, timestamp FROM profiles ORDER BY timestamp DESC LIMIT 1;")) .fetchone()
    if res:
        return res[0], res[1]
    return None, None


def save_profile_text(text_val: str):
    ts = datetime.now(timezone.utc).isoformat()
    with conn.session as s:
        s.execute(text("INSERT INTO profiles (profile_text, timestamp) VALUES (:t, :ts)"), {"t": text_val, "ts": ts})
        s.commit()

# ──────────────────────────────────────────────────────────────────────────────
# Memory (single upserted row id=1 — already efficient)
# ──────────────────────────────────────────────────────────────────────────────

def load_memory() -> tuple[list, int]:
    with conn.session as s:
        result = s.execute(text("SELECT session_data FROM memory WHERE id = 1;")) .fetchone()
    if result:
        try:
            data = json.loads(result[0])
            return data.get("messages", []), data.get("tokens_since_last_profile", 0)
        except Exception:
            return [], 0
    return [], 0


def save_memory(history: list, tokens_count: int):
    data = {"messages": history, "tokens_since_last_profile": tokens_count}
    js = json.dumps(data)
    ts = datetime.now(timezone.utc).isoformat()
    with conn.session as s:
        s.execute(text("""
            INSERT INTO memory (id, session_data, last_saved)
            VALUES (1, :data, :ts)
            ON CONFLICT (id) DO UPDATE SET session_data = EXCLUDED.session_data, last_saved = EXCLUDED.last_saved;
        """), {"data": js, "ts": ts})
        s.commit()

# ──────────────────────────────────────────────────────────────────────────────
# RAG (inline) — embeddings in real[]; retrieval with neighbor expansion
# ──────────────────────────────────────────────────────────────────────────────

def get_embedding(text: str) -> list:
    r = client.embeddings.create(model="text-embedding-3-small", input=[text])
    # convert to float32 for storage efficiency
    return np.array(r.data[0].embedding, dtype=np.float32).tolist()


def add_chat_history_to_rag(chat_history: list):
    """Add assistant turns with preceding user context as RAG rows."""
    if len(chat_history) < 2:
        return
    with conn.session as s:
        for i, m in enumerate(chat_history):
            if m.get('role') != 'assistant':
                continue
            ts = m.get('timestamp')
            if not ts:
                continue
            doc_id = f"chatlog-{ts}"
            exists = s.execute(text("SELECT 1 FROM rag_entries WHERE doc_id=:d"), {"d": doc_id}).fetchone()
            if exists:
                continue
            reply = (m.get('content') or '').strip()
            if len(reply) < MIN_ASSISTANT_LEN_FOR_RAG:
                continue
            # gather preceding user msgs until previous assistant
            user_msgs = []
            for j in range(i-1, -1, -1):
                mm = chat_history[j]
                if mm.get('role') == 'user':
                    user_msgs.insert(0, mm)
                elif mm.get('role') == 'assistant':
                    break
            if not user_msgs:
                continue
            if len(user_msgs) == 1:
                text_doc = f"User: {user_msgs[0]['content']}\nAssistant: {reply}"
            else:
                text_doc = "\n".join(f"User: {um['content']}" for um in user_msgs) + f"\nAssistant: {reply}"
            meta = {
                "source": "chatlog",
                "timestamp": ts,
                "user_count": len(user_msgs),
                "assistant_content": reply[:100],
            }
            payload = json.dumps({"text": text_doc, "metadata": meta}, ensure_ascii=False)
            emb = get_embedding(text_doc)
            try:
                ts_dt = datetime.fromisoformat(ts.replace("Z","+00:00"))
            except Exception:
                ts_dt = datetime.now(timezone.utc)
            s.execute(text("""
                INSERT INTO rag_entries(doc_id, text_blob, embedding, ts)
                VALUES (:d, :b, :e, :ts)
            """), {"d": doc_id, "b": payload, "e": emb, "ts": ts_dt})
        s.commit()


def _decrypt_parse_blob(blob: str) -> dict:
    # no encryption in Ambrose single-user; keep helper for parity with Multicoach
    try:
        return json.loads(blob)
    except Exception:
        return {}


def rag_retrieve(query: str, n: int = 5, prev_neighbors: int = NEIGHBOR_PREV, next_neighbors: int = NEIGHBOR_NEXT, neighbor_max_age_hours: int = NEIGHBOR_MAX_AGE_HOURS) -> dict:
    """Semantic search on chat pairs, then expand each hit with adjacent pairs."""
    # load rows ordered by ts for neighbor traversal
    with conn.session as s:
        rows = s.execute(text("SELECT doc_id, text_blob, embedding, ts FROM rag_entries ORDER BY ts")) .fetchall()
    if not rows:
        return {"context_str": "No relevant context found.", "raw_results": {}}

    q = np.array(get_embedding(query))
    scored = []
    for doc_id, blob, emb, ts in rows:
        vec = np.array(emb, dtype=float)
        sim = float(np.dot(q, vec) / (np.linalg.norm(q) * np.linalg.norm(vec)))
        scored.append({"sim": sim, "doc_id": doc_id, "blob": blob, "ts": ts})

    scored.sort(key=lambda x: x["sim"], reverse=True)
    top = scored[:n]

    # index for neighbor lookup
    by_idx = list(enumerate(rows))
    doc_to_idx = {doc_id: idx for idx, (doc_id, *_rest) in by_idx}

    selected_ids: list[str] = []
    selected_map: dict[str, dict] = {}

    def maybe_add(doc_id: str, blob: str, ts_val):
        if doc_id in selected_ids:
            return
        rec = _decrypt_parse_blob(blob)
        if not rec:
            return
        src = (rec.get("metadata") or {}).get("source")
        if src and src != "chatlog":
            return
        selected_ids.append(doc_id)
        selected_map[doc_id] = {"text": rec.get("text", ""), "ts": ts_val, "meta": rec.get("metadata", {})}

    for hit in top:
        idx = doc_to_idx.get(hit["doc_id"])
        if idx is None:
            continue
        # add hit
        maybe_add(hit["doc_id"], hit["blob"], hit["ts"])
        # prev neighbors
        for k in range(1, (prev_neighbors or 0) + 1):
            j = idx - k
            if j < 0:
                break
            d_id, b, _e, ts_prev = rows[j]
            if (hit["ts"] - ts_prev).total_seconds() > (neighbor_max_age_hours * 3600):
                break
            maybe_add(d_id, b, ts_prev)
        # next neighbors (optional)
        for k in range(1, (next_neighbors or 0) + 1):
            j = idx + k
            if j >= len(rows):
                break
            d_id, b, _e, ts_next = rows[j]
            if (ts_next - hit["ts"]).total_seconds() > (neighbor_max_age_hours * 3600):
                break
            maybe_add(d_id, b, ts_next)

    selected_ids.sort(key=lambda did: selected_map[did]["ts"])  # chronological
    docs = [selected_map[did]["text"] for did in selected_ids]
    ctx = "\n\n---\n\n".join(docs) if docs else "No relevant context found."

    return {"context_str": ctx, "raw_results": {"doc_ids": selected_ids, "count": len(selected_ids)}}


def debug_rag_status() -> dict:
    with conn.session as s:
        total = s.execute(text("SELECT COUNT(*) FROM rag_entries")).fetchone()[0]
        approx = s.execute(text("SELECT COALESCE(SUM(array_length(embedding,1)),0) FROM rag_entries")).fetchone()[0] or 0
    return {"total_rows": total, "avg_dims": approx / max(total, 1)}

# ──────────────────────────────────────────────────────────────────────────────
# Housekeeping helpers (optional, non-destructive paths)
# ──────────────────────────────────────────────────────────────────────────────

# --- Danger: wipe this app’s schema (tables only) ---

def reset_app_schema():
    """Drops this app's tables in the configured schema, then reruns."""
    schema = st.secrets.get("DB_SCHEMA")
    with conn.session as s:
        # Ensure schema exists & set search_path when provided
        if schema:
            s.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}";'))
            s.execute(text(f'SET search_path TO "{schema}";'))
            prefix = f'"{schema}".'
        else:
            prefix = ''
        # Drop only our app tables
        for tbl in ["rag_entries", "memory", "profiles", "schema_migrations", "rag_data"]:
            s.execute(text(f"DROP TABLE IF EXISTS {prefix}{tbl} CASCADE;"))
        s.commit()
    st.success("App schema wiped. Recreating…")
    st.rerun()

# Optional one-shot schema reset via Streamlit secret (guarded to avoid loops)
if str(st.secrets.get("RESET_SCHEMA_ON_START", "false")).lower() in ("1", "true", "yes"):
    if not st.session_state.get("_schema_reset_once"):
        st.session_state["_schema_reset_once"] = True
        reset_app_schema()
    # else: already reset once this session; ignore


def export_legacy_rag_json() -> str | None:
    """Return legacy rag_data JSON blob if present (for backup/export)."""
    with conn.session as s:
        row = s.execute(text("SELECT json_data FROM rag_data WHERE id=1")).fetchone()
    return row[0] if row and row[0] else None


def purge_legacy_rag_json():
    """Delete legacy rag_data row after you've exported a backup. New rag_entries table remains."""
    with conn.session as s:
        s.execute(text("DELETE FROM rag_data;"))
        s.commit()


def normalize_memory_table() -> int:
    """Keep only the latest memory snapshot at id=1; remove any stray rows (defensive). Returns deleted count."""
    with conn.session as s:
        rows = s.execute(text("SELECT id, session_data, last_saved FROM memory ORDER BY last_saved DESC")).fetchall()
        if not rows:
            return 0
        latest = rows[0]
        s.execute(text("""
            INSERT INTO memory (id, session_data, last_saved)
            VALUES (1, :data, :ts)
            ON CONFLICT (id) DO UPDATE SET session_data=EXCLUDED.session_data, last_saved=EXCLUDED.last_saved;
        """), {"data": latest[1], "ts": latest[2]})
        deleted = s.execute(text("DELETE FROM memory WHERE id <> 1")).rowcount
        s.commit()
        return deleted


def vacuum_tables():
    """Optional: ask Postgres to reclaim/analyze space. Safe to skip if provider restricts it."""
    with conn.session as s:
        try:
            s.execute(text("VACUUM ANALYZE rag_entries;"))
            s.execute(text("VACUUM ANALYZE memory;"))
            s.execute(text("VACUUM ANALYZE profiles;"))
            s.commit()
        except Exception as e:
            # Surface to caller; some serverless plans may limit VACUUM
            raise e

# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────────

def format_timedelta(delta: timedelta) -> str:
    days, seconds = delta.days, delta.seconds
    if days > 1: return f"{days} days"
    if days == 1: return "1 day"
    hours = seconds // 3600
    if hours > 1: return f"{hours} hours"
    if hours == 1: return "1 hour"
    minutes = (seconds % 3600) // 60
    if minutes > 5: return f"{minutes} minutes"
    return ""


def truncate_history(history: list, max_tokens: int) -> list:
    truncated_history, current_tokens = [], 0
    for message in reversed(history):
        content = message.get("content", "")
        if isinstance(content, str):
            message_tokens = tok_len(content)
            if current_tokens + message_tokens <= max_tokens:
                truncated_history.insert(0, message)
                current_tokens += message_tokens
            else:
                break
    return truncated_history


def get_image_as_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# ──────────────────────────────────────────────────────────────────────────────
# Profiler (unchanged behavior; saves into profiles)
# ──────────────────────────────────────────────────────────────────────────────

def run_periodic_profiler(chat_history: list):
    st.toast("Synthesizing new user profile in the background…")
    old_profile_content, _ = get_latest_profile()
    if old_profile_content is None:
        old_profile_content = "No previous file exists for Mr. Hansen."
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history if m.get('content')])

    profiler_prompt = f"""
You are the personal chronicler for Ambrose, a wise and insightful personal mentor. Your task is to maintain a detailed, yet concise, personal file on his client, Mr. Hansen. You will write in a clear, objective, but slightly formal British style, mirroring Ambrose's own manner.

**Your Instructions:**
1. Review the existing 'PERSONAL FILE' on Mr. Hansen.
2. Analyze the 'RECENT CORRESPONDENCE'.
3. **Only derive facts from lines prefixed with `user:`.** Assistant lines are context only.
4. Produce a consolidated, non-redundant 'UPDATED PERSONAL FILE'.
5. Maintain continuity; don't overweight the current topic.
6. **Output ONLY** the updated file text.

--- PERSONAL FILE (PREVIOUS) ---
{old_profile_content}

--- RECENT CORRESPONDENCE ---
{history_text}

--- UPDATED PERSONAL FILE for Mr. Hansen ---
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "system", "content": profiler_prompt}],
            temperature=0.2,
        )
        new_profile_content = response.choices[0].message.content
        save_profile_text(new_profile_content)
        st.toast("✅ User profile updated successfully in the database!")
    except Exception as e:
        st.error(f"Failed to update profile: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# Response streaming
# ──────────────────────────────────────────────────────────────────────────────

def get_response_stream(user_query: str, system_prompt: str, chat_history: list, profile_text: str, time_gap_note: str):
    profile_summary = f"\n--- USER PROFILE ---\n{profile_text}"
    messages = [{"role": "system", "content": system_prompt}] + chat_history
    if time_gap_note:
        messages.append({"role": "system", "content": f"[System note: It has been {time_gap_note} since your last exchange. Acknowledge this appropriately if relevant.]"})
    augmented_query = f"{profile_summary}\n\n--- USER QUERY ---\n{user_query}"
    messages.append({"role": "user", "content": augmented_query})
    try:
        response = client.chat.completions.create(model="gpt-5", messages=messages, stream=True)
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        st.error(f"An error occurred: {e}")
        yield "Apologies, an error occurred while processing your request."

# ──────────────────────────────────────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────────────────────────────────────
initialize_database()
migrate_rag_json_to_table()  # safe: copies legacy rag_data → rag_entries if present
migrate_text_timestamps_to_timestamptz()  # convert TEXT timestamps → timestamptz once

st.set_page_config(layout="wide", page_title="Ambrose AI", page_icon="emperoricon.png")

prompt_file_path = Path(__file__).parent / 'system_prompt.txt'

def load_prompt(file_path: str) -> str:
    try:
        return Path(file_path).read_text(encoding='utf-8')
    except FileNotFoundError:
        return "You are a helpful assistant."

SYSTEM_PROMPT = load_prompt(str(prompt_file_path))

# Session state
if "messages" not in st.session_state:
    messages, _ = load_memory()
    st.session_state.messages = messages
    # conservatively recompute token count since last profile
    def recalc_tokens(messages: list) -> int:
        try:
            encoder = tiktoken.get_encoding("cl100k_base")
            _, latest_profile_ts = get_latest_profile()
            total = 0
            for m in messages:
                ts = m.get("timestamp")
                if (latest_profile_ts is None) or (ts and ts > latest_profile_ts):
                    c = m.get("content", "")
                    if isinstance(c, str) and c.strip():
                        total += len(encoder.encode(c))
            return total
        except Exception:
            return len(messages) * 50
    st.session_state.tokens_since_last_profile = recalc_tokens(messages)

# Sidebar
with st.sidebar:
    gif_path = Path(__file__).parent / "emperor.gif"
    if gif_path.exists():
        st.markdown(
            f"""
        <div style='text-align: center; margin-bottom: 20px;'>
            <img src='data:image/gif;base64,{get_image_as_base64(gif_path)}' alt='emperor gif' width='150'>
        </div>
        """,
            unsafe_allow_html=True,
        )
    st.header("⚙️ Controls")
    st.subheader("Profile Synthesis")
    tokens_count = st.session_state.get('tokens_since_last_profile', 0)
    progress_percent = min(tokens_count / PROFILE_UPDATE_THRESHOLD, 1.0)
    st.progress(progress_percent)
    st.markdown(f"**{tokens_count} / {PROFILE_UPDATE_THRESHOLD}** tokens until next update.")

    st.subheader("Most Recent Profile")
    latest_profile_text, latest_profile_timestamp = get_latest_profile()
    if latest_profile_text and latest_profile_timestamp:
        try:
            dt_object = datetime.fromisoformat(str(latest_profile_timestamp).replace("Z","+00:00")).astimezone(timezone.utc)
            formatted_ts = dt_object.strftime('%b %d, %Y, %H:%M %Z')
            st.caption(f"Last updated: {formatted_ts}")
        except Exception:
            st.caption(f"Last updated: {latest_profile_timestamp}")
        with st.expander("View Profile"):
            st.text_area("Profile Content:", latest_profile_text, height=300, disabled=True)
    else:
        st.info("No profile has been created yet.")

    if st.button("Wipe Memory & Profiles"):
        st.session_state.messages, st.session_state.tokens_since_last_profile = [], 0
        with conn.session as s:
            s.execute(text("DELETE FROM memory;"))
            s.execute(text("DELETE FROM profiles;"))
            # keep rag_entries and legacy rag_data — user asked to retain long-tail memory
            s.commit()
        if "last_rag_results" in st.session_state:
            del st.session_state.last_rag_results
        st.toast("✅ Reset chat & profiles. RAG store kept.")
        st.rerun()

    st.divider()
    st.subheader("Data housekeeping")
    try:
        legacy_json = export_legacy_rag_json()
    except Exception as e:
        legacy_json = None
        st.caption(f"Legacy rag_data check failed: {e}")

    if legacy_json:
        st.caption("Legacy RAG JSON from an older app version is still stored.")
        st.download_button(
            label="Download legacy RAG JSON backup",
            data=legacy_json,
            file_name=f"ambrose_rag_legacy_{datetime.now(timezone.utc).strftime('%Y%m%d')}.json",
            mime="application/json",
        )
        confirm = st.checkbox("I have saved a backup and want to purge the old rag_data JSON")
        if st.button("Purge legacy rag_data now") and confirm:
            try:
                purge_legacy_rag_json()
                st.success("Purged legacy rag_data. New RAG table remains.")
            except Exception as e:
                st.error(f"Purge failed: {e}")
    else:
        st.caption("No legacy rag_data JSON detected.")

    if st.button("Normalize memory table (keep latest snapshot only)"):
        try:
            removed = normalize_memory_table()
            st.success(f"Memory table normalized. Removed {removed} old row(s) (if any).")
        except Exception as e:
            st.error(f"Normalize failed: {e}")

    if st.button("Run VACUUM ANALYZE (optional)"):
        try:
            vacuum_tables()
            st.toast("VACUUM/ANALYZE requested. It may take a moment.")
        except Exception as e:
            st.error(f"VACUUM failed: {e}")

    st.divider()
    st.subheader("Danger zone")
    if st.button("Wipe ALL app data (this schema)"):
        reset_app_schema()

# Render chat
display_messages = st.session_state.messages[-UI_DISPLAY_MESSAGES:]
for message in display_messages:
    if message.get("role") in ["user", "assistant"] and message.get("content"):
        avatar_icon = "⭐" if message["role"] == "user" else "🧙‍♂️"
        with st.chat_message(message["role"], avatar=avatar_icon):
            st.markdown(message["content"])

# Main chat loop
if prompt := st.chat_input("Awaiting your query, Mr. Hansen..."):
    time_gap_note = ""
    if st.session_state.messages and "timestamp" in st.session_state.messages[-1]:
        try:
            last_message_time = datetime.fromisoformat(st.session_state.messages[-1].get("timestamp").replace("Z","+00:00"))
            time_gap = datetime.now(timezone.utc) - last_message_time
            time_gap_note = format_timedelta(time_gap)
        except Exception:
            time_gap_note = ""

    user_message = {"role": "user", "content": prompt, "timestamp": datetime.now(timezone.utc).isoformat()}
    st.session_state.messages.append(user_message)
    with st.chat_message("user", avatar="⭐"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🧙‍♂️"):
        retrieval_output = rag_retrieve(prompt, n=5, prev_neighbors=NEIGHBOR_PREV, next_neighbors=NEIGHBOR_NEXT)
        rag_context = retrieval_output['context_str']
        st.session_state.last_rag_results = retrieval_output['raw_results']

        augmented_system_prompt = f"""{SYSTEM_PROMPT}

--- RELEVANT CONTEXT FROM YOUR MEMORY ---
You have recalled the following information that may be relevant to the user's query. Use it to inform your response if appropriate. Do not mention this context unless the user asks about it.
{rag_context}
--- END OF CONTEXT ---
"""
        truncated_chat = truncate_history(st.session_state.messages, MAX_TOKENS - SAFETY_MARGIN)
        current_profile_text, _ = get_latest_profile()
        if current_profile_text is None:
            current_profile_text = "No profile created yet."

        response_stream = get_response_stream(prompt, augmented_system_prompt, truncated_chat, current_profile_text, time_gap_note)
        full_response = st.write_stream(response_stream)

    assistant_message = {"role": "assistant", "content": full_response, "timestamp": datetime.now(timezone.utc).isoformat()}
    st.session_state.messages.append(assistant_message)

    # Update RAG from this turn
    add_chat_history_to_rag(st.session_state.messages)

    # Token accounting for profiler trigger
    try:
        tokens_this_turn = tok_len(prompt) + tok_len(full_response)
        st.session_state.tokens_since_last_profile += tokens_this_turn
    except Exception:
        pass

    if st.session_state.tokens_since_last_profile > PROFILE_UPDATE_THRESHOLD:
        recent_history = st.session_state.messages[-20:]
        run_periodic_profiler(recent_history)
        st.session_state.tokens_since_last_profile = 0
        audit_message = {
            "role": "system",
            "content": "[System Note: User profile was synthesized. Token counter reset.]",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        st.session_state.messages.append(audit_message)

    save_memory(st.session_state.messages, st.session_state.tokens_since_last_profile)

# Debug Tools
if st.checkbox("Show Debug Info", False):
    st.subheader("Raw Message Data")
    for i, message in enumerate(st.session_state.messages):
        st.write(f"Message {i+1}:")
        st.json(message)
        st.write("---")

    st.subheader("RAG Debug Tools")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Force RAG Update"):
            add_chat_history_to_rag(st.session_state.messages)
            st.write("Added any missing assistant turns to RAG.")
    with col2:
        if st.button("Check RAG Status"):
            st.json(debug_rag_status())

    st.subheader("Last RAG Retrieval")
    if st.session_state.get('last_rag_results'):
        st.json(st.session_state['last_rag_results'])
    else:
        st.write("No RAG retrieval has happened in this session yet.")
