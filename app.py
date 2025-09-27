# app.py
# Streamlit app with interactive practice loop for DKT JEE model.
# Place at project root (where 'models/' and 'processed/' live).
# Auto-detects question CSV inside processed/.

import os
from pathlib import Path
import json
from typing import List, Dict
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow import keras

# -------------------- Configuration --------------------
BASE_DIR = os.environ.get("EDNET_BASE_DIR", ".")
MODEL_REL = "models/dkt_model_jee_adapted_finetuned.keras"
QMAP_REL = "processed/kt_numpy/qid2idx.json"
QBANK_DIR = Path(BASE_DIR) / "processed"
MAX_SEQ_LEN = 200
PAD_INDEX = 0

st.set_page_config(page_title="DKT JEE Interactive Practice", layout="wide")

# -------------------- Utilities --------------------
def find_question_csv(qbank_dir: Path) -> Path:
    # look for g_jee_questions_*.csv inside processed/
    patterns = ["g_jee_questions_merged_650.csv", "g_jee_questions_merged_650 (1).csv", "g_jee_questions_*.csv"]
    for p in patterns:
        matches = list(qbank_dir.glob(p))
        if matches:
            return matches[0]
    # fallback: any csv in processed
    csvs = list(qbank_dir.glob("*.csv"))
    if csvs:
        return csvs[0]
    raise FileNotFoundError(f"No question CSV found in {qbank_dir}")

# cache heavy loads
@st.cache_resource(show_spinner=False)
def load_resources(base_dir: str = BASE_DIR):
    base = Path(base_dir)
    model_path = base / MODEL_REL
    qmap_path = base / QMAP_REL
    qbank_path = find_question_csv(base / "processed")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not qmap_path.exists():
        raise FileNotFoundError(f"QID map not found: {qmap_path}")
    if not qbank_path.exists():
        raise FileNotFoundError(f"Question bank not found: {qbank_path}")

    model = keras.models.load_model(str(model_path), compile=False)

    with open(qmap_path, "r") as f:
        qid2idx = json.load(f)
    qid2idx = {str(k): int(v) for k,v in qid2idx.items()}

    qdf = pd.read_csv(qbank_path, dtype=str)
    if "difficulty" in qdf.columns:
        qdf["difficulty"] = pd.to_numeric(qdf["difficulty"], errors="coerce").fillna(0.5)
    else:
        qdf["difficulty"] = 0.5
    # ensure question_id column exists and is string
    if "question_id" not in qdf.columns:
        raise RuntimeError(f"Question bank CSV must have 'question_id' column: {qbank_path}")
    qdf["question_id"] = qdf["question_id"].astype(str)

    # convenience lookup dict
    qdict = qdf.set_index("question_id").to_dict("index")
    return model, qid2idx, qdf, qdict

def encode_student_history(history_qids: List[str], history_correct: List[int], qid2idx: Dict[str,int], max_seq_len: int = MAX_SEQ_LEN):
    if history_qids is None:
        history_qids = []
    if history_correct is None:
        history_correct = []
    assert len(history_qids) == len(history_correct), "history_qids and history_correct must align"
    mapped = [qid2idx.get(str(q), PAD_INDEX) for q in history_qids]
    resps = [int(x) for x in history_correct]
    if len(mapped) > max_seq_len:
        mapped = mapped[-max_seq_len:]
        resps = resps[-max_seq_len:]
    seq_len = len(mapped)
    pad_len = max_seq_len - seq_len
    q_padded = [PAD_INDEX] * pad_len + mapped
    r_padded = [0] * pad_len + resps
    items_input = np.array(q_padded[:-1], dtype=np.int32).reshape(1, -1)
    resps_input = np.array(r_padded[:-1], dtype=np.float32).reshape(1, -1)
    return items_input, resps_input

def predict_prob_for_candidate(model, qid2idx, input_items, input_resps, candidate_qid):
    cand_idx = qid2idx.get(str(candidate_qid), PAD_INDEX) if (candidate_qid := candidate_qid) else PAD_INDEX
    items = input_items.copy()
    resps = input_resps.copy()
    items[0, -1] = cand_idx
    resps[0, -1] = 0.0
    preds = model.predict([items, resps], verbose=0)
    preds = np.array(preds)
    if preds.ndim == 3 and preds.shape[-1] == 1:
        preds = preds.reshape(preds.shape[0], preds.shape[1])
    prob = float(preds[0, -1])
    return prob

def compute_chapter_mastery(history_qids: List[str], history_correct: List[int], qdf: pd.DataFrame):
    df = pd.DataFrame({"question_id": history_qids, "correct": history_correct})
    if df.empty:
        return {}
    merged = df.merge(qdf[['question_id','chapter']].drop_duplicates(), on='question_id', how='left')
    merged['chapter'] = merged['chapter'].fillna('unknown')
    return merged.groupby('chapter')['correct'].mean().to_dict()

def recommend_next_questions(model, qid2idx, qdf, history_qids, history_correct, k=5, candidates_pool_size=200, target_prob=0.6):
    mastery = compute_chapter_mastery(history_qids, history_correct, qdf)
    all_chapters = qdf['chapter'].fillna('unknown').unique().tolist()
    ch_priority = {ch: (1.0 - mastery.get(ch, 0.0)) for ch in all_chapters}
    attempted = set(history_qids or [])
    pool = qdf[~qdf['question_id'].isin(attempted)].copy()
    if pool.empty:
        pool = qdf.copy()
    if len(pool) > candidates_pool_size:
        pool = pool.sample(n=candidates_pool_size, random_state=42)
    items_in, resps_in = encode_student_history(history_qids, history_correct, qid2idx, max_seq_len=MAX_SEQ_LEN)
    scored = []
    for _, row in pool.iterrows():
        qid = str(row['question_id'])
        prob = predict_prob_for_candidate(model, qid2idx, items_in, resps_in, qid)
        scored.append({
            'question_id': qid,
            'question': row.get('question',''),
            'chapter': row.get('chapter',''),
            'difficulty': float(row.get('difficulty', 0.5) or 0.5),
            'predicted_prob': float(prob),
            'chapter_priority': ch_priority.get(row.get('chapter',''), 1.0)
        })
    for s in scored:
        s['score'] = s['chapter_priority'] - abs(s['predicted_prob'] - target_prob)
    scored_sorted = sorted(scored, key=lambda x: x['score'], reverse=True)
    return scored_sorted[:k]

def sample_coldstart(qdf, k=10):
    med = qdf[(qdf['difficulty']>=0.35) & (qdf['difficulty']<=0.65)]
    if len(med) >= k:
        return med.sample(n=k, random_state=42).to_dict('records')
    else:
        extra = qdf.sample(n=k, random_state=42)
        return extra.to_dict('records')

# -------------------- Load resources --------------------
with st.spinner("Loading model and question bank (this may take a few seconds)..."):
    try:
        model, qid2idx, qdf, qdict = load_resources(BASE_DIR)
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        st.stop()

# -------------------- UI --------------------
st.title("DKT (LSTM) — JEE Interactive Practice")
st.markdown("Streamlit UI: cold-start practice + manual testing + recommendations")

# sidebar
st.sidebar.title("Settings")
K = st.sidebar.number_input("Top-k recommendations", min_value=1, max_value=20, value=5)
POOL = st.sidebar.number_input("Candidate pool size", min_value=20, max_value=1000, value=200)

# main layout
col1, col2 = st.columns([1,2])

with col1:
    st.subheader("Manual History Input")
    st.write("Paste a comma-separated list of question_ids and 0/1 answers (chronological).")
    q_input = st.text_area("Question IDs", value="", placeholder="Ph_Mec_11_0, Ph_Mec_11_4")
    ans_input = st.text_input("Answers (0/1)", value="", placeholder="1,0")
    if st.button("Load demo history"):
        sample_demo = qdf.sample(n=6, random_state=1)
        st.session_state['manual_q_input'] = ", ".join(sample_demo['question_id'].tolist())
        st.session_state['manual_ans_input'] = ", ".join(map(str, np.random.randint(0,2,len(sample_demo))))
        # safe rerun for different streamlit versions
        try:
            st.rerun()
        except Exception:
            try:
                st.experimental_rerun()
            except Exception:
                pass

    history_qids = [s.strip() for s in (q_input or st.session_state.get('manual_q_input','')).split(",") if s.strip()]
    history_correct = []
    ans_text = ans_input or st.session_state.get('manual_ans_input','')
    if ans_text.strip():
        for s in ans_text.split(","):
            s2 = s.strip()
            if s2=="":
                continue
            history_correct.append(int(s2))
    if len(history_correct) < len(history_qids):
        history_correct += [0] * (len(history_qids) - len(history_correct))
    if len(history_correct) > len(history_qids):
        history_correct = history_correct[:len(history_qids)]

    st.write(f"History length: {len(history_qids)}")
    if st.checkbox("Show history table"):
        rows = []
        for q,c in zip(history_qids, history_correct):
            meta = qdict.get(q, {})
            rows.append({"question_id": q, "chapter": meta.get("chapter",""), "difficulty": meta.get("difficulty",""), "correct": c})
        st.dataframe(pd.DataFrame(rows))

    if st.button("Compute chapter mastery"):
        mastery = compute_chapter_mastery(history_qids, history_correct, qdf)
        st.json(mastery)

with col2:
    st.subheader("Candidate scoring & Recommendation")
    st.write("You can provide a candidate question id to score, or use recommender for next question.")

    candidate_qid = st.text_input("Candidate question_id (optional)", value="", placeholder="Ph_Mec_11_4")
    items_in, resps_in = encode_student_history(history_qids, history_correct, qid2idx)
    if candidate_qid.strip():
        prob = predict_prob_for_candidate(model, qid2idx, items_in, resps_in, candidate_qid.strip())
        st.metric(f"P(correct) for {candidate_qid.strip()}", f"{prob:.3f}")
        meta = qdict.get(candidate_qid.strip(), {})
        if meta:
            st.write(meta.get("question",""))
            for opt in ["option_a","option_b","option_c","option_d"]:
                if meta.get(opt):
                    st.write(f"- {meta.get(opt)}")

    if st.button("Recommend next questions"):
        recs = recommend_next_questions(model, qid2idx, qdf, history_qids, history_correct, k=K, candidates_pool_size=POOL)
        display_rows = []
        for r in recs:
            display_rows.append({
                "question_id": r['question_id'],
                "chapter": r['chapter'],
                "difficulty": round(r['difficulty'],3),
                "pred_prob": round(r['predicted_prob'],3),
                "score": round(r['score'],4)
            })
        st.dataframe(pd.DataFrame(display_rows))

    if st.button("Cold-start pick (k=10)"):
        cs = sample_coldstart(qdf, k=10)
        st.write("Cold-start questions (sample):")
        st.table(pd.DataFrame(cs)[["question_id","chapter","difficulty","question"]])

    st.write("---")
    st.write("Question preview")
    qid_preview = st.text_input("Preview question_id", value="", placeholder="Ph_Mec_11_4")
    if qid_preview.strip():
        meta = qdict.get(qid_preview.strip())
        if meta:
            st.markdown(f"**{qid_preview.strip()}** — Chapter: {meta.get('chapter','')}, Difficulty: {meta.get('difficulty','')}")
            st.write(meta.get("question",""))
            for opt in ["option_a","option_b","option_c","option_d"]:
                if meta.get(opt):
                    st.write(f"- {meta.get(opt)}")
        else:
            st.warning("Question id not found.")

# -------------------- Interactive live practice loop --------------------
st.markdown("---")
st.subheader("Interactive Practice (cold-start -> adaptive)")

# session state keys
if 'candidate_queue' not in st.session_state:
    st.session_state['candidate_queue'] = []
if 'history_qids_live' not in st.session_state:
    st.session_state['history_qids_live'] = []
if 'history_correct_live' not in st.session_state:
    st.session_state['history_correct_live'] = []

cols = st.columns([1,1,1])
with cols[0]:
    if st.button("Start cold-start practice (10)"):
        cs = sample_coldstart(qdf, k=10)
        st.session_state['candidate_queue'] = [c['question_id'] for c in cs]
        st.session_state['history_qids_live'] = []
        st.session_state['history_correct_live'] = []
        try:
            st.rerun()
        except Exception:
            try:
                st.experimental_rerun()
            except Exception:
                pass

with cols[1]:
    if st.button("Reset live session"):
        st.session_state['candidate_queue'] = []
        st.session_state['history_qids_live'] = []
        st.session_state['history_correct_live'] = []
        try:
            st.rerun()
        except Exception:
            try:
                st.experimental_rerun()
            except Exception:
                pass

with cols[2]:
    st.write(f"Live history length: {len(st.session_state['history_qids_live'])}")

# display current candidate
if st.session_state['candidate_queue']:
    current = st.session_state['candidate_queue'][0]
    st.markdown("### Current Question (Live)")
    meta = qdict.get(current, {})
    st.write(f"**{current}** — {meta.get('question','(no text)')}")
    opt_labels = []
    for label in ["A","B","C","D"]:
        opt_text = meta.get(f"option_{label.lower()}", "")
        if opt_text:
            opt_labels.append((label, opt_text))

    if opt_labels:
        # show the actual option text as "A: <text>"
        opt_display = [f"{lab}: {txt}" for lab, txt in opt_labels]
        selected = st.radio("Choose answer", options=opt_display, index=0, key="live_choice")
        # extract chosen label like "A" from "A: <text>"
        selected_label = selected.split(":")[0].strip()
        if st.button("Submit answer (live)"):
            correct_ans = (meta.get("correct_answer","") or "").strip().upper()
            is_correct = 1 if selected_label == correct_ans else 0
            st.session_state['history_qids_live'].append(current)
            st.session_state['history_correct_live'].append(is_correct)
            st.session_state['candidate_queue'].pop(0)
            # get a new recommendation (top-1) and insert as next
            recs = recommend_next_questions(model, qid2idx, qdf, st.session_state['history_qids_live'],
                                           st.session_state['history_correct_live'], k=1, candidates_pool_size=200)
            if recs:
                st.session_state['candidate_queue'].insert(0, recs[0]['question_id'])
            try:
                st.rerun()
            except Exception:
                try:
                    st.experimental_rerun()
                except Exception:
                    pass
    else:
        st.write("No options available for this question. Skipping.")
        st.session_state['candidate_queue'].pop(0)
        try:
            st.rerun()
        except Exception:
            try:
                st.experimental_rerun()
            except Exception:
                pass
else:
    st.info("No active live candidate. Click 'Start cold-start practice' to begin.")

# show live history table
if st.session_state['history_qids_live']:
    st.markdown("#### Live session history (most recent last)")
    rows = []
    for q,c in zip(st.session_state['history_qids_live'], st.session_state['history_correct_live']):
        meta = qdict.get(q, {})
        rows.append({"question_id": q, "chapter": meta.get("chapter",""), "difficulty": meta.get("difficulty",""), "correct": c})
    st.dataframe(pd.DataFrame(rows))

# footer info
st.sidebar.markdown("---")
st.sidebar.write(f"Model layers: {len(model.layers)}")
st.sidebar.write(f"Questions: {len(qdf)}")
st.sidebar.write(f"Mapping size: {len(qid2idx)}")
