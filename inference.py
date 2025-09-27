"""
inference.py

Small inference module for DKT model adapted -> JEE.

Functions:
 - load_resources(base_dir=None)
 - encode_student_history(history_qids, history_correct, qid2idx, max_seq_len)
 - predict_prob_for_candidate(model, qid2idx, items_in, resps_in, candidate_qid)
 - compute_chapter_mastery(history_qids, history_correct, qdf)
 - recommend_next_questions(...)
 - sample_coldstart(qdf, k)
"""

import os
import json
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras

# --- Adjustable defaults ---
DEFAULT_BASE_DIR = os.environ.get("EDNET_BASE_DIR", ".")  # project root (where models/ and processed/ are)
DEFAULT_MODEL_REL = "models/dkt_model_jee_adapted_finetuned.keras"
DEFAULT_QMAP_REL = "processed/kt_numpy/qid2idx.json"
DEFAULT_QBANK_REL = "processed/g_jee_questions_merged_650.csv"
MAX_SEQ_LEN = 200
PAD_INDEX = 0

# -------------------------
def load_resources(base_dir: Optional[str] = None,
                   model_rel: str = DEFAULT_MODEL_REL,
                   qmap_rel: str = DEFAULT_QMAP_REL,
                   qbank_rel: str = DEFAULT_QBANK_REL):
    base = Path(base_dir or DEFAULT_BASE_DIR)
    model_path = base / model_rel
    qmap_path = base / qmap_rel
    qbank_path = base / qbank_rel

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not qmap_path.exists():
        raise FileNotFoundError(f"QID map not found: {qmap_path}")
    if not qbank_path.exists():
        raise FileNotFoundError(f"Question bank not found: {qbank_path}")

    model = keras.models.load_model(str(model_path), compile=False)
    with open(qmap_path, "r") as f:
        qid2idx = json.load(f)
    # ensure ints
    qid2idx = {str(k): int(v) for k, v in qid2idx.items()}

    qdf = pd.read_csv(qbank_path, dtype=str)
    if "difficulty" in qdf.columns:
        qdf["difficulty"] = pd.to_numeric(qdf["difficulty"], errors="coerce").fillna(0.5)
    else:
        qdf["difficulty"] = 0.5

    # ensure question_id column exists
    if "question_id" not in qdf.columns:
        raise RuntimeError("Question bank must have 'question_id' column")

    # keep question_id as string column
    qdf["question_id"] = qdf["question_id"].astype(str)

    return model, qid2idx, qdf


def encode_student_history(history_qids: List[str],
                           history_correct: List[int],
                           qid2idx: Dict[str,int],
                           max_seq_len: int = MAX_SEQ_LEN) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return items_input, resps_input shapes (1, max_seq_len-1)
    History lists must be chronological (oldest -> newest)
    """
    if history_qids is None:
        history_qids = []
    if history_correct is None:
        history_correct = []
    assert len(history_qids) == len(history_correct), "history_qids and history_correct must align"

    mapped = [qid2idx.get(str(q), PAD_INDEX) for q in history_qids]
    resps = [int(x) for x in history_correct]

    # truncate to last max_seq_len tokens if needed
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


def predict_prob_for_candidate(model: keras.Model,
                               qid2idx: Dict[str,int],
                               input_items: np.ndarray,
                               input_resps: np.ndarray,
                               candidate_qid: str) -> float:
    """
    Append candidate index at the final input position and get model prediction for the final timestep.
    """
    cand_idx = qid2idx.get(str(candidate_qid), PAD_INDEX)
    items = input_items.copy()
    resps = input_resps.copy()
    items[0, -1] = cand_idx
    resps[0, -1] = 0.0  # unknown response — we want to predict it
    preds = model.predict([items, resps], verbose=0)
    preds = np.array(preds)
    # normalize shape
    if preds.ndim == 3 and preds.shape[-1] == 1:
        preds = preds.reshape(preds.shape[0], preds.shape[1])
    prob = float(preds[0, -1])
    return prob


def compute_chapter_mastery(history_qids: List[str], history_correct: List[int], qdf: pd.DataFrame) -> Dict[str,float]:
    """
    Simple average correctness per chapter (only for chapters present in history).
    """
    df = pd.DataFrame({"question_id": history_qids, "correct": history_correct})
    if df.empty:
        return {}
    merged = df.merge(qdf[['question_id','chapter']].drop_duplicates(), on='question_id', how='left')
    merged['chapter'] = merged['chapter'].fillna('unknown')
    return merged.groupby('chapter')['correct'].mean().to_dict()


def recommend_next_questions(model: keras.Model,
                             qid2idx: Dict[str,int],
                             qdf: pd.DataFrame,
                             history_qids: List[str],
                             history_correct: List[int],
                             k: int = 5,
                             candidates_pool_size: int = 200,
                             target_prob: float = 0.6) -> List[Dict]:
    """
    Return top-k recommended questions (list of dicts with metadata + predicted_prob + score).
    Simple policy: prioritize low-chapter mastery and predicted probability close to target_prob.
    """
    # compute mastery
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

    # score: higher chapter_priority & predicted_prob closeness to target
    for s in scored:
        s['score'] = s['chapter_priority'] - abs(s['predicted_prob'] - target_prob)

    scored_sorted = sorted(scored, key=lambda x: x['score'], reverse=True)
    return scored_sorted[:k]


def sample_coldstart(qdf: pd.DataFrame, k:int=10) -> List[Dict]:
    med_lo, med_hi = 0.35, 0.65
    med = qdf[(qdf['difficulty'] >= med_lo) & (qdf['difficulty'] <= med_hi)]
    if len(med) >= k:
        return med.sample(n=k, random_state=42).to_dict('records')
    else:
        extra = qdf.sample(n=k, random_state=42)
        return extra.to_dict('records')
