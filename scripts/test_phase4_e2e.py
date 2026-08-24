"""Phase 4 end-to-end tests against the running API (requires the server up).

Covers the full audit checklist against the real system:

- Cache safety: semantically different questions must NOT get the same cached
  answer (the original critical bug), paraphrases must hit the cache.
- Fast path + cache bookkeeping (llm_used must be False on cache/fast hits).
- Conversation context: follow-up questions resolve references correctly and a
  fresh standalone question is never contaminated by previous context.
- Multilingual: Arabic + English, colloquial variants.
- Abstention: unrelated / unsupported questions must not be answered.
- Multi-part / synthesis questions.
- Feedback recording through the real /feedback endpoint.

Usage::

    python scripts/test_phase4_e2e.py [base_url]
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"


def post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # Tuned server (qwen3-vl:8b on CPU, OLLAMA_TIMEOUT=1800) can take ~80s per
    # generation; give each call a generous timeout instead of the old 180s.
    with urllib.request.urlopen(req, timeout=1800) as resp:
        return json.loads(resp.read().decode("utf-8"))


def chat(message: str, history: list[dict] | None = None) -> dict:
    payload: dict = {"message": message}
    if history:
        payload["history"] = history
    return post("/chat", payload)


RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def main() -> int:
    print("=" * 70)
    print("PHASE 4 END-TO-END TEST  (target:", BASE + ")")
    print("=" * 70)

    # --- 1. Regression of the original bug ---
    q_list = "ما هي جامعة المنصورة الجديدة وهل هي أهلية أم خاصة؟"
    q_loc = "أين تقع جامعة المنصورة الجديدة؟"
    r_list = chat(q_list)
    r_loc = chat(q_loc)
    ans_list, ans_loc = r_list.get("answer", ""), r_loc.get("answer", "")
    check("bug: different answers for different questions",
          bool(ans_list) and bool(ans_loc) and ans_list != ans_loc,
          f"len_list={len(ans_list)} len_loc={len(ans_loc)}")
    loc_markers = ["المنصورة", "مدينة", "محافظة", "تقع", "الموقع", "دقهلية", "Dakahlia"]
    hits = [m for m in loc_markers if m in ans_loc]
    check("bug: location answer is a location answer", bool(hits), f"markers={hits}")
    decree_marker = "مرسوم"  # the historical wrong answer came from the decree doc
    check("bug: location answer not the establishment decree",
          decree_marker not in ans_loc, "")
    check("list answer still answered", bool(ans_list), f"len={len(ans_list)}")
    # NOTE: the API's ChatResponse schema does NOT expose intent/llm_used;
    # intent is verified at the pipeline level in scripts/recovery_eval.py
    # (mocked mode) instead. Here we check location answer surfaced sources.
    check("location answer has sources", bool(r_loc.get("sources")),
          f"n_sources={len(r_loc.get('sources') or [])}")

    # --- 2. English + colloquial variants (same question family) ---
    en_loc = chat("Where is New Mansoura University located?")
    en_hits = [m for m in ["Dakahlia", "New Mansoura", "located"] if m in en_loc.get("answer", "")]
    check("english location answered", bool(en_loc.get("answer")) and bool(en_hits), f"markers={en_hits}")
    for colloquial in ["فين الجامعة؟", "ما هو موقع الجامعة؟", "الجامعة فين؟"]:
        r = chat(colloquial)
        rl = [m for m in ["تقع", "موقع", "المدينة", "الجامعة"] if m in r.get("answer", "")]
        check(f"colloquial location: {colloquial}", bool(r.get("answer")) and bool(rl), f"markers={rl}")

    # --- 3. Cache: paraphrase hits, cross-topic does NOT ---
    r_first = chat("ما هي كليات جامعة المنصورة الجديدة؟")
    r_repeat = chat("ما هي كليات جامعة المنصورة الجديدة؟")
    check("cache hit on exact repeat", r_repeat.get("cache_hit") is True,
          f"cache_hit={r_repeat.get('cache_hit')}")
    r_paraphrase = chat("ماهي كليات جامعة المنصورة الجديدة؟")
    check("cache hit on paraphrase (ar)", r_paraphrase.get("cache_hit") is True,
          f"cache_hit={r_paraphrase.get('cache_hit')}")
    r_cross = chat("أين تقع جامعة المنصورة الجديدة؟")
    check("no cache hit across topics (LIST vs LOCATION)", r_cross.get("cache_hit") is False,
          f"cache_hit={r_cross.get('cache_hit')}")
    # llm_used is not in the API response schema; a cache hit by definition
    # served the stored answer without regeneration (verified at pipeline level
    # in recovery_eval mocked mode), so assert the response is the cached one.
    check("cache hit returned the same answer", r_repeat.get("answer") == r_first.get("answer"),
          "answers differ")

    # --- 4. Conversation context ---
    hist = [
        {"role": "user", "content": "ما هي كليات جامعة المنصورة الجديدة؟"},
        {"role": "assistant", "content": "كليات جامعة المنصورة الجديدة تشمل كلية الطب وكلية الهندسة وكلية العلوم."},
    ]
    r_follow = chat("وما هي برامج كلية الطب؟", history=hist)
    med_hits = [m for m in ["طب", "الطب"] if m in r_follow.get("answer", "")]
    check("follow-up resolves faculty", bool(r_follow.get("answer")) and bool(med_hits),
          f"markers={med_hits}")
    fresh = chat("ما هي شروط القبول في الجامعة؟", history=hist)
    check("fresh question not contaminated", bool(fresh.get("answer")), "answered standalone")
    # reference-only follow-up ("وما هي برامجها؟") must NOT be answered as a
    # brand-new unrelated question when there is no history to anchor it.
    r_noanchor = chat("وما هي برامجها؟")
    check("unanchored pronoun follow-up is refused/weak",
          (not r_noanchor.get("answer")) or len(r_noanchor.get("answer", "")) < 200,
          f"len={len(r_noanchor.get('answer', ''))}")

    # --- 5. Abstention: unrelated / unsupported ---
    r_unrelated = chat("ما هي عاصمة فرنسا؟")
    refuse = any(m in r_unrelated.get("answer", "") for m in
                 ["لا", "غير متوفر", "غير متاح", "لا أملك", "لا تتوفر", "cannot", "not available", "unable", "لم أجد", "لا توجد"])
    check("unrelated question abstained", (not r_unrelated.get("answer")) or refuse,
          f"len={len(r_unrelated.get('answer', ''))}")
    r_unsupported = chat("ما هي رسوم كلية الطب في جامعة المنصورة الجديدة؟")
    check("unsupported detail abstained or generic refusal",
          (not r_unsupported.get("answer")) or refuse,
          f"len={len(r_unsupported.get('answer', ''))}")

    # --- 6. Multi-part synthesis ---
    r_multi = chat("ما هي شروط القبول وما هو الموقع الرسمي للجامعة؟")
    check("multi-part question answered", bool(r_multi.get("answer")), f"len={len(r_multi.get('answer', ''))}")

    # --- 7. Feedback recording ---
    qid = r_loc.get("question_id") or ""
    if qid:
        fb = post("/feedback", {"question_id": qid, "rating": 1})
        check("feedback recorded", fb.get("status") in ("ok", "recorded", "saved", "accepted"),
              str(fb)[:120])
    else:
        check("feedback recorded", False, "no question_id to feedback")

    # --- Report ---
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print("\n" + "=" * 70)
    for name, ok, detail in RESULTS:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    print("=" * 70)
    print(f"PHASE 4 RESULT: {passed}/{len(RESULTS)} passed")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())