"""
Comprehensive test suite for saintGobainSearch backend.
Run: python test_suite.py
"""
import sys, json, time, requests
sys.path.insert(0, '.')
from ragService import detect_language, search_similar, generate_answer_stream

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

errors = []

# ─────────────────────────────────────────────────────────────────
# TEST 1: Language Detection
# ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("TEST 1: Language Detection")
print("=" * 60)

lang_cases = [
    ("services provided in Glassdrive Fatima",   "en"),
    ("what are the opening hours",               "en"),
    ("book an appointment",                      "en"),
    ("show me all locations",                    "en"),
    ("any specific product",                     "en"),
    ("I need glass repair",                      "en"),
    ("how do i contact a centre",                "en"),
    ("Ou se trouve le centre Glassdrive",        "fr"),
    ("Comment prendre rendez-vous",              "fr"),
    ("Quels services proposez-vous",             "fr"),
    ("Je veux un rendez-vous",                   "fr"),
    ("Onde fica o Glassdrive mais proximo",      "pt"),
    ("Como marcar uma consulta",                 "pt"),
    ("Quais os servicos disponiveis",            "pt"),
    ("Preciso de uma marcacao",                  "pt"),
]
lang_passed = 0
for text, expected in lang_cases:
    detected = detect_language(text)
    ok = detected == expected
    if ok:
        lang_passed += 1
    else:
        errors.append(f"Lang detection FAIL: '{text}' -> got '{detected}', expected '{expected}'")
    status = PASS if ok else FAIL
    print(f"  {status} {detected:3s} (expected {expected}) | {text[:50]}")

print(f"  Result: {lang_passed}/{len(lang_cases)} passed\n")

# ─────────────────────────────────────────────────────────────────
# TEST 2: Relevance Gate (search_similar scores)
# ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("TEST 2: Similarity Scores (relevance gate)")
print("=" * 60)

domain_keywords = [
    'glassdrive', 'france pare-brise', 'france pare brise', 'pare-brise', 'parebrise',
    'fpb', 'glass', 'vitr', 'vidr', 'para-bris', 'parabris', 'windshield',
    'windscreen', 'window', 'repar', 'substitu', 'troc', 'consert',
    'chang', 'remplac', 'appoint', 'rendez-vous', 'rendezvous', 'marc', 'agend',
    'insur', 'segur', 'assur', 'adas', 'calibr', 'hour', 'horair', 'ouvert', 'abert',
    'sabad', 'samedi', 'doming', 'dimanch', 'centr', 'servic', 'camping-car', 'autocaravana',
    'motorhome', 'van', 'camion', 'truck', 'vehic', 'véhic', 'veícul',
    'location', 'near', 'close', 'where', 'perto', 'proche', 'adresse', 'address', 'map'
]

search_cases = [
    # (query, should_pass_threshold, description)
    ("Glassdrive opening hours",          True,  "on-topic: hours"),
    ("windshield repair appointment",     True,  "on-topic: repair/appointment"),
    ("glass replacement insurance",       True,  "on-topic: insurance"),
    ("services Glassdrive Fatima",        True,  "on-topic: specific centre"),
    ("prime minister of india",           False, "off-topic: politics"),
    ("what is the capital of france",     False, "off-topic: geography"),
    ("recipe for chocolate cake",         False, "off-topic: food"),
]

for query_text, should_pass, desc in search_cases:
    try:
        results = search_similar(query_text, 5)
        top_sim = results[0]['similarity'] if results else 0.0
        
        has_domain_keyword = any(kw in query_text.lower() for kw in domain_keywords)
        threshold = 0.44 if has_domain_keyword else 0.70
        passes_gate = top_sim >= threshold
        
        ok = passes_gate == should_pass
        if not ok:
            errors.append(f"Relevance FAIL for '{query_text}': sim={top_sim:.3f}, expected pass={should_pass}")
        status = PASS if ok else FAIL
        gate_label = "PASS" if passes_gate else "BLOCK"
        print(f"  {status} [{gate_label}] sim={top_sim:.3f} | {desc}: '{query_text[:40]}'")
    except Exception as e:
        errors.append(f"Search error on '{query}': {e}")
        print(f"  {FAIL} ERROR: {e}")

print()

# ─────────────────────────────────────────────────────────────────
# TEST 3: Stream structure (check event types arrive correctly)
# ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("TEST 3: Stream event structure")
print("=" * 60)

stream_cases = [
    ("what services does Glassdrive offer",  "en", "on-topic EN"),
    ("Quels sont les services Glassdrive",   "fr", "on-topic FR"),
    ("Quais os servicos Glassdrive",         "pt", "on-topic PT"),
    ("prime minister of india",              "en", "off-topic EN"),
]

for query_text, lang, desc in stream_cases:
    try:
        results = search_similar(query_text, 5)
        top_sim = results[0]['similarity'] if results else 0.0
        has_domain_keyword = any(kw in query_text.lower() for kw in domain_keywords)
        threshold = 0.44 if has_domain_keyword else 0.70
        if top_sim < threshold:
            results = []

        events = {'sources': 0, 'token': 0, 'followup': 0, 'done': 0}
        followup_items = []
        answer_text = ""

        for raw in generate_answer_stream(query_text, results, lang=lang):
            raw = raw.strip()
            if not raw:
                continue
            try:
                evt = json.loads(raw)
                t = evt.get('type', '')
                if t in events:
                    events[t] += 1
                if t == 'token':
                    answer_text += evt.get('token', '')
                if t == 'followup':
                    followup_items = evt.get('followup', [])
            except:
                pass

        # Checks
        has_sources  = events['sources'] >= 1
        has_token    = events['token'] >= 1
        has_followup = events['followup'] == 1
        has_done     = events['done'] == 1
        no_marker    = '[FOLLOWUPS]' not in answer_text
        no_template  = 'Follow-up question' not in answer_text and 'Question 1' not in answer_text
        no_gps       = '75.857' not in answer_text and '26.799' not in answer_text
        followups_ok = len(followup_items) >= 2

        all_ok = all([has_sources, has_token, has_followup, has_done, no_marker, no_template, followups_ok])
        status = PASS if all_ok else FAIL
        if not all_ok:
            errors.append(f"Stream FAIL [{desc}]: sources={has_sources} token={has_token} followup={has_followup} done={has_done} no_marker={no_marker} no_template={no_template} followups_ok={followups_ok}")

        print(f"  {status} [{desc}]")
        print(f"         events={events} | followups={len(followup_items)} | answer_len={len(answer_text)}")
        if followup_items:
            print(f"         followups: {followup_items[:2]}")

    except Exception as e:
        errors.append(f"Stream error [{desc}]: {e}")
        print(f"  {FAIL} ERROR: {e}")

print()

# ─────────────────────────────────────────────────────────────────
# TEST 4: No placeholder text leaking
# ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("TEST 4: Placeholder / template text detection")
print("=" * 60)

BAD_PATTERNS = [
    '[Center Name]', '[Address', '[City', '[Country]',
    'Follow-up question 1', 'Follow-up question 2', 'Follow-up question 3',
    '[FOLLOWUPS]',
]

placeholder_query = "services provided in Glassdrive Fatima"
try:
    results = search_similar(placeholder_query, 5)
    top_sim = results[0]['similarity'] if results else 0.0
    has_domain_keyword = any(kw in placeholder_query.lower() for kw in domain_keywords)
    threshold = 0.44 if has_domain_keyword else 0.70
    if top_sim < threshold:
        results = []
    answer_text = ""
    for raw in generate_answer_stream(placeholder_query, results, lang='en'):
        raw = raw.strip()
        if not raw:
            continue
        try:
            evt = json.loads(raw)
            if evt.get('type') == 'token':
                answer_text += evt.get('token', '')
        except:
            pass

    found_bad = [p for p in BAD_PATTERNS if p in answer_text]
    if found_bad:
        errors.append(f"Placeholder leak: {found_bad}")
        print(f"  {FAIL} Bad patterns found: {found_bad}")
    else:
        print(f"  {PASS} No placeholder/template text in response")
    print(f"       Answer preview: {answer_text[:200].strip()!r}")
except Exception as e:
    errors.append(f"Placeholder test error: {e}")
    print(f"  {FAIL} ERROR: {e}")

print()

# ─────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("SUMMARY")
print("=" * 60)
if errors:
    print(f"  {len(errors)} issue(s) found:")
    for e in errors:
        print(f"    - {e}")
else:
    print(f"  All tests passed!")
print()
