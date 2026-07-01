"""
100-Question Comprehensive Test Suite — saintGobainSearch / Glassdrive
Tests:
  Phase 1 — Language Detection       (instant, all 100 queries)
  Phase 2 — Relevance Gate           (embedding similarity, no LLM)
  Phase 3 — Full LLM Stream          (20 representative queries)

Run:  python test_100.py
"""
import sys, json, time, re, textwrap
sys.path.insert(0, '.')
from ragService import detect_language, search_similar, generate_answer_stream

# ─────────────────────────────────────────────────────────────────
# 100-question bank
# Columns: (query, expected_lang, on_topic: bool, description)
# ─────────────────────────────────────────────────────────────────
QUESTIONS = [
    # ── On-topic English (20) ──────────────────────────────────
    ("What services does Glassdrive offer?",                            "en", True,  "EN: services overview"),
    ("What are the opening hours for Glassdrive?",                      "en", True,  "EN: opening hours"),
    ("How do I book an appointment at Glassdrive?",                     "en", True,  "EN: appointment"),
    ("Where is the nearest Glassdrive centre?",                         "en", True,  "EN: location"),
    ("Does Glassdrive handle windshield replacement?",                  "en", True,  "EN: windshield replacement"),
    ("Can Glassdrive repair a cracked windshield?",                     "en", True,  "EN: windshield repair"),
    ("What is ADAS calibration?",                                       "en", True,  "EN: ADAS calibration"),
    ("Does Glassdrive work with my insurance?",                         "en", True,  "EN: insurance"),
    ("How long does windshield replacement take?",                      "en", True,  "EN: repair duration"),
    ("Services provided in Glassdrive Fátima",                         "en", True,  "EN: specific centre"),
    ("What glass products does Glassdrive sell?",                       "en", True,  "EN: products"),
    ("Is there a Glassdrive centre in Lisbon?",                         "en", True,  "EN: city location"),
    ("Can I get a quote for windshield repair?",                        "en", True,  "EN: quote"),
    ("Does Glassdrive replace rear windshields?",                       "en", True,  "EN: rear glass"),
    ("What warranty does Glassdrive offer?",                            "en", True,  "EN: warranty"),
    ("How do I contact a Glassdrive centre?",                           "en", True,  "EN: contact"),
    ("Do you offer glass repair for vans?",                             "en", True,  "EN: van repair"),
    ("Is Saturday opening available at Glassdrive?",                    "en", True,  "EN: Saturday hours"),
    ("Does Glassdrive offer rain sensor recalibration?",                "en", True,  "EN: sensor recal"),
    ("Show all Glassdrive locations available",                         "en", True,  "EN: all locations"),

    # ── On-topic French (20) ───────────────────────────────────
    ("Quels sont les services offerts par Glassdrive ?",                "fr", True,  "FR: services overview"),
    ("Quels sont les horaires d'ouverture de Glassdrive ?",             "fr", True,  "FR: opening hours"),
    ("Comment prendre rendez-vous chez Glassdrive ?",                   "fr", True,  "FR: appointment"),
    ("Où se trouve le centre Glassdrive le plus proche ?",              "fr", True,  "FR: nearest centre"),
    ("Glassdrive fait-il le remplacement de pare-brise ?",              "fr", True,  "FR: windshield replacement"),
    ("Combien de temps dure une réparation de vitre ?",                 "fr", True,  "FR: repair duration"),
    ("Glassdrive gère-t-il les assurances ?",                           "fr", True,  "FR: insurance"),
    ("Y a-t-il un centre Glassdrive à Porto ?",                         "fr", True,  "FR: city location"),
    ("Quels produits Glassdrive propose-t-il ?",                        "fr", True,  "FR: products"),
    ("Glassdrive répare-t-il les fissures de pare-brise ?",             "fr", True,  "FR: crack repair"),
    ("Puis-je avoir un devis pour remplacement de vitre ?",             "fr", True,  "FR: quote"),
    ("Y a-t-il des services pour camping-car ?",                        "fr", True,  "FR: motorhome"),
    ("Glassdrive est-il ouvert le samedi ?",                            "fr", True,  "FR: Saturday"),
    ("Quelle garantie Glassdrive offre-t-il ?",                         "fr", True,  "FR: warranty"),
    ("Comment contacter le centre Glassdrive de Lisbonne ?",            "fr", True,  "FR: contact"),
    ("Glassdrive propose-t-il la calibration ADAS ?",                   "fr", True,  "FR: ADAS"),
    ("Est-ce que Glassdrive est couvert par mon assurance ?",           "fr", True,  "FR: insurance coverage"),
    ("Quels sont les tarifs de Glassdrive ?",                           "fr", True,  "FR: pricing"),
    ("Glassdrive remplace-t-il les vitres latérales ?",                 "fr", True,  "FR: side glass"),
    ("Où sont les centres Glassdrive au Portugal ?",                    "fr", True,  "FR: Portugal centres"),

    # ── On-topic Portuguese (20) ───────────────────────────────
    ("Quais são os serviços da Glassdrive?",                            "pt", True,  "PT: services overview"),
    ("Quais são os horários da Glassdrive?",                            "pt", True,  "PT: opening hours"),
    ("Como posso marcar uma consulta na Glassdrive?",                   "pt", True,  "PT: appointment"),
    ("Onde fica o centro Glassdrive mais próximo?",                     "pt", True,  "PT: nearest centre"),
    ("A Glassdrive faz substituição de para-brisas?",                   "pt", True,  "PT: windshield replacement"),
    ("Quanto tempo demora uma reparação de vidro?",                     "pt", True,  "PT: repair duration"),
    ("A Glassdrive trata do processo de seguro?",                       "pt", True,  "PT: insurance"),
    ("Existe um centro Glassdrive em Lisboa?",                          "pt", True,  "PT: city location"),
    ("Que produtos a Glassdrive vende?",                                "pt", True,  "PT: products"),
    ("A Glassdrive repara fissuras no para-brisas?",                    "pt", True,  "PT: crack repair"),
    ("Posso obter um orçamento para substituição de vidro?",            "pt", True,  "PT: quote"),
    ("A Glassdrive tem serviço para autocaravanas?",                    "pt", True,  "PT: motorhome"),
    ("A Glassdrive está aberta ao sábado?",                             "pt", True,  "PT: Saturday"),
    ("Que garantia oferece a Glassdrive?",                              "pt", True,  "PT: warranty"),
    ("Como contacto o centro Glassdrive Fátima?",                       "pt", True,  "PT: contact specific"),
    ("A Glassdrive faz calibração ADAS?",                               "pt", True,  "PT: ADAS"),
    ("O meu seguro cobre reparação na Glassdrive?",                     "pt", True,  "PT: insurance coverage"),
    ("Quais são os preços da Glassdrive?",                              "pt", True,  "PT: pricing"),
    ("Onde ficam os centros Glassdrive em Portugal?",                   "pt", True,  "PT: all locations"),
    ("Serviços disponíveis na Glassdrive Santarém",                     "pt", True,  "PT: specific centre"),

    # ── Off-topic English (15) ─────────────────────────────────
    ("Who is the prime minister of India?",                             "en", False, "OFF-EN: politics"),
    ("What is the capital of France?",                                  "en", False, "OFF-EN: geography"),
    ("Give me a recipe for chocolate cake",                             "en", False, "OFF-EN: food"),
    ("What is the latest news?",                                        "en", False, "OFF-EN: news"),
    ("How do I fix my iPhone?",                                         "en", False, "OFF-EN: tech"),
    ("Who won the World Cup?",                                          "en", False, "OFF-EN: sports"),
    ("What is 2+2?",                                                    "en", False, "OFF-EN: math"),
    ("Tell me a joke",                                                  "en", False, "OFF-EN: entertainment"),
    ("What is the weather today?",                                      "en", False, "OFF-EN: weather"),
    ("Book me a flight to Paris",                                       "en", False, "OFF-EN: travel"),
    ("What stocks should I buy?",                                       "en", False, "OFF-EN: finance"),
    ("Write me a poem about love",                                      "en", False, "OFF-EN: creative writing"),
    ("How do I learn Python programming?",                              "en", False, "OFF-EN: education"),
    ("What movies are showing today?",                                  "en", False, "OFF-EN: entertainment"),
    ("Can you translate this to Spanish?",                              "en", False, "OFF-EN: translation"),

    # ── Off-topic French (8) ──────────────────────────────────
    ("Qui est le président de la France ?",                             "fr", False, "OFF-FR: politics"),
    ("Quelle est la recette du boeuf bourguignon ?",                    "fr", False, "OFF-FR: food"),
    ("Quel temps fait-il à Paris ?",                                    "fr", False, "OFF-FR: weather"),
    ("Qu'est-ce que l'intelligence artificielle ?",                     "fr", False, "OFF-FR: tech"),
    ("Quelles sont les dernières nouvelles ?",                          "fr", False, "OFF-FR: news"),
    ("Recommande-moi un bon restaurant à Lyon",                         "fr", False, "OFF-FR: restaurant"),
    ("Qui a gagné le Tour de France ?",                                 "fr", False, "OFF-FR: sports"),
    ("Comment apprendre l'espagnol rapidement ?",                       "fr", False, "OFF-FR: education"),

    # ── Off-topic Portuguese (7) ───────────────────────────────
    ("Quem é o presidente de Portugal?",                                "pt", False, "OFF-PT: politics"),
    ("Qual é a receita do bacalhau à brás?",                            "pt", False, "OFF-PT: food"),
    ("Como está o tempo em Lisboa?",                                    "pt", False, "OFF-PT: weather"),
    ("Quem ganhou o Campeonato do Mundo?",                              "pt", False, "OFF-PT: sports"),
    ("Como aprender inglês rapidamente?",                               "pt", False, "OFF-PT: education"),
    ("Recomenda um bom restaurante no Porto?",                          "pt", False, "OFF-PT: restaurant"),
    ("Qual é o melhor smartphone de 2024?",                             "pt", False, "OFF-PT: tech"),

    # ── Edge cases (10) ───────────────────────────────────────
    ("hello",                                                           "en", False, "EDGE: single word greeting"),
    ("Glassdrive",                                                      "en", True,  "EDGE: brand name only"),
    ("?",                                                               "en", False, "EDGE: punctuation only"),
    ("GLASSDRIVE FATIMA SERVICES HOURS ADDRESS",                        "en", True,  "EDGE: all caps"),
    ("glassdrive fatima opening hours address repair",                   "en", True,  "EDGE: all lowercase"),
    ("horaires glassdrive fatima",                                       "fr", True,  "EDGE: FR short phrase"),
    ("glassdrive para-brisas",                                          "pt", True,  "EDGE: PT short phrase"),
    ("repair    my   glass",                                            "en", True,  "EDGE: extra whitespace"),
    ("Can Glassdrive fix my car AND the weather?",                      "en", True,  "EDGE: mixed topic (Glassdrive mentioned)"),
    ("I want to know EVERYTHING about Glassdrive services please",       "en", True,  "EDGE: verbose request"),
]

assert len(QUESTIONS) == 100, f"Expected 100 questions, got {len(QUESTIONS)}"

# ─────────────────────────────────────────────────────────────────
# Select 20 representative queries for full LLM stream test
# ─────────────────────────────────────────────────────────────────
STREAM_TEST_INDICES = [
    0,   # EN: services overview
    1,   # EN: opening hours
    8,   # EN: repair duration
    9,   # EN: specific centre (Fátima — the previous problem case)
    16,  # EN: van repair
    20,  # FR: services overview
    22,  # FR: appointment
    26,  # FR: insurance
    35,  # FR: ADAS
    40,  # PT: services overview
    44,  # PT: windshield replacement
    46,  # PT: insurance
    59,  # PT: specific centre
    60,  # OFF-EN: politics
    62,  # OFF-EN: food
    69,  # OFF-EN: travel
    75,  # OFF-FR: politics
    82,  # OFF-PT: politics
    90,  # EDGE: single word greeting
    99,  # EDGE: verbose request
]

THRESHOLD = 0.50
PASS = "PASS"
FAIL = "FAIL"

results = {
    "lang_pass": 0, "lang_fail": 0, "lang_fail_list": [],
    "gate_pass": 0, "gate_fail": 0, "gate_fail_list": [],
    "stream_pass": 0, "stream_fail": 0, "stream_fail_list": [],
}

def section(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)

# ─────────────────────────────────────────────────────────────────
# PHASE 1: Language Detection (all 100, instant)
# ─────────────────────────────────────────────────────────────────
section("PHASE 1: Language Detection — 100 queries")

for i, (q, exp_lang, on_topic, desc) in enumerate(QUESTIONS):
    detected = detect_language(q)
    ok = detected == exp_lang
    if ok:
        results["lang_pass"] += 1
    else:
        results["lang_fail"] += 1
        results["lang_fail_list"].append(f"  Q{i+1:03d} got={detected} exp={exp_lang} | {desc}: {q[:50]}")

    mark = PASS if ok else FAIL
    print(f"  [{mark}] Q{i+1:03d} {detected:3s}|{exp_lang:3s}  {desc}")

print(f"\n  >> Lang detection: {results['lang_pass']}/100 passed, {results['lang_fail']} failed")

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

# ─────────────────────────────────────────────────────────────────
# PHASE 2: Relevance Gate (all 100, embedding similarity only)
# ─────────────────────────────────────────────────────────────────
section("PHASE 2: Relevance Gate — 100 queries")
print("  (checking embedding similarity — no LLM calls)")
print()

start_gate = time.time()
for i, (q, exp_lang, on_topic, desc) in enumerate(QUESTIONS):
    try:
        results_sim = search_similar(q, 5)
        top_sim = results_sim[0]['similarity'] if results_sim else 0.0
        
        query_lower = q.lower()
        has_domain_keyword = any(kw in query_lower for kw in domain_keywords)
        threshold = 0.44 if has_domain_keyword else 0.70
        passes = top_sim >= threshold
        
        correct = passes == on_topic
        if correct:
            results["gate_pass"] += 1
        else:
            results["gate_fail"] += 1
            results["gate_fail_list"].append(
                f"  Q{i+1:03d} sim={top_sim:.3f} passes={passes} expected_ontopic={on_topic} | {desc}"
            )
        mark = PASS if correct else FAIL
        gate_label = "PASS gate" if passes else "BLOCKED  "
        print(f"  [{mark}] Q{i+1:03d} sim={top_sim:.3f} [{gate_label}] {desc}: {q[:45]}")
    except Exception as e:
        results["gate_fail"] += 1
        results["gate_fail_list"].append(f"  Q{i+1:03d} ERROR: {e}")
        print(f"  [FAIL] Q{i+1:03d} ERROR: {e}")

gate_time = time.time() - start_gate
print(f"\n  >> Relevance gate: {results['gate_pass']}/100 passed, {results['gate_fail']} failed  ({gate_time:.1f}s)")

# ─────────────────────────────────────────────────────────────────
# PHASE 3: Full LLM Stream (20 representative queries)
# ─────────────────────────────────────────────────────────────────
section("PHASE 3: Full LLM Stream — 20 representative queries")
print("  (language + structure + follow-up quality + no placeholders)")
print()

BAD_PATTERNS = [
    '[Center Name]', '[Address', '[City', '[Country]',
    'Follow-up question 1', 'Follow-up question 2', 'Follow-up question 3',
    '[FOLLOWUPS]', '<write a', '<another',
]

stream_start = time.time()
for idx in STREAM_TEST_INDICES:
    q, exp_lang, on_topic, desc = QUESTIONS[idx]
    q_num = idx + 1
    try:
        sim_results = search_similar(q, 5)
        top_sim = sim_results[0]['similarity'] if sim_results else 0.0
        
        query_lower = q.lower()
        has_domain_keyword = any(kw in query_lower for kw in domain_keywords)
        threshold = 0.44 if has_domain_keyword else 0.70
        chunks = sim_results if top_sim >= threshold else []

        events = {"sources": 0, "token": 0, "followup": 0, "done": 0}
        answer = ""
        followups = []

        for raw in generate_answer_stream(q, chunks, lang=exp_lang):
            raw = raw.strip()
            if not raw:
                continue
            try:
                evt = json.loads(raw)
                t = evt.get("type", "")
                if t in events:
                    events[t] += 1
                if t == "token":
                    answer += evt.get("token", "")
                if t == "followup":
                    followups = evt.get("followup", [])
            except:
                pass

        # Quality checks
        has_structure  = events["sources"] >= 1 and events["token"] >= 1 and events["done"] == 1
        has_followup   = len(followups) >= 2
        no_marker      = "[FOLLOWUPS]" not in answer
        no_placeholder = not any(p in answer for p in BAD_PATTERNS)
        no_gps_leak    = "75.857" not in answer and "26.799" not in answer

        all_ok = all([has_structure, has_followup, no_marker, no_placeholder, no_gps_leak])

        mark = PASS if all_ok else FAIL
        gate = "ON " if chunks else "OFF"
        print(f"  [{mark}] Q{q_num:03d} [{exp_lang}/{gate}] {desc}")
        if not all_ok:
            issues = []
            if not has_structure:  issues.append(f"events={events}")
            if not has_followup:   issues.append(f"followups={len(followups)}")
            if not no_marker:      issues.append("FOLLOWUPS_MARKER_LEAKED")
            if not no_placeholder: issues.append("PLACEHOLDER_LEAKED")
            if not no_gps_leak:    issues.append("GPS_COORDS_LEAKED")
            print(f"         Issues: {', '.join(issues)}")
            results["stream_fail"] += 1
            results["stream_fail_list"].append(f"  Q{q_num:03d} [{desc}]: {', '.join(issues)}")
        else:
            results["stream_pass"] += 1

        # Print first follow-up for spot-check
        if followups:
            print(f"         followup[0]: {followups[0][:80]}")
        print(f"         answer_len={len(answer)} | sim={top_sim:.3f}")

    except Exception as e:
        results["stream_fail"] += 1
        results["stream_fail_list"].append(f"  Q{q_num:03d} [{desc}]: ERROR {e}")
        print(f"  [FAIL] Q{q_num:03d} ERROR: {e}")

stream_time = time.time() - stream_start
print(f"\n  >> Stream tests: {results['stream_pass']}/20 passed, {results['stream_fail']} failed  ({stream_time:.1f}s)")

# ─────────────────────────────────────────────────────────────────
# FINAL REPORT
# ─────────────────────────────────────────────────────────────────
section("FINAL REPORT")
total_pass = results["lang_pass"] + results["gate_pass"] + results["stream_pass"]
total_fail = results["lang_fail"] + results["gate_fail"] + results["stream_fail"]
total = 220  # 100 + 100 + 20

print(f"  Language Detection : {results['lang_pass']:3d}/100 passed")
print(f"  Relevance Gate     : {results['gate_pass']:3d}/100 passed")
print(f"  LLM Stream Quality : {results['stream_pass']:3d}/20  passed")
print(f"  ---------------------------------")
print(f"  TOTAL              : {total_pass}/{total} passed  ({100*total_pass/total:.1f}%)")

if total_fail == 0:
    print("\n  *** ALL TESTS PASSED -- System is working correctly! ***")
else:
    print(f"\n  WARNING: {total_fail} test(s) failed:")
    for lst_key in ("lang_fail_list", "gate_fail_list", "stream_fail_list"):
        for item in results[lst_key]:
            print(item)
print()
