import os
import json
import re
import math
import requests
from dotenv import load_dotenv
load_dotenv(override=True)
from db import query

# Ollama settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://192.168.10.148:11434")
OLLAMA_GEN_MODEL = os.getenv("OLLAMA_GEN_MODEL", "qwen2.5:14b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

print(f"OLLAMA RAG SERVICE: Base URL: {OLLAMA_BASE_URL}, Gen: {OLLAMA_GEN_MODEL}, Embed: {OLLAMA_EMBED_MODEL}")


# ──────────────────────────────────────────────────────────────
#  Data loading & processing
# ──────────────────────────────────────────────────────────────

def load_data(directory_path):
    """Load and parse JSON files"""
    files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
    all_data = []

    for file in files:
        with open(os.path.join(directory_path, file), 'r', encoding='utf-8') as f:
            data = json.load(f)
            enriched_data = []
            if isinstance(data, list):
                for item in data:
                    item['_source'] = file
                    enriched_data.append(item)
            else:
                data['_source'] = file
                enriched_data.append(data)
            all_data.extend(enriched_data)

    return all_data


def process_data(items):
    """Process items into text with metadata"""
    processed_items = []

    for item in items:
        text_parts = []

        def walk_obj(obj):
            if isinstance(obj, dict):
                for key, val in obj.items():
                    if key.startswith('_') or key in [
                        'metadata', 'id', 'created_at', 'search_vector', 'slug',
                        'thumbnails', 'url', 'image', 'height', 'width', 'locale',
                        'primary_key', 'image_url', 'favicon'
                    ]:
                        continue
                    if key in ['hours', 'monday', 'tuesday', 'wednesday', 'thursday',
                               'friday', 'saturday', 'sunday']:
                        text_parts.append(f"{key.capitalize()}:")
                    if isinstance(val, (dict, list)):
                        walk_obj(val)
                    elif val and not isinstance(val, bool):
                        str_val = str(val).strip()
                        if (str_val.startswith(('http', '//')) or
                                str_val.endswith(('.png', '.jpg', '.jpeg', '.webp', '.pdf'))):
                            continue
                        if key in ['start', 'end', 'date'] or ':' in str_val:
                            text_parts.append(f"{key} {str_val}")
                        elif not (re.match(r'^[0-9\s\-\.\/]+$', str_val) or len(str_val) < 3):
                            text_parts.append(str_val)
            elif isinstance(obj, list):
                for item_in_list in obj:
                    walk_obj(item_in_list)

        walk_obj(item)

        title = item.get('metadata', {}).get('title') or item.get('title')
        if not title and 'data' in item:
            title = item['data'].get('name') or item['data'].get('title')

        processed_items.append({
            'text': '\n'.join(text_parts),
            'metadata': {
                'source': item.get('_source', 'unknown'),
                'original_title': title or 'Untitled'
            }
        })

    return processed_items


def chunk_text(text, min_words=200, max_words=500):
    """Split text into chunks"""
    sentences = re.findall(r'[^.!?]+[.!?]+', text) or [text]
    chunks = []
    current_chunk = ""
    current_word_count = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())
        if current_word_count + sentence_words > max_words and current_word_count >= min_words:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_word_count = sentence_words
        else:
            current_chunk += " " + sentence
            current_word_count += sentence_words

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# ──────────────────────────────────────────────────────────────
#  Embedding & vector search
# ──────────────────────────────────────────────────────────────

def get_embedding(text):
    """Get text embedding from local Ollama"""
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": OLLAMA_EMBED_MODEL, "input": text},
            timeout=30
        )
        response.raise_for_status()
        return response.json()["embeddings"][0]
    except Exception as err:
        print(f"Error getting embedding: {err}")
        return [0.0] * 768


def store_in_db(chunk, metadata, embedding):
    """Store chunk and its embedding in database"""
    sql = 'INSERT INTO document_chunks (content, metadata, embedding) VALUES (%s, %s, %s)'
    query(sql, [chunk, json.dumps(metadata), embedding])


def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors"""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot_prod = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(a * a for a in v2))
    if mag1 == 0.0 or mag2 == 0.0:
        return 0.0
    return dot_prod / (mag1 * mag2)


def search_similar(query_text, limit=5):
    """Search for similar chunks using cosine similarity over stored embeddings"""
    try:
        query_emb = get_embedding(query_text)

        rows = query('SELECT content, metadata, embedding FROM document_chunks')
        if not rows:
            print("No document chunks found in database.")
            return []

        results = []
        for row in rows:
            chunk_emb = row['embedding']
            if not chunk_emb:
                continue
            similarity = cosine_similarity(query_emb, chunk_emb)
            results.append({
                'content': row['content'],
                'metadata': row['metadata'],
                'similarity': similarity
            })

        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:limit]

    except Exception as err:
        print(f'DATABASE ERROR in search_similar: {err}')
        raise err


# ──────────────────────────────────────────────────────────────
#  Language detection
# ──────────────────────────────────────────────────────────────

# Common words per language (ordered by discriminative power)
_LANG_WORDS = {
    'en': [
        'what', 'where', 'when', 'how', 'the', 'is', 'are', 'can', 'do',
        'i', 'any', 'specific', 'product', 'my', 'your', 'for', 'with',
        'have', 'has', 'want', 'need', 'find', 'get', 'tell', 'about',
        'please', 'help', 'does', 'will', 'show', 'know', 'which',
    ],
    'fr': [
        'où', 'comment', 'quel', 'quelle', 'quels', 'quelles', 'je', 'vous',
        'le', 'la', 'les', 'est', 'sont', 'puis-je', 'que', 'quoi',
        'pourquoi', 'puis', 'mon', 'ma', 'mes', 'du', 'de', 'des',
        'une', 'un', 'avec', 'pour', 'dans', 'sur', 'avoir', 'être',
    ],
    'pt': [
        'onde', 'como', 'qual', 'quais', 'eu', 'você', 'voce', 'ele',
        'ela', 'os', 'as', 'está', 'são', 'posso', 'que', 'meu', 'minha',
        'por', 'para', 'com', 'tem', 'ter', 'ser', 'uma', 'num', 'numa',
        'quero', 'preciso', 'gostaria', 'existe', 'há',
    ],
}

def detect_language(text: str) -> str:
    """Detect query language using common word frequency. Returns 'en', 'fr', or 'pt'."""
    words = set(re.findall(r"[a-záàâãäéèêëíìîïóòôõöúùûüçñ']+", text.lower()))
    scores = {lang: sum(1 for w in kws if w in words) for lang, kws in _LANG_WORDS.items()}
    best = max(scores, key=scores.get)
    # Default to 'en' when scores are tied or all zero
    return best if scores[best] > 0 else 'en'


# ──────────────────────────────────────────────────────────────
#  Streaming answer generation
# ──────────────────────────────────────────────────────────────

def generate_answer_stream(query_text, context_chunks, location=None, lang=None):
    """Generate streaming answer and follow-up questions from Ollama."""
    # Detect language from the query itself, not from context documents
    if lang is None:
        lang = detect_language(query_text)

    LANG_NAMES = {'en': 'English', 'fr': 'French', 'pt': 'Portuguese'}
    lang_name = LANG_NAMES.get(lang, 'English')

    # Language-specific off-topic / no-results messages
    OFF_TOPIC = {
        'en': (
            "Sorry, I can only answer questions about Glassdrive services "
            "(centres, opening hours, glass repair/replacement, insurance, etc.).\n\n"
            "Your question seems outside our service scope. "
            "Feel free to ask me anything about Glassdrive!"
        ),
        'fr': (
            "Je suis désolé, je ne peux répondre qu'aux questions relatives aux services "
            "Glassdrive (centres, horaires, réparation/remplacement de vitres, assurances, etc.).\n\n"
            "Votre question semble hors du périmètre de nos services. "
            "N'hésitez pas à me poser une question sur Glassdrive !"
        ),
        'pt': (
            "Lamentamos, só podemos responder a questões sobre os serviços "
            "Glassdrive (centros, horários, reparação/substituição de vidros, seguros, etc.).\n\n"
            "A sua pergunta parece estar fora do âmbito dos nossos serviços. "
            "Não hesite em fazer-nos uma pergunta sobre a Glassdrive!"
        ),
    }
    OFF_TOPIC_FOLLOWUPS = {
        'en': ["Where is the nearest Glassdrive centre?", "What services do you offer?", "How can I book an appointment?"],
        'fr': ["Où se trouve le centre Glassdrive le plus proche ?", "Quels services proposez-vous ?", "Comment prendre rendez-vous ?"],
        'pt': ["Onde fica o centro Glassdrive mais próximo?", "Que serviços oferecem?", "Como posso marcar uma consulta?"],
    }
    if not context_chunks:
        yield json.dumps({'type': 'sources', 'sources': []}) + '\n'
        yield json.dumps({
            'type': 'token',
            'token': OFF_TOPIC.get(lang, OFF_TOPIC['en'])
        }) + '\n'
        yield json.dumps({
            'type': 'followup',
            'followup': OFF_TOPIC_FOLLOWUPS.get(lang, OFF_TOPIC_FOLLOWUPS['en'])
        }) + '\n'
        yield json.dumps({'type': 'done'}) + '\n'
        return

    yield json.dumps({'type': 'sources', 'sources': context_chunks}) + '\n'

    context = '\n\n---\n\n'.join([c['content'] for c in context_chunks])

    system_prompt = f"""You are a Glassdrive customer service assistant. Your ONLY role is to answer questions about Glassdrive services.

⚠️ LANGUAGE RULE — MANDATORY: Respond ENTIRELY in {lang_name}. Do NOT use any other language.

Glassdrive topics you can answer:
- Glassdrive center locations, addresses, and opening hours
- Windshield / automotive glass repair and replacement
- ADAS calibration and driver assistance systems
- Insurance procedures for glass repair
- Appointments and service bookings
- Glassdrive products, warranties, and partnerships

CRITICAL RULES:
1. Language: Respond ONLY in {lang_name}.
2. Be concise and direct. List relevant centers/information clearly. Do not write long introductions.
3. NEVER include parenthetical notes, reasoning, or internal thoughts. Start directly with the answer.
4. LOCATION PRIVACY (very important):
   - If GPS coordinates appear in the context, use them ONLY to rank centers by proximity.
   - NEVER reveal, mention, or quote the GPS coordinates in your response.
   - NEVER tell the user where they currently are (city, country, or region).
   - NEVER comment on whether the user is in Portugal or not.
   - Simply list the relevant Glassdrive centers sorted by proximity without any geographic commentary.
5. If the question is NOT about Glassdrive, reply ONLY:
   "{OFF_TOPIC.get(lang, OFF_TOPIC['en']).split(chr(10))[0]}"
6. Answer ONLY using the provided context. Do not add external information.
7. At the end of every on-topic answer, write exactly:
[FOLLOWUPS]
- Follow-up question 1 in {lang_name} ?
- Follow-up question 2 in {lang_name} ?
- Follow-up question 3 in {lang_name} ?

Do not write anything after the follow-up questions."""

    # Inject user location as an INTERNAL system note
    # The model is told to use coordinates for proximity ranking ONLY — never output them
    location_context = ""
    if location and isinstance(location, dict):
        lat = location.get('lat')
        lng = location.get('lng')
        if lat is not None and lng is not None:
            location_context = (
                f"\n\n[INTERNAL — DO NOT REVEAL TO USER: "
                f"User GPS = {lat:.5f}, {lng:.5f}. "
                f"Use ONLY to silently sort/prioritise the nearest Glassdrive centers. "
                f"Do NOT mention coordinates, do NOT say where the user is, "
                f"do NOT comment on their geographic location. "
                f"Just list relevant centers.]"
            )

    user_content = f"""Context:
{context}{location_context}

User question: {query_text}"""

    try:
        print(f"RAG SERVICE: Streaming answer from Ollama ({OLLAMA_GEN_MODEL}) for: {query_text}")
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_GEN_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                "stream": True,
                "options": {"temperature": 0.7}
            },
            stream=True,
            timeout=120
        )
        response.raise_for_status()

        full_text = ""
        in_followup = False
        followup_buffer = ""
        MARKER = "[FOLLOWUPS]"
        pending = ""

        # ── Thinking-text filter state ─────────────────────────────────────
        # Models sometimes output parenthetical reasoning like:
        #   (Since the user asked in French, I'll respond accordingly...)
        # We track open/close parens and silently discard any (...) block
        # that contains known reasoning keywords.
        REASONING_KW = [
            'since', 'note that', "i'll", 'i will', 'typically', 'as per',
            'correct translation', 'the user asked', 'respond accordingly',
            'given that', 'let me', 'internal', 'context as per',
            'should stick', 'i should', 'provided doc'
        ]
        paren_depth = 0
        paren_buf = ""       # Accumulates text inside suspected thinking block
        PAREN_SIZE_LIMIT = 600  # Safety: if a paren block is huge, keep it
        # ──────────────────────────────────────────────────────────────────

        for line in response.iter_lines():
            if not line:
                continue
            chunk_data = json.loads(line.decode('utf-8'))
            token = chunk_data.get("message", {}).get("content", "")
            if not token:
                continue

            if in_followup:
                followup_buffer += token
                continue

            # ── Per-character processing to strip thinking parentheticals ──
            clean_token = ""
            for char in token:
                if paren_depth == 0:
                    if char == '(':
                        paren_depth = 1
                        paren_buf = "("
                    else:
                        clean_token += char
                else:
                    if char == '(':
                        paren_depth += 1
                        paren_buf += char
                    elif char == ')':
                        paren_depth -= 1
                        paren_buf += char
                        if paren_depth == 0:
                            # Decide: reasoning or legitimate parenthetical?
                            is_reasoning = (
                                len(paren_buf) > 30 and
                                any(k in paren_buf.lower() for k in REASONING_KW)
                            )
                            if not is_reasoning:
                                clean_token += paren_buf  # Keep it
                            # else: silently discard
                            paren_buf = ""
                    else:
                        paren_buf += char
                        # Safety valve: very long paren block → keep as content
                        if len(paren_buf) > PAREN_SIZE_LIMIT:
                            clean_token += paren_buf
                            paren_buf = ""
                            paren_depth = 0

            if not clean_token:
                continue
            # ── End thinking filter ───────────────────────────────────────

            pending += clean_token
            full_text += clean_token

            # Check if FOLLOWUPS marker has fully arrived
            if MARKER in pending:
                in_followup = True
                before, after = pending.split(MARKER, 1)
                if before.strip():
                    yield json.dumps({'type': 'token', 'token': before}) + '\n'
                followup_buffer = after
                pending = ""
                continue

            # Flush safe portion (keep last len(MARKER)-1 chars in buffer)
            safe_len = max(0, len(pending) - len(MARKER) + 1)
            if safe_len > 0:
                yield json.dumps({'type': 'token', 'token': pending[:safe_len]}) + '\n'
                pending = pending[safe_len:]

        # Flush remaining pending (no marker arrived)
        if pending and not in_followup:
            yield json.dumps({'type': 'token', 'token': pending}) + '\n'

        # Parse follow-ups
        followups = []
        if in_followup and followup_buffer:
            for l in followup_buffer.strip().split('\n'):
                cleaned = l.strip().lstrip('-*•0123456789.:').strip()
                # Also strip "Question N:" style prefixes
                import re as _re
                cleaned = _re.sub(r'^question\s*\d+[:.]\s*', '', cleaned, flags=_re.IGNORECASE).strip()
                if cleaned and len(cleaned) > 3:
                    followups.append(cleaned)

        if not followups:
            followups = [
                "Comment puis-je prendre rendez-vous ?",
                "Quels sont les tarifs ?",
                "Comment contacter un expert ?"
            ]

        yield json.dumps({'type': 'followup', 'followup': followups[:3]}) + '\n'
        yield json.dumps({'type': 'done'}) + '\n'

    except Exception as err:
        print(f"Error in streaming: {err}")
        yield json.dumps({
            'type': 'token',
            'token': "\n\n**[Erreur lors de la génération]**\n\n"
        }) + '\n'
        yield json.dumps({
            'type': 'followup',
            'followup': ["Réessayer la recherche", "Contacter un centre", "Voir nos services"]
        }) + '\n'
        yield json.dumps({'type': 'done'}) + '\n'


# ──────────────────────────────────────────────────────────────
#  Dynamic suggestions
# ──────────────────────────────────────────────────────────────

def get_dynamic_suggestions(context_query=None):
    """Generate dynamic search suggestions.

    - If context_query is provided (after a search), generate follow-up questions
      based on the most relevant chunks for that query.
    - Otherwise (on startup), generate suggestions from random DB chunks.
    """
    try:
        if context_query:
            # Context-aware: use semantically relevant chunks
            try:
                relevant = search_similar(context_query, limit=3)
                context = "\n\n".join([r['content'][:400] for r in relevant]) if relevant else context_query
            except Exception:
                context = context_query

            system_prompt = (
                "You are a France Pare-Brise assistant. Based on the user's last question and related "
                "context, generate exactly 4 short follow-up search queries in French.\n"
                "CRITICAL REQUIREMENTS:\n"
                "- Each suggestion must be a natural follow-up or related question to the user's question.\n"
                "- Suggestions must be answerable from the context. Do not invent topics.\n"
                "- Keep each suggestion concise (max 10 words).\n"
                "- Respond ONLY with a valid JSON object containing the key \"suggestions\"."
            )
            user_content = (
                f"Dernière question de l'utilisateur : {context_query}\n\n"
                f"Contexte associé :\n{context}\n\n"
                "IMPORTANT: Respond with valid JSON only:\n"
                "{\"suggestions\": [\"q1\", \"q2\", \"q3\", \"q4\"]}"
            )
        else:
            # Startup: random chunks
            results = query('SELECT content FROM document_chunks ORDER BY RANDOM() LIMIT 3')
            if not results:
                return [
                    "Comment prendre rendez-vous ?",
                    "Quels sont vos horaires ?",
                    "Quels services proposez-vous ?",
                    "Comment contacter un centre ?"
                ]

            context = "\n\n".join([r['content'][:500] for r in results])
            system_prompt = (
                "You are a France Pare-Brise assistant. Generate exactly 4 short and relevant search "
                "suggestions in French based on the provided context.\n"
                "CRITICAL REQUIREMENTS:\n"
                "- Every suggestion must be a specific question answerable from the context.\n"
                "- Be specific to the services, locations, or features mentioned.\n"
                "- Keep each suggestion concise (max 10 words).\n"
                "- Respond ONLY with a valid JSON object containing the key \"suggestions\"."
            )
            user_content = (
                f"Contexte :\n{context}\n\n"
                "IMPORTANT: Respond with valid JSON only:\n"
                "{\"suggestions\": [\"q1\", \"q2\", \"q3\", \"q4\"]}"
            )

        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_GEN_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                "stream": False,
                "format": "json"
            },
            timeout=60
        )
        response.raise_for_status()
        raw_content = response.json().get("message", {}).get("content", "")
        data = json.loads(raw_content)

        suggestions = data.get('suggestions') or data.get('questions') or []

        # Deduplicate and sanitise
        seen = set()
        clean = []
        for s in suggestions:
            s = s.strip()
            if s and s not in seen:
                seen.add(s)
                clean.append(s)

        if not clean:
            clean = [
                "Comment prendre rendez-vous ?",
                "Quels sont vos horaires ?",
                "Quels services proposez-vous ?",
                "Comment contacter un centre ?"
            ]

        return clean[:4]

    except Exception as err:
        print(f"Error generating dynamic suggestions: {err}")
        return [
            "Comment prendre rendez-vous ?",
            "Quels sont vos horaires ?",
            "Quels services proposez-vous ?",
            "Comment contacter un centre ?"
        ]
