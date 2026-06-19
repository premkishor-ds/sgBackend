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
#  Streaming answer generation
# ──────────────────────────────────────────────────────────────

def generate_answer_stream(query_text, context_chunks):
    """Generate streaming answer and follow-up questions from Ollama"""
    if not context_chunks:
        yield json.dumps({'type': 'sources', 'sources': []}) + '\n'
        yield json.dumps({
            'type': 'token',
            'token': (
                "Je suis uniquement en mesure de répondre aux questions relatives aux services "
                "Glassdrive (centres, horaires, réparation/remplacement de vitres, assurances, etc.).\n\n"
                "Votre question semble hors du périmètre de nos services. "
                "N'hésitez pas à me poser une question sur Glassdrive !"
            )
        }) + '\n'
        yield json.dumps({
            'type': 'followup',
            'followup': ["Où se trouve le centre Glassdrive le plus proche ?", "Quels services proposez-vous ?", "Comment prendre rendez-vous ?"]
        }) + '\n'
        yield json.dumps({'type': 'done'}) + '\n'
        return

    yield json.dumps({'type': 'sources', 'sources': context_chunks}) + '\n'

    context = '\n\n---\n\n'.join([c['content'] for c in context_chunks])

    system_prompt = """You are a Glassdrive customer service assistant. Your ONLY role is to answer questions about Glassdrive services.

Glassdrive topics you can answer:
- Glassdrive center locations, addresses, and opening hours
- Windshield / automotive glass repair and replacement
- ADAS calibration and driver assistance systems
- Insurance procedures for glass repair
- Appointments and service bookings
- Glassdrive-specific products, warranties, and partnerships

CRITICAL RULES:
1. If the question is NOT about Glassdrive or its services, you MUST reply ONLY with:
   "Je suis uniquement en mesure de répondre aux questions concernant les services Glassdrive. Pour toute autre question, veuillez contacter directement un centre."
   Do NOT attempt to answer off-topic questions under any circumstances.
2. Answer ONLY using the provided context. Do not add information from outside the context.
3. Always respond in French.
4. At the end of every on-topic answer, write exactly:
[FOLLOWUPS]
- Follow-up question 1 ?
- Follow-up question 2 ?
- Follow-up question 3 ?

Do not write anything after the follow-up questions."""

    user_content = f"""Contexte FPB :
{context}

Question Utilisateur :
{query_text}"""

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
        # Pending buffer: hold back up to len("[FOLLOWUPS]") chars so the
        # marker is never partially sent to the client
        MARKER = "[FOLLOWUPS]"
        pending = ""

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

            pending += token
            full_text += token

            # Check if marker has fully arrived
            if MARKER in pending:
                in_followup = True
                before, after = pending.split(MARKER, 1)
                # Flush everything before the marker
                if before.strip():
                    yield json.dumps({'type': 'token', 'token': before}) + '\n'
                followup_buffer = after
                pending = ""
                continue

            # Flush all but the last (len(MARKER)-1) chars — those could be
            # the start of a split marker arriving in the next token
            safe_len = max(0, len(pending) - len(MARKER) + 1)
            if safe_len > 0:
                yield json.dumps({'type': 'token', 'token': pending[:safe_len]}) + '\n'
                pending = pending[safe_len:]

        # Flush any remaining pending text (no marker arrived)
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
