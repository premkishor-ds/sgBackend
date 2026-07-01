import os
import json
import re
import math
import requests
from db import query

# Ollama settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://192.168.10.148:11434")
OLLAMA_GEN_MODEL = os.getenv("OLLAMA_GEN_MODEL", "qwen2.5:14b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

print(f"OLLAMA RAG SERVICE: Base URL: {OLLAMA_BASE_URL}, Gen: {OLLAMA_GEN_MODEL}, Embed: {OLLAMA_EMBED_MODEL}")


# ──────────────────────────────────────────────────────────────
#  Data loading & processing
# ──────────────────────────────────────────────────────────────

import csv

def load_data(directory_path):
    """Load and parse JSON or CSV files"""
    # Check for CSV first
    csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
    all_data = []

    for file in csv_files:
        filepath = os.path.join(directory_path, file)
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row['_source'] = file
                row['_type'] = 'csv'
                all_data.append(row)

    # Fallback to JSON if no CSV data loaded
    if not all_data:
        json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
        for file in json_files:
            filepath = os.path.join(directory_path, file)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                enriched_data = []
                if isinstance(data, list):
                    for item in data:
                        item['_source'] = file
                        item['_type'] = 'json'
                        enriched_data.append(item)
                else:
                    data['_source'] = file
                    data['_type'] = 'json'
                    enriched_data.append(data)
                all_data.extend(enriched_data)

    return all_data


def clean_rich_text(html_str, city_name=None):
    if not html_str:
        return ""
    # Strip HTML tags
    text = re.sub(r'<[^>]+>', ' ', html_str)
    # Replace city placeholders
    if city_name:
        text = re.sub(r'\[\[\s*address\.city\s*\]\]', city_name, text, flags=re.IGNORECASE)
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def process_data(items):
    """Process items into text with metadata"""
    processed_items = []

    for item in items:
        # Check if item is from CSV
        if item.get('_type') == 'csv':
            name = item.get('Page Heading H1') or item.get('Name') or 'France Pare-Brise'
            city = item.get('Address > City', '').strip()
            
            # Format address
            addr_parts = []
            for field in ['Address > Line 1', 'Address > Line 2', 'Address > Postal Code', 'Address > City', 'Country Name']:
                val = item.get(field, '').strip()
                if val:
                    addr_parts.append(val)
            full_address = ", ".join(addr_parts)
            
            phone = item.get('Main Phone > Phone Number', '').strip()
            
            # Coordinates
            lat = item.get('Display Lat/Long > Latitude') or item.get('Yext Display Lat/Long > Latitude') or item.get('City Lat/Long > Latitude')
            lng = item.get('Display Lat/Long > Longitude') or item.get('Yext Display Lat/Long > Longitude') or item.get('City Lat/Long > Longitude')
            coords_text = ""
            if lat and lng:
                coords_text = f"latitude {lat.strip()}\nlongitude {lng.strip()}"
            
            # Descriptions from various columns
            desc = item.get('Description', '').strip() or item.get('Business Description', '').strip()
            desc = clean_rich_text(desc, city)
            
            h2 = item.get('Business Heading H2', '').strip()
            
            desc_rtv2 = clean_rich_text(item.get('Business Description (Rich Text (v2)) > HTML', ''), city)
            desc_v2 = clean_rich_text(item.get('Description v2 > HTML', ''), city)
            
            descriptions = []
            if desc:
                descriptions.append(desc)
            if desc_rtv2 and desc_rtv2 not in descriptions:
                descriptions.append(desc_rtv2)
            if desc_v2 and desc_v2 not in descriptions:
                descriptions.append(desc_v2)
            combined_desc = "\n".join(descriptions)
            
            # Format hours
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            hours_list = []
            for day in days:
                h_val = item.get(f'Hours > {day}', '').strip()
                if h_val and h_val.lower() != 'closed':
                    hours_list.append(f"  {day}: {h_val}")
                else:
                    hours_list.append(f"  {day}: Closed")
            hours_text = "\n".join(hours_list)
            
            # Format services & advantages
            services = []
            services_val = item.get('Services') or item.get('FPB Services Locations > Name') or item.get('Related Core Services > Name')
            if services_val:
                services = [s.strip() for s in services_val.split(',') if s.strip()]
            
            advantages_val = item.get('FPB Advantages data > Name') or item.get('Related Advantages > Name')
            if advantages_val:
                services.extend([a.strip() for a in advantages_val.split(',') if a.strip()])
                
            services_text = "\n".join([f"  - {s}" for s in services])
            
            # Rich columns from Yext location object
            vehicle_types = item.get('Types de Véhicules') or item.get('Type de Vehicules > Name')
            vehicle_text = f"Vehicle Types Supported: {vehicle_types.strip()}" if vehicle_types else ""
            
            insurance = item.get('Related Insurance Companies > Name') or item.get('Insurance Company Title')
            insurance_text = f"Insurance Partners: {insurance.strip()}" if insurance else ""
            
            car_brands = item.get('Related Car Brands > Name')
            car_brands_text = f"Car Brands Supported: {car_brands.strip()}" if car_brands else ""
            
            cta_link = item.get('Book an Appointment CTA > Link') or item.get('Book Appointment CTA > Link')
            cta_text = f"Online Appointment Booking Link: {cta_link.strip()}" if cta_link else ""
            
            parts = [
                f"Center Name: {name}",
                f"Address: {full_address}",
                f"Phone: {phone}",
                coords_text,
                f"Business Heading H2: {h2}" if h2 else "",
                f"Description:\n{combined_desc}" if combined_desc else "",
                f"Opening Hours:\n{hours_text}",
                f"Services Offered:\n{services_text}" if services_text else "",
                vehicle_text,
                insurance_text,
                car_brands_text,
                cta_text
            ]
            
            text = "\n".join([p for p in parts if p])
            
            processed_items.append({
                'text': text,
                'metadata': {
                    'source': item.get('_source', 'unknown'),
                    'original_title': name
                }
            })
            continue

        # Check if this is a location object
        is_location = False
        data_obj = item.get('data', {})
        if isinstance(data_obj, dict) and data_obj.get('type') == 'location':
            is_location = True

        if is_location:
            # Format location data structured
            name = data_obj.get('name', 'Glassdrive Center')
            addr = data_obj.get('address', {})
            line1 = addr.get('line1', '')
            city = addr.get('city', '')
            postal = addr.get('postalCode', '')
            country = addr.get('countryCode', 'PT')
            full_address = f"{line1}, {city}, {postal}, {country}"
            
            phone = data_obj.get('mainPhone', '')
            desc = data_obj.get('description', '') or data_obj.get('c_businessDescription', '')
            
            # Format hours
            hours_text = ""
            hours_obj = data_obj.get('hours', {})
            if isinstance(hours_obj, dict):
                days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
                intervals = []
                for day in days:
                    day_info = hours_obj.get(day, {})
                    if day_info.get('isClosed'):
                        intervals.append(f"  {day.capitalize()}: Closed")
                    else:
                        intervals_list = day_info.get('openIntervals', [])
                        if intervals_list:
                            time_strs = [f"{t.get('start', '')}-{t.get('end', '')}" for t in intervals_list]
                            intervals.append(f"  {day.capitalize()}: {', '.join(time_strs)}")
                        else:
                            intervals.append(f"  {day.capitalize()}: Closed")
                hours_text = "\n".join(intervals)

            # Format services
            services = []
            core_services = data_obj.get('c_relatedCoreServices', [])
            if isinstance(core_services, list):
                for s in core_services:
                    s_name = s.get('name') or s.get('c_serviceName')
                    if s_name:
                        services.append(s_name)
            add_services = data_obj.get('c_relatedAdditionalService', [])
            if isinstance(add_services, list):
                for s in add_services:
                    s_name = s.get('name') or s.get('c_serviceName')
                    if s_name:
                        services.append(s_name)
            
            services_text = "\n".join([f"  - {s}" for s in services])
            
            # Format coordinates
            coords_text = ""
            coords = data_obj.get('geocodedCoordinate') or data_obj.get('yextDisplayCoordinate')
            if isinstance(coords, dict):
                coords_text = f"latitude {coords.get('latitude', '')}\nlongitude {coords.get('longitude', '')}"

            text = f"""Center Name: {name}
Address: {full_address}
Phone: {phone}
{coords_text}
Description: {desc}
Opening Hours:
{hours_text}
Services Offered:
{services_text}"""
            
            title = name
        else:
            # Walk object fallback for general services html/jsons
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
            text = '\n'.join(text_parts)
            title = item.get('metadata', {}).get('title') or item.get('title')
            if not title and 'data' in item:
                title = item['data'].get('name') or item['data'].get('title')

        processed_items.append({
            'text': text,
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

        # Clean query for robust matching: split into clean alphabetic words
        query_words = set(re.findall(r'[a-z]+', query_text.lower()))

        results = []

        for row in rows:
            chunk_emb = row['embedding']
            if not chunk_emb:
                continue
            similarity = cosine_similarity(query_emb, chunk_emb)

            meta = row['metadata']
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except:
                    meta = {}
            elif not isinstance(meta, dict):
                meta = {}

            title = meta.get('original_title', '').lower()
            center_name = (title.replace('glassdrive', '')
                           .replace('france pare-brise', '')
                           .replace('france pare brise', '')
                           .strip())
            
            # Get clean words for center name
            center_words = [re.sub(r'[^a-z]', '', w) for w in center_name.split()]
            center_words = [w for w in center_words if w]
            
            # Check if all non-empty words of the center name are present in the query words
            if center_words and all(w in query_words for w in center_words):
                similarity += 10.0

            results.append({
                'content': row['content'],
                'metadata': meta,
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

# Common words per language
# English list is large to catch short, high-frequency words that often appear
# alone in short queries like "services provided in Glassdrive Fátima"
_LANG_WORDS = {
    'en': [
        # Question words
        'what', 'where', 'when', 'how', 'which', 'who', 'why',
        # Articles & prepositions (very common in English)
        'the', 'a', 'an', 'in', 'at', 'of', 'on', 'by', 'to', 'from',
        'with', 'for', 'about', 'into', 'near', 'between', 'through',
        # Verbs
        'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'do', 'does', 'did', 'can', 'could', 'will', 'would', 'should',
        'have', 'has', 'had', 'want', 'need', 'get', 'find', 'show',
        'tell', 'give', 'list', 'provide', 'provided', 'know',
        'help', 'make', 'take', 'use', 'work', 'look', 'book',
        # Pronouns
        'i', 'my', 'me', 'we', 'our', 'you', 'your', 'it', 'its',
        # Common nouns relevant to queries
        'services', 'service', 'product', 'products', 'centre', 'center',
        'location', 'locations', 'address', 'hours', 'time', 'appointment',
        'insurance', 'repair', 'replacement', 'glass', 'windshield',
        # Other common English words
        'any', 'all', 'more', 'please', 'specific', 'available',
        'nearest', 'closest', 'nearby', 'open', 'closed',
        # Greetings
        'hello', 'hi', 'hey', 'howdy', 'greetings',
    ],
    'fr': [
        'où', 'comment', 'quel', 'quelle', 'quels', 'quelles', 'qui', 'pourquoi',
        'je', 'tu', 'vous', 'nous', 'il', 'elle', 'ils', 'elles',
        'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'au', 'aux',
        'est', 'sont', 'puis-je', 'que', 'quoi', 'qui',
        'mon', 'ma', 'mes', 'avec', 'pour', 'dans', 'sur',
        'avoir', 'être', 'faire', 'prendre', 'trouver',
        'service', 'services', 'centre', 'horaires', 'rendez-vous',
        # Greetings
        'bonjour', 'salut', 'bonsoir', 'coucou',
    ],
    'pt': [
        # Question/relative words
        'onde', 'como', 'qual', 'quais', 'quem', 'porque', 'quanto', 'que',
        # Pronouns and articles unique to PT
        'eu', 'você', 'voce', 'ele', 'ela', 'nos', 'eles', 'elas',
        'os', 'as', 'um', 'uma', 'uns', 'umas',
        # PT prepositions and contractions
        'de', 'do', 'da', 'dos', 'das', 'ao', 'aos', 'pelo', 'pela',
        'num', 'numa', 'nuns', 'numas',
        'no', 'na', 'se',
        # Common PT verbs
        'está', 'são', 'posso', 'meu', 'minha',
        'por', 'para', 'com', 'tem', 'ter', 'ser', 'faz', 'fazer',
        'foi', 'será', 'pode', 'posso', 'vende', 'vender', 'oferece', 'oferecer',
        'quero', 'preciso', 'gostaria', 'existe', 'há', 'trata',
        # PT-specific nouns (Glassdrive domain)
        'serviços', 'centro', 'horário', 'seguro', 'vidro', 'reparação',
        'para-brisas', 'substituição', 'marcação', 'processo', 'aberta',
        'sábado', 'preços', 'orçamento', 'garantia', 'contacto',
        'produtos', 'presidente', 'melhor',
        # Greetings
        'olá', 'ola', 'oi', 'bom dia', 'boa tarde', 'boa noite',
    ],
}

def detect_language(text: str) -> str:
    """Detect query language using weighted word frequency. Returns 'en', 'fr', or 'pt'."""
    lower = text.lower()
    words = set(re.findall(r"[a-záàâãäéèêëíìîïóòôõöúùûüçñ'-]+", lower))

    # Base score: 1 per matching keyword
    scores = {lang: sum(1 for w in kws if w in words) for lang, kws in _LANG_WORDS.items()}

    # HIGH-VALUE discriminators: unique-to-one-language markers score +3
    # These words appear in ONLY one language so are very reliable
    PT_STRONG = {'da', 'dos', 'das', 'ao', 'aos', 'pelo', 'pela', 'num', 'numa',
                 'faz', 'fazer', 'trata', 'foi', 'será', 'pode', 'há', 'quem', 'qual', 'quais',
                 'para-brisas', 'substituição', 'marcação', 'sábado',
                 'serviços', 'horário', 'reparação', 'vidro', 'oferece', 'vende',
                 'olá', 'ola', 'oi', 'bom dia', 'boa tarde', 'boa noite'}
    FR_STRONG = {'où', 'est-ce', 'puis-je', 'rendez-vous', 'êtes', 'être', 'horaires',
                 'les', 'des', 'aux', 'du', 'vous', 'proposez',
                 'bonjour', 'salut', 'bonsoir', 'coucou'}
    EN_STRONG = {'the', 'is', 'are', 'was', 'were', 'does', 'did', 'will', 'would',
                 'should', 'book', 'windshield', 'insurance', 'warranty', 'nearest',
                 'replacement', 'repair', 'appointment', 'location',
                 'hello', 'hi', 'hey', 'howdy', 'greetings'}

    for w in words:
        if w in PT_STRONG: scores['pt'] += 3
        if w in FR_STRONG: scores['fr'] += 3
        if w in EN_STRONG: scores['en'] += 3

    # Accented-character tiebreaker: ã, õ, ç, â+a are strongly PT
    pt_accents = len(re.findall(r'[ãõ]', lower))
    if pt_accents > 0:
        scores['pt'] += pt_accents * 2

    best = max(scores, key=scores.get)
    # Default to 'en' when all scores are zero (very short/ambiguous query)
    return best if scores[best] > 0 else 'en'


# ──────────────────────────────────────────────────────────────
#  Streaming answer generation
# ──────────────────────────────────────────────────────────────

def haversine_distance(lat1, lon1, lat2, lon2):
    """Compute haversine distance in km between two coordinate points."""
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return r * c

def extract_coords(text):
    """Parse latitude and longitude from the text content of a chunk."""
    lat_match = re.search(r'latitude\s+([-+]?[0-9]*\.?[0-9]+)', text)
    lng_match = re.search(r'longitude\s+([-+]?[0-9]*\.?[0-9]+)', text)
    if lat_match and lng_match:
        return float(lat_match.group(1)), float(lng_match.group(1))
    return None

def is_greeting(query_text):
    """Determine if query is a simple greeting."""
    q = re.sub(r'[^\w\s]', '', query_text.lower().strip())
    greetings = {
        'hi', 'hello', 'hey', 'greetings', 'howdy', 'good morning', 'good afternoon', 'good evening',
        'bonjour', 'salut', 'bonsoir', 'coucou',
        'olá', 'ola', 'oi', 'bom dia', 'boa tarde', 'boa noite'
    }
    return q in greetings or any(q.startswith(g + ' ') for g in greetings)

def is_near_query(query_text):
    """Determine if query is asking for closest/nearest centers."""
    q = query_text.lower()
    near_words = [
        'nearest', 'closest', 'nearby', 'near me', 'near you',
        'próximo', 'proximo', 'mais perto', 'perto de mim',
        'proche', 'plus proche', 'près de', 'pres de'
    ]
    return any(w in q for w in near_words)

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
            "Sorry, I can only answer questions about France Pare-Brise / Glassdrive services "
            "(centres, opening hours, glass repair/replacement, insurance, etc.).\n\n"
            "Your question seems outside our service scope. "
            "Feel free to ask me anything about France Pare-Brise / Glassdrive!"
        ),
        'fr': (
            "Je suis désolé, je ne peux répondre qu'aux questions relatives aux services "
            "France Pare-Brise / Glassdrive (centres, horaires, réparation/remplacement de vitres, assurances, etc.).\n\n"
            "Votre question semble hors du périmètre de nos services. "
            "N'hésitez pas à me poser une question sur France Pare-Brise / Glassdrive !"
        ),
        'pt': (
            "Lamentamos, só podemos responder a questões sobre os serviços "
            "France Pare-Brise / Glassdrive (centros, horários, reparação/substituição de vidros, seguros, etc.).\n\n"
            "A sua pergunta parece estar fora do âmbito dos nossos serviços. "
            "Não hesite em fazer-nos uma pergunta sobre a France Pare-Brise / Glassdrive!"
        ),
    }

    GREETINGS = {
        'en': (
            "Hello! I am your France Pare-Brise / Glassdrive customer service assistant. "
            "How can I help you with our centres, services, or bookings today?"
        ),
        'fr': (
            "Bonjour ! Je suis votre assistant service client France Pare-Brise / Glassdrive. "
            "Comment puis-je vous aider avec nos centres, services ou rendez-vous aujourd'hui ?"
        ),
        'pt': (
            "Olá! Sou o seu assistente de apoio ao cliente da France Pare-Brise / Glassdrive. "
            "Como posso ajudar com os nossos centros, serviços ou marcações hoje?"
        ),
    }

    OFF_TOPIC_FOLLOWUPS = {
        'en': ["Where is the nearest France Pare-Brise centre?", "What services do you offer?", "How can I book an appointment?"],
        'fr': ["Où se trouve le centre France Pare-Brise le plus proche ?", "Quels services proposez-vous ?", "Comment prendre rendez-vous ?"],
        'pt': ["Onde fica o centro France Pare-Brise mais próximo?", "Que serviços oferecem?", "Como posso marcar uma consulta?"],
    }

    if is_greeting(query_text):
        yield json.dumps({'type': 'sources', 'sources': []}) + '\n'
        yield json.dumps({
            'type': 'token',
            'token': GREETINGS.get(lang, GREETINGS['en'])
        }) + '\n'
        yield json.dumps({
            'type': 'followup',
            'followup': OFF_TOPIC_FOLLOWUPS.get(lang, OFF_TOPIC_FOLLOWUPS['en'])
        }) + '\n'
        yield json.dumps({'type': 'done'}) + '\n'
        return

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

    # Process location coordinates and sort context chunks if it's a proximity query
    processed_chunks = []
    for c in context_chunks:
        processed_chunks.append(dict(c))

    user_lat = location.get('lat') if location else None
    user_lng = location.get('lng') if location else None

    if user_lat is not None and user_lng is not None:
        for chunk in processed_chunks:
            coords = extract_coords(chunk['content'])
            if coords:
                c_lat, c_lng = coords
                dist = haversine_distance(user_lat, user_lng, c_lat, c_lng)
                chunk['distance'] = dist
                chunk['content'] = f"[Distance: {dist:.1f} km from user]\n" + chunk['content']
            else:
                chunk['distance'] = 999999.0

        if is_near_query(query_text):
            processed_chunks = sorted(processed_chunks, key=lambda x: x.get('distance', 999999.0))

    yield json.dumps({'type': 'sources', 'sources': context_chunks}) + '\n'

    context = ""
    for idx, c in enumerate(processed_chunks, 1):
        title = c.get('metadata', {}).get('original_title', 'Untitled')
        context += f"--- Source {idx}: {title} ---\n{c['content']}\n\n"

    system_prompt = f"""You are a France Pare-Brise / Glassdrive customer service assistant. Your ONLY role is to answer questions about France Pare-Brise and Glassdrive services.

⚠️ LANGUAGE RULE — MANDATORY: Respond ENTIRELY in {lang_name}. Do NOT use any other language.

France Pare-Brise / Glassdrive topics you can answer:
- Center locations, addresses, and opening hours
- Windshield / automotive glass repair and replacement
- ADAS calibration and driver assistance systems
- Insurance procedures for glass repair
- Appointments and service bookings
- Products, warranties, and partnerships

CRITICAL RULES:
1. Language: Respond ONLY in {lang_name}.
2. Be concise and direct. List relevant centers/information clearly. Do not write long introductions.
3. NEVER include parenthetical notes, reasoning, or internal thoughts. Start directly with the answer.
4. NO PLACEHOLDERS — NEVER output template text like [Center Name], [Address of Center],
   [City, Country], or any text in square brackets. Use ONLY real data from the context.
   If real data is unavailable, say so plainly (e.g. "Address not listed in our records").
5. BRAND NAME — MANDATORY: NEVER translate or modify the brand names.
   - Always write "France Pare-Brise" — NEVER "France Windshield" or "France Glass" or any translation.
   - Always write "Glassdrive" — never translate it.
   - Center names must be copied EXACTLY as they appear in the Context (e.g. "France Pare-Brise Hagetmau").
6. LOCATION & PROXIMITY SEARCH:
   - When listing centers, use this EXACT bullet format:
     - **<Center Name>**: <Address>, Tél: <Phone>, <Distance if present>
   - Copy addresses and phone numbers EXACTLY from the Context. Do NOT paraphrase or invent them.
   - Do NOT invent coordinates, guess locations, or comment on the user's geographic GPS coordinates.
7. If the question is NOT about France Pare-Brise or Glassdrive, reply ONLY:
   "{OFF_TOPIC.get(lang, OFF_TOPIC['en']).split(chr(10))[0]}"
8. STRICT CONTEXT CONSTRAINT — CRITICAL:
   - Use ONLY the addresses, phone numbers, center names, and details present in the provided Context below.
   - Do NOT use your pre-training knowledge about any addresses, streets, or phone numbers.
   - If a center's address in the Context is "410 Avenue de la Gare", you MUST write "410 Avenue de la Gare" — do not invent "21 Avenue de la Libération" or any other street.
   - Only list centers explicitly present in the Context. If a center is not in the Context, do not mention it.
9. After every on-topic answer, add follow-up questions in this EXACT format:
[FOLLOWUPS]
- <write a real, specific follow-up question in {lang_name} based ONLY on the centers present in the provided Context above>
- <write another real, specific follow-up question in {lang_name} based ONLY on the centers present in the provided Context above>
- <write a third real, specific follow-up question in {lang_name} based ONLY on the centers present in the provided Context above>

For example, good follow-ups look like:
- Comment prendre rendez-vous chez France Pare-Brise Hagetmau ?
- Quels sont les horaires d'ouverture le samedi ?
- Est-ce que France Pare-Brise gère les démarches avec mon assurance ?

Do not write anything after the follow-up questions."""

    user_content = f"""Context:
{context}

User question: {query_text}"""

    # Language-priming assistant turn — placed BEFORE the user question so the
    # model continues in the target language even with foreign-language context docs.
    _LANG_PRIMER = {
        'en': 'Here is the information about France Pare-Brise / Glassdrive',
        'fr': 'Voici les informations sur France Pare-Brise / Glassdrive',
        'pt': 'Aqui estão as informações sobre a France Pare-Brise / Glassdrive',
    }
    lang_primer = _LANG_PRIMER.get(lang, _LANG_PRIMER['en'])

    try:
        print(f"RAG SERVICE: Streaming answer from Ollama ({OLLAMA_GEN_MODEL}) for: {query_text}")
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_GEN_MODEL,
                "messages": [
                    {"role": "system",    "content": system_prompt},
                    {"role": "user",      "content": user_content},
                    {"role": "assistant", "content": lang_primer},
                ],
                "stream": True,
                "options": {"temperature": 0.0}
            },
            stream=True,
            timeout=120
        )
        response.raise_for_status()

        full_text = ""  # kept for debug logging if needed
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
            MARKER_PATTERN = re.compile(r'\[?FOLLOW-?UPS\]?:?', re.IGNORECASE)
            match = MARKER_PATTERN.search(pending)
            if match:
                in_followup = True
                marker_start = match.start()
                marker_end = match.end()
                before = pending[:marker_start]
                after = pending[marker_end:]
                if before.strip():
                    yield json.dumps({'type': 'token', 'token': before}) + '\n'
                followup_buffer = after
                pending = ""
                continue

            # Flush safe portion (keep last 20 chars in buffer)
            safe_len = max(0, len(pending) - 20)
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
                # Strip "Question N:" style prefixes
                cleaned = re.sub(r'^question\s*\d+[:.]\s*', '', cleaned, flags=re.IGNORECASE).strip()
                # Strip angle-bracket template text like <write a question here>
                cleaned = re.sub(r'^<.*?>$', '', cleaned, flags=re.IGNORECASE).strip()
                if cleaned and len(cleaned) > 5 and not cleaned.startswith('<'):
                    followups.append(cleaned)

        # Language-aware fallback follow-ups (used only when model produces none)
        FALLBACK_FOLLOWUPS = {
            'en': ["How do I book an appointment?", "What are the opening hours?", "How does insurance work with France Pare-Brise?"],
            'fr': ["Comment puis-je prendre rendez-vous chez France Pare-Brise ?", "Quels sont les tarifs ?", "Comment contacter un expert ?"],
            'pt': ["Como posso marcar uma consulta?", "Quais são os horários?", "Como funciona o seguro com a France Pare-Brise?"],
        }
        if not followups:
            followups = FALLBACK_FOLLOWUPS.get(lang, FALLBACK_FOLLOWUPS['en'])

        yield json.dumps({'type': 'followup', 'followup': followups[:3]}) + '\n'
        yield json.dumps({'type': 'done'}) + '\n'

    except Exception as err:
        print(f"Error in streaming: {err}")
        ERR_MSGS = {
            'en': "An error occurred while generating the response. Please try again.",
            'fr': "Une erreur s'est produite lors de la génération. Veuillez réessayer.",
            'pt': "Ocorreu um erro ao gerar a resposta. Por favor, tente novamente.",
        }
        ERR_FOLLOWUPS = {
            'en': ["Try the search again", "Contact a centre", "See our services"],
            'fr': ["Réessayer la recherche", "Contacter un centre", "Voir nos services"],
            'pt': ["Tentar novamente", "Contactar um centro", "Ver os nossos serviços"],
        }
        yield json.dumps({'type': 'token', 'token': ERR_MSGS.get(lang, ERR_MSGS['en'])}) + '\n'
        yield json.dumps({'type': 'followup', 'followup': ERR_FOLLOWUPS.get(lang, ERR_FOLLOWUPS['en'])}) + '\n'
        yield json.dumps({'type': 'done'}) + '\n'


# ──────────────────────────────────────────────────────────────
#  Dynamic suggestions
# ──────────────────────────────────────────────────────────────

# High-quality French fallback suggestions — varied so each startup feels fresh
_FALLBACK_SUGGESTIONS = [
    "Où se trouve le centre France Pare-Brise le plus proche ?",
    "Comment prendre rendez-vous chez France Pare-Brise ?",
    "Quels sont les horaires d'ouverture du samedi ?",
    "France Pare-Brise prend-il en charge mon assurance ?",
    "Est-ce que France Pare-Brise propose le remplacement à domicile ?",
    "Quels véhicules sont acceptés chez France Pare-Brise ?",
    "Comment contacter un expert France Pare-Brise ?",
    "France Pare-Brise effectue-t-il le recalibrage ADAS ?",
]

def get_dynamic_suggestions(context_query=None):
    """Generate dynamic search suggestions.

    - If context_query is provided (after a search), generate follow-up questions
      based on the most relevant chunks for that query (same language as query).
    - Otherwise (on startup), generate specific French suggestions using real
      center names and services from the DB.
    """
    try:
        if context_query:
            # ── Post-search: context-aware follow-ups ──────────────
            try:
                relevant = search_similar(context_query, limit=4)
                context = "\n\n".join([r['content'][:400] for r in relevant]) if relevant else context_query
            except Exception:
                context = context_query
                
            lang = detect_language(context_query)
            LANG_NAMES = {'en': 'English', 'fr': 'French', 'pt': 'Portuguese'}
            lang_name = LANG_NAMES.get(lang, 'French')

            system_prompt = (
                "You are a France Pare-Brise customer service assistant. "
                "Based on the user's last question and the related context below, "
                f"generate exactly 4 short follow-up search queries in {lang_name}.\n"
                "CRITICAL REQUIREMENTS:\n"
                f"- ALWAYS respond in {lang_name}. This is mandatory.\n"
                "- Each suggestion must be a natural, specific follow-up to the user's question.\n"
                "- Use real center names, services, or details from the context.\n"
                "- NEVER mention competitor brands. Only France Pare-Brise / Glassdrive.\n"
                "- Keep each suggestion concise (max 12 words).\n"
                "- Respond ONLY with a valid JSON object: {\"suggestions\": [\"q1\",\"q2\",\"q3\",\"q4\"]}"
            )
            
            # Use English instructions here to not bias the model into French, 
            # relying on the explicit {lang_name} instruction instead.
            user_content = (
                f"User's last question: {context_query}\n\n"
                f"Related context:\n{context}\n\n"
                "IMPORTANT: Respond ONLY with valid JSON:\n"
                "{\"suggestions\": [\"q1\", \"q2\", \"q3\", \"q4\"]}"
            )

        else:
            # ── Startup: use real center names + services from DB ──
            # Pull diverse chunks: mix center names and service descriptions
            results = query(
                "SELECT content FROM document_chunks ORDER BY RANDOM() LIMIT 6"
            )
            if not results:
                import random
                return random.sample(_FALLBACK_SUGGESTIONS, 4)

            context = "\n\n".join([r['content'][:600] for r in results])

            system_prompt = (
                "You are a France Pare-Brise customer service assistant. "
                "Generate exactly 4 search suggestions a customer would realistically type, "
                "based ONLY on the France Pare-Brise data provided below.\n"
                "CRITICAL REQUIREMENTS:\n"
                "- ALL 4 suggestions MUST be written in FRENCH.\n"
                "- Use specific center names, cities, services, or details from the context.\n"
                "- Each suggestion must be a realistic customer question (not generic).\n"
                "- Mix different topics: location, opening hours, services, appointments, insurance.\n"
                "- NEVER write generic suggestions like 'How do I book?' — be specific.\n"
                "- NEVER translate brand name: always write 'France Pare-Brise', never 'France Windshield'.\n"
                "- Keep each suggestion concise (max 12 words).\n"
                "- Respond ONLY with valid JSON: {\"suggestions\": [\"q1\",\"q2\",\"q3\",\"q4\"]}"
            )
            user_content = (
                f"Données France Pare-Brise :\n{context}\n\n"
                "Génère 4 suggestions de recherche SPÉCIFIQUES en FRANÇAIS.\n"
                "Réponds UNIQUEMENT avec du JSON valide :\n"
                "{\"suggestions\": [\"q1\", \"q2\", \"q3\", \"q4\"]}"
            )

        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_GEN_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_content}
                ],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.4}
            },
            timeout=60
        )
        response.raise_for_status()
        raw_content = response.json().get("message", {}).get("content", "")
        data = json.loads(raw_content)

        suggestions = data.get('suggestions') or data.get('questions') or []

        # Deduplicate, strip and validate
        seen = set()
        clean = []
        for s in suggestions:
            s = s.strip().strip('"').strip()
            # Skip if too short, empty, or looks like a placeholder
            if s and len(s) > 8 and s not in seen and not s.startswith('<'):
                seen.add(s)
                clean.append(s)

        if len(clean) < 2:
            import random
            clean = random.sample(_FALLBACK_SUGGESTIONS, 4)

        return clean[:4]

    except Exception as err:
        print(f"Error generating dynamic suggestions: {err}")
        import random
        return random.sample(_FALLBACK_SUGGESTIONS, 4)

