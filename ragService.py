import os
import json
import re
import requests
from dotenv import load_dotenv
load_dotenv(override=True)
from db import query
import openai

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY not found in environment!")
else:
    print(f'RAG SERVICE: Using API Key starting with: {OPENAI_API_KEY[:12]}...')

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

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
                    # Skip technical keys and metadata we don't want in the text
                    if key.startswith('_') or key in ['metadata', 'id', 'created_at', 'search_vector', 'slug', 'thumbnails', 'url', 'image', 'height', 'width', 'locale', 'primary_key', 'image_url', 'favicon']:
                        continue
                    if isinstance(val, (dict, list)):
                        walk_obj(val)
                    elif val and not isinstance(val, bool):
                        str_val = str(val).strip()
                        # Skip URLs, images, and purely numeric/technical short strings
                        if (str_val.startswith(('http', '//')) or 
                            str_val.endswith(('.png', '.jpg', '.jpeg', '.webp', '.pdf')) or
                            re.match(r'^[0-9\s\-\.\:\/]+$', str_val) or # Just numbers or separators
                            len(str_val) < 4):
                            continue
                        text_parts.append(str_val)
            elif isinstance(obj, list):
                for item_in_list in obj:
                    walk_obj(item_in_list)
        
        walk_obj(item)
        
        # Better title extraction for both services and locations
        title = item.get('metadata', {}).get('title') or item.get('title')
        if not title and 'data' in item:
            # Fallback for location.json structure
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

def store_in_db(chunk, metadata):
    """Store chunk in database"""
    import json
    sql = 'INSERT INTO document_chunks (content, metadata) VALUES (%s, %s)'
    query(sql, [chunk, json.dumps(metadata)])

def search_similar(query_text, limit=5):
    """Search for similar chunks using full-text search"""
    try:
        sql = '''
            SELECT content, metadata, ts_rank_cd(search_vector, websearch_to_tsquery('simple', %s)) AS similarity
            FROM document_chunks
            WHERE search_vector @@ websearch_to_tsquery('simple', %s)
            OR content ILIKE '%%' || %s || '%%'
            ORDER BY similarity DESC
            LIMIT %s
        '''
        result = query(sql, [query_text, query_text, query_text, limit])
        
        if len(result) == 0:
            print('No direct matches found for query.')
            return []
        
        return result
    except Exception as err:
        print(f'DATABASE ERROR in search_similar: {err}')
        raise err

def generate_answer(query_text, context_chunks):
    """Generate answer and follow-up questions using OpenAI"""
    if not context_chunks:
        return {
            'answer': "Désolé, je n'ai pas trouvé d'informations spécifiques pour répondre à votre question dans nos documents. N'hésitez pas à reformuler ou à contacter un centre France Pare-Brise.",
            'sources': [],
            'followup': ["Comment puis-je prendre rendez-vous ?", "Où se trouve le centre le plus proche ?", "Quels sont vos horaires ?"]
        }
    
    context = '\n\n---\n\n'.join([c['content'] for c in context_chunks])
    
    try:
        print(f"RAG SERVICE: Requesting answer from gpt-4o for query: {query_text}")
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """Vous êtes un expert France Pare-Brise. Répondez de manière amicale et professionnelle.
                    
                    IMPORTANT : Vous devez TOUJOURS répondre au format JSON suivant :
                    {
                        "answer": "Votre réponse détaillée ici (en Markdown)",
                        "followup": ["Question de suivi 1 ?", "Question de suivi 2 ?", "Question de suivi 3 ?"]
                    }
                    Assurez-vous que les questions de suivi sont pertinentes par rapport au contexte et à la question de l'utilisateur."""
                },
                {
                    "role": "user",
                    "content": f"""Contexte FPB :
{context}

Question Utilisateur :
{query_text}"""
                }
            ],
            response_format={ "type": "json_object" },
            max_tokens=1000,
            temperature=0.7
        )
        
        raw_content = response.choices[0].message.content
        print(f"RAG SERVICE: AI Raw Response: {raw_content}")
        
        result_json = json.loads(raw_content)
        answer = result_json.get("answer", "Erreur lors de la génération du résumé.")
        followup = result_json.get("followup", [
            "Comment puis-je prendre rendez-vous ?",
            "Quels sont les tarifs ?",
            "Comment contacter un expert ?"
        ])
        
        return {
            'answer': answer,
            'sources': context_chunks,
            'followup': followup[:3] # Ensure exactly 3
        }
    except Exception as err:
        print(f'RAG SERVICE ERROR: {err}')
        
        return {
            'answer': "**[SERVICE IA INDISPONIBLE]**\n\nVoici les documents trouvés :\n\n" + 
                     '\n\n'.join([f"### {c['metadata'].get('original_title')}\n{c['content'][:200]}..." for c in context_chunks]),
            'sources': context_chunks,
            'followup': ["Réessayer la recherche", "Contacter un centre", "Voir nos services"]
        }

def get_dynamic_suggestions():
    """Generate dynamic search suggestions based on DB content"""
    try:
        # Get random chunks from DB to generate variety
        sql = 'SELECT content FROM document_chunks ORDER BY RANDOM() LIMIT 3'
        results = query(sql)
        
        if not results:
            return [
                "Comment prendre rendez-vous ?",
                "Quels sont vos horaires ?",
                "Quels services proposez-vous ?",
                "Comment contacter un centre ?"
            ]
            
        context = "\n\n".join([r['content'][:500] for r in results])
        
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Générez 4 suggestions de recherche courtes et pertinentes pour un utilisateur de France Pare-Brise, basées sur le contexte fourni. Répondez UNIQUEMENT avec un objet JSON : {\"suggestions\": [\"q1\", \"q2\", \"q3\", \"q4\"]}"
                },
                {
                    "role": "user",
                    "content": f"Contexte :\n{context}"
                }
            ],
            response_format={ "type": "json_object" }
        )
        
        data = json.loads(response.choices[0].message.content)
        return data.get('suggestions', [])
    except Exception as err:
        print(f"Error generating dynamic suggestions: {err}")
        return [
            "Comment prendre rendez-vous ?",
            "Quels sont vos horaires ?",
            "Quels services proposez-vous ?",
            "Comment contacter un centre ?"
        ]
