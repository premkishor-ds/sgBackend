from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import os
import sys
from datetime import datetime
import logging
from db import init_db, query
import ragService

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Setup logging
log_file = os.path.join(os.path.dirname(__file__), 'server.log')
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def log_message(msg):
    logger.info(msg)
    print(msg)

@app.route('/')
def home():
    return 'RAG Server is up. access /app/index.html for frontend or /test for testapi'

# Serve the main frontend
@app.route('/app')
@app.route('/app/<path:path>')
def serve_frontend(path=''):
    frontend_path = os.path.join(os.path.dirname(__file__), '..', 'frontend')
    if path == '':
        path = 'index.html'
    return send_from_directory(frontend_path, path)

# Serve the testapi.html
@app.route('/test')
def serve_test():
    test_path = os.path.join(os.path.dirname(__file__), 'testapi.html')
    return send_from_directory(os.path.dirname(__file__), 'testapi.html')

@app.route('/suggestions', methods=['GET'])
def get_suggestions():
    """Return dynamic suggested search queries from backend"""
    # Optional context: last user query to generate conversation-aware suggestions
    context_query = request.args.get('context', '').strip()
    suggestions = ragService.get_dynamic_suggestions(context_query=context_query if context_query else None)
    return jsonify({'suggestions': suggestions})

@app.route('/search', methods=['POST'])
def search():
    log_message(f'Search request received: {request.get_json()}')
    
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({'error': 'Query is required'}), 400
    
    query_text = data['query']
    location = data.get('location')  # Optional: { lat: float, lng: float }

    try:
        # Retrieval
        relevant_chunks = ragService.search_similar(query_text, 5)

        # Relevance gate: if the top result similarity is below threshold,
        # the query is off-topic — pass empty chunks so we return a refusal
        # without ever calling the LLM
        RELEVANCE_THRESHOLD = 0.28
        if relevant_chunks and relevant_chunks[0].get('similarity', 0) < RELEVANCE_THRESHOLD:
            log_message(f"Off-topic query rejected (best similarity: {relevant_chunks[0].get('similarity', 0):.3f}): {query_text}")
            relevant_chunks = []

        # Stream Generation
        return Response(
            ragService.generate_answer_stream(query_text, relevant_chunks, location=location),
            mimetype='text/event-stream'
        )
    except Exception as err:
        log_message(f'Search error: {err}')
        return jsonify({'error': 'Failed to process request'}), 500

def setup():
    """Initialize DB and Process Data"""
    init_db()
    
    # Script to ingest data if table is empty
    count_result = query('SELECT COUNT(*) as count FROM document_chunks')
    if count_result[0]['count'] == 0:
        print('Ingesting data...')
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        raw_items = ragService.load_data(data_dir)
        processed_items = ragService.process_data(raw_items)
        
        for item in processed_items:
            chunks = ragService.chunk_text(item['text'])
            for chunk in chunks:
                emb = ragService.get_embedding(chunk)
                ragService.store_in_db(chunk, item['metadata'], emb)
        
        print('Ingestion complete.')

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    
    try:
        setup()
        log_message(f'Server running on http://localhost:{port}')
        app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)
    except Exception as err:
        log_message(f'Setup error: {err}')
