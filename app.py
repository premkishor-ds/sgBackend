from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
from datetime import datetime
import logging
from db import init_db, query
import ragService

app = Flask(__name__)
CORS(app)

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

@app.route('/search', methods=['POST'])
def search():
    log_message(f'Search request received: {request.get_json()}')
    
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({'error': 'Query is required'}), 400
    
    query_text = data['query']
    
    try:
        # Retrieval
        relevant_chunks = ragService.search_similar(query_text, 5)
        
        # Generation
        result = ragService.generate_answer(query_text, relevant_chunks)
        print(f"BACKEND DEBUG: Search Result: {result}")
        
        return jsonify(result)
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
                ragService.store_in_db(chunk, item['metadata'])
        
        print('Ingestion complete.')

if __name__ == '__main__':
    port = int(os.getenv('PORT', 3001))
    
    try:
        setup()
        log_message(f'Server running on http://localhost:{port}')
        app.run(host='0.0.0.0', port=port, debug=True)
    except Exception as err:
        log_message(f'Setup error: {err}')
