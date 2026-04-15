import psycopg2
import os
from dotenv import load_dotenv
import uuid

load_dotenv()

class Database:
    def __init__(self):
        self.connection_string = os.getenv('DATABASE_URL')
        self.pool = None
    
    def get_connection(self):
        return psycopg2.connect(self.connection_string)
    
    def query(self, query_text, params=None):
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(query_text, params)
                if query_text.strip().upper().startswith('SELECT'):
                    columns = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()
                    return [dict(zip(columns, row)) for row in rows]
                else:
                    conn.commit()
                    return cursor.rowcount
        finally:
            conn.close()
    
    def init_db(self):
        try:
            # Create extension
            self.query('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')
            
            # Drop table if it exists to refresh the search_vector definition
            self.query('DROP TABLE IF EXISTS document_chunks;')
            
            # Create table with full-text search
            create_table_query = '''
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    content TEXT NOT NULL,
                    metadata JSONB,
                    search_vector tsvector GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            '''
            self.query(create_table_query)
            
            # Create index for search
            self.query('CREATE INDEX IF NOT EXISTS idx_chunks_search ON document_chunks USING GIN (search_vector);')
            
            print('Database initialized with Simple Full-Text Search.')
        except Exception as err:
            print(f'Error initializing database: {err}')
            raise err

# Create global instance
db = Database()

# Export functions for compatibility with Node.js version
def query(text, params=None):
    return db.query(text, params)

def init_db():
    return db.init_db()
