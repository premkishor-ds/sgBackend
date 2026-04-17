import sys
import os
sys.path.append(os.getcwd())
from db import query
import json

res = query("SELECT content FROM document_chunks WHERE metadata->>'source' = 'location.json' LIMIT 3")
for r in res:
    print("-" * 20)
    print(r['content'])
