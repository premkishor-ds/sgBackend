import sys
import os
sys.path.append(os.getcwd())
from db import query
import json

res = query("SELECT metadata->>'source' as source, count(*) as count FROM document_chunks GROUP BY metadata->>'source'")
for r in res:
    print(f"{r['source']}: {r['count']}")
