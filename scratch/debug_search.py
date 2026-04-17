import sys
import os

# Add Backend to path
sys.path.append(os.path.join(os.getcwd(), 'Backend'))

from db import query
import ragService

try:
    print("Testing search_similar...")
    res = ragService.search_similar("test", 5)
    print(f"Result count: {len(res)}")
except Exception as e:
    print(f"Error caught in scratch script: {e}")
    import traceback
    traceback.print_exc()
