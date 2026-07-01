import requests
import json
import sys

def test_live_query(query_text, location=None):
    url = "http://localhost:8000/search"
    payload = {
        "query": query_text,
        "location": location
    }
    
    print(f"Querying: '{query_text}'")
    if location:
        print(f"Location coordinates: {location}")
    print("-" * 60)
    
    try:
        response = requests.post(url, json=payload, stream=True, timeout=30)
        response.raise_for_status()
        
        answer = ""
        sources = []
        followups = []
        
        for line in response.iter_lines():
            if not line:
                continue
            decoded_line = line.decode('utf-8').strip()
            try:
                event = json.loads(decoded_line)
                evt_type = event.get("type")
                
                if evt_type == "sources":
                    sources = event.get("sources", [])
                    print(f"Sources Consulted: {[s.get('metadata', {}).get('original_title') for s in sources]}")
                    print("\nFirst 2 Chunks Context:")
                    for idx, s in enumerate(sources[:2], 1):
                        print(f"--- Chunk {idx} ({s.get('metadata', {}).get('original_title')}) ---")
                        print(s.get('content')[:500])
                    print("="*60)
                    print("\nStreaming Response:")
                elif evt_type == "token":
                    token = event.get("token", "")
                    answer += token
                    sys.stdout.write(token)
                    sys.stdout.flush()
                elif evt_type == "followup":
                    followups = event.get("followup", [])
                elif evt_type == "done":
                    print("\n" + "-" * 60)
            except Exception as e:
                # In case of non-json debug lines
                pass
                
        print(f"\nSuggestions/Follow-up Questions:")
        for i, f in enumerate(followups, 1):
            print(f"  {i}. {f}")
            
    except Exception as e:
        print(f"Error querying API: {e}")

if __name__ == "__main__":
    # Test 1: With GPS location
    print("=== TEST 1: WITH GPS LOCATION ===")
    test_live_query(
        query_text="services provided in Glassdrive Fátima",
        location={"lat": 37.354107, "lng": -121.955238}
    )
    print("\n" + "="*60 + "\n")
    # Test 2: Without GPS location
    print("=== TEST 2: WITHOUT GPS LOCATION ===")
    test_live_query(
        query_text="services provided in Glassdrive Fátima",
        location=None
    )
