import os
import json
import re
from bs4 import BeautifulSoup

data_dir = r"d:\saintGobainSearch\Backend\data"
output_file = r"d:\saintGobainSearch\Backend\data\extracted_data.json"

results = []

def clean_text(text):
    # Remove video player controls artifacts
    artifacts = [
        "Pause", "Play", "% buffered", "Unmute", "Mute", 
        "Disable captions", "Enable captions", "Settings", "Captions",
        "Disabled", "Quality", "undefined", "Speed", "Normal", 
        "Go back to previous menu", "0.5×", "0.75×", "1.25×", "1.5×", 
        "1.75×", "2×", "PIP", "Exit fullscreen", "Enter fullscreen"
    ]
    # Remove timestamps like 00:00 or -00:14
    text = re.sub(r'-?\d{1,2}:\d{2}', '', text)
    
    for art in artifacts:
        text = text.replace(art, "")
    
    # Remove excessive whitespace and control chars
    text = text.replace('\u00a0', ' ') # Non-breaking space
    text = text.replace('\u200b', '') # Zero-width space
    text = re.sub(r' +', ' ', text)
    return text.strip()

if not os.path.exists(data_dir):
    print(f"Error: Directory {data_dir} does not exist.")
    exit(1)

for filename in os.listdir(data_dir):
    if filename.endswith(".html"):
        filepath = os.path.join(data_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f, "html.parser")
                
                # Metadata
                title = soup.title.string.strip() if soup.title else filename
                description_meta = soup.find("meta", {"name": "description"})
                description = description_meta["content"].strip() if description_meta and "content" in description_meta.attrs else ""
                
                canonical = soup.find("link", {"rel": "canonical"})
                url = canonical["href"] if canonical and "href" in canonical.attrs else ""
                
                breadcrumb_nav = soup.find("nav", {"aria-label": "Breadcrumb"})
                breadcrumbs = [li.get_text(strip=True) for li in breadcrumb_nav.find_all("li")] if breadcrumb_nav else []

                # Cleanup HTML
                for element in soup(["script", "style", "svg", "noscript", "form"]):
                    element.decompose()
                
                # Extraction
                main_content = soup.find("main") or soup.find("div", id="main-content") or \
                               soup.find("div", class_="region-content") or soup.find("body")

                sections = []
                if main_content:
                    current_section = {"heading": "Introduction", "content": [], "list_items": []}
                    
                    # We extract h-tags, p, and li in order
                    for elem in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li']):
                        text = clean_text(elem.get_text(strip=True))
                        if not text: continue
                        
                        if elem.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                            if current_section["content"] or current_section["list_items"]:
                                sections.append(current_section)
                            current_section = {"heading": text, "content": [], "list_items": []}
                        elif elem.name == 'p':
                            current_section["content"].append(text)
                        elif elem.name == 'li':
                            current_section["list_items"].append(text)
                    
                    if current_section["content"] or current_section["list_items"] or current_section["heading"] != "Introduction":
                        sections.append(current_section)

                results.append({
                    "filename": filename,
                    "metadata": {
                        "title": title,
                        "description": description,
                        "url": url,
                        "breadcrumbs": breadcrumbs
                    },
                    "sections": sections
                })
        except Exception as e:
            print(f"Error processing {filename}: {e}")

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Successfully extracted structured data from {len(results)} files to {output_file}")
