import os
import json
from bs4 import BeautifulSoup
import glob

def extract_from_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Metadata
    title = soup.title.string.strip() if soup.title else os.path.basename(file_path)
    # Remove branding from title if present
    if ' | Glassdrive' in title:
        title = title.replace(' | Glassdrive', '')
    
    description = ""
    desc_tag = soup.find('meta', attrs={'name': 'description'})
    if desc_tag:
        description = desc_tag.get('content', '').strip()
    
    url = ""
    url_tag = soup.find('link', attrs={'rel': 'canonical'})
    if url_tag:
        url = url_tag.get('href', '').strip()
    
    # Sections
    sections = []
    
    # Try to find main content
    main_content = soup.find('main')
    if not main_content:
        main_content = soup.find('article')
    if not main_content:
        main_content = soup.body
        
    if main_content:
        # Look for headings and their following content
        current_section = {"heading": title, "content": [], "list_items": []}
        
        # Flattened list of tags we care about in the main area
        # We also look for specific divs that might contain content if the main is too large
        content_area = main_content.find('div', class_='layout-content') or main_content
        
        for element in content_area.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'ul', 'ol']):
            # Skip if element is inside footer or header
            if element.find_parent(['header', 'footer']):
                continue
                
            if element.name in ['h1', 'h2', 'h3', 'h4']:
                # Save previous section if it has content
                if current_section["content"] or current_section["list_items"]:
                    sections.append(current_section)
                
                current_section = {
                    "heading": element.get_text(separator=' ', strip=True),
                    "content": [],
                    "list_items": []
                }
            elif element.name == 'p':
                text = element.get_text(separator=' ', strip=True)
                if text and len(text) > 10: # Skip very short snippets
                    current_section["content"].append(text)
            elif element.name in ['ul', 'ol']:
                for li in element.find_all('li'):
                    li_text = li.get_text(separator=' ', strip=True)
                    if li_text:
                        current_section["list_items"].append(li_text)
        
        # Add the last section
        if current_section["content"] or current_section["list_items"]:
            sections.append(current_section)

    return {
        "filename": os.path.basename(file_path),
        "metadata": {
            "title": title,
            "description": description,
            "url": url,
            "breadcrumbs": []
        },
        "sections": sections
    }

def main():
    data_dir = r'd:\saintGobainSearch\Backend\data'
    output_file = os.path.join(data_dir, 'glassdrive_services.json')
    
    html_files = glob.glob(os.path.join(data_dir, '*.html'))
    
    all_data = []
    for html_file in html_files:
        print(f"Processing {html_file}...")
        try:
            data = extract_from_html(html_file)
            all_data.append(data)
        except Exception as e:
            print(f"Error processing {html_file}: {e}")
            
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    
    print(f"Successfully saved {len(all_data)} items to {output_file}")

if __name__ == "__main__":
    main()
