import feedparser
import requests
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_single_episode(url, filename):
    """Helper function to download a single episode"""
    if os.path.exists(filename):
        print(f"Skipping {filename} - already exists")
        return False
        
    try:
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        print(f"Successfully downloaded {filename}")
        return True
        
    except Exception as e:
        print(f"Error downloading {filename}: {str(e)}")
        return False

def download_podcast(rss_url, output_dir, max_workers=4):
    """
    Downloads all podcast episodes from an RSS feed URL using parallel downloads
    
    Args:
        rss_url (str): URL of the podcast RSS feed
        output_dir (str): Directory to save downloaded files
        max_workers (int): Maximum number of concurrent downloads
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Parse the RSS feed
    feed = feedparser.parse(rss_url)
    print(f"Found {len(feed.entries)} episodes")

    # Create a list of download tasks
    download_tasks = []
    for entry in feed.entries:
        for enclosure in entry.enclosures:
            if enclosure.type and enclosure.type.startswith('audio'):
                url_path = enclosure.href.split('?')[0]
                filename = os.path.join(output_dir, os.path.basename(url_path))
                download_tasks.append((enclosure.href, filename))

    # Download files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_single_episode, url, filename) 
                  for url, filename in download_tasks]
        
        # Wait for all downloads to complete
        for future in as_completed(futures):
            future.result()  # This ensures we see any exceptions that occurred

if __name__ == "__main__":    
    # download_podcast("https://feeds.transistor.fm/technology-brother", "data/technology-brothers/audio")
    download_podcast("https://audioboom.com/channels/5114330.rss", "data/girls-next-level/audio")