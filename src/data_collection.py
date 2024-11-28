import requests
from bs4 import BeautifulSoup
import os

def scrape_gutenberg(book_urls, output_file):
    """
    Scrape text from Project Gutenberg books and save to a file.
    Args:
        book_urls (list): List of URLs for Gutenberg books.
        output_file (str): Path to save the collected text.
    """
    all_text = []
    for url in book_urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract the book's text (assumes Gutenberg format)
        text = soup.get_text()
        start_index = text.find("*** START OF THIS PROJECT GUTENBERG EBOOK")
        end_index = text.find("*** END OF THIS PROJECT GUTENBERG EBOOK")
        if start_index != -1 and end_index != -1:
            text = text[start_index:end_index]
        all_text.append(text)
        print(f"Scraped: {url}")

    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_text))
    print(f"Saved collected data to {output_file}")

if __name__ == "__main__":
    urls = [
        "https://www.gutenberg.org/files/46/46-h/46-h.htm",  # A Christmas Carol
        "https://www.gutenberg.org/files/19337/19337-h/19337-h.htm",  # The Night Before Christmas
        "https://www.gutenberg.org/files/2591/2591-h/2591-h.htm",  # Little Women
    ]
    output_path = "../data/christmas_corpus.txt"
    scrape_gutenberg(urls, output_path)
