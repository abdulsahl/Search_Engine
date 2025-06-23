import requests
from bs4 import BeautifulSoup
from newspaper import Article, Config as NewspaperConfig, ArticleException
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import os
import re
import time
from collections import deque
import traceback
import json
import hashlib
import signal

# config
START_URLS = ["https://www.cnnindonesia.com/olahraga/sepakbola"]
DOCS_DIR = '../docs'
MAX_ARTICLES_TO_SAVE = 2400
MIN_TEXT_LENGTH = 200
USER_AGENT = "MyAdvancedCrawler/1.2 (+http://example.com/bot.html)"
POLITENESS_DELAY = 1
ALLOWED_DOMAINS = []
ARTICLE_LANGUAGE = 'id'

# File untuk menyimpan state resume
QUEUE_FILE = 'crawl_queue.json'
PROCESSED_URLS_FILE = 'crawl_processed_urls.json'
SAVED_HASHES_FILE = 'crawl_saved_hashes.json'

# --- Global Vars for state saving ---
queue = deque()
urls_in_system = set() # Termasuk yang di antrian dan yang sudah dikunjungi
saved_article_text_hashes = set()
articles_saved_count = 0

# --- FUNGSI UTILITAS ---
def sanitize_filename_with_spaces(title):
    if not title:
        title = "Untitled_Article_From_Crawler"
    title = re.sub(r'[<>:"/\\|?*]', '', title)
    title = re.sub(r'\s+', ' ', title).strip()
    max_len = 180
    if len(title) > max_len:
        title_kepotong = title[:max_len]
        if ' ' in title_kepotong:
            title = title_kepotong.rsplit(' ', 1)[0]
        else:
            title = title_kepotong
    return title + ".txt"

def get_normalized_domain(netloc_str):
    netloc_str = netloc_str.lower()
    if netloc_str.startswith("www."):
        return netloc_str[4:]
    return netloc_str

# --- FUNGSI STATE (RESUME) ---
def load_state():
    global queue, urls_in_system, saved_article_text_hashes, articles_saved_count
    
    # Hitung dulu artikel yang sudah ada di direktori
    if os.path.exists(DOCS_DIR):
        articles_saved_count = len([name for name in os.listdir(DOCS_DIR) if os.path.isfile(os.path.join(DOCS_DIR, name)) and name.endswith('.txt')])
        print(f"Resume: Ditemukan {articles_saved_count} artikel yang sudah disimpan.")


    if os.path.exists(QUEUE_FILE):
        try:
            with open(QUEUE_FILE, 'r') as f:
                queue_list = json.load(f)
                queue = deque(queue_list)
            print(f"Resume: Antrian ({len(queue)} URL) dimuat dari {QUEUE_FILE}")
        except Exception as e:
            print(f"Resume Error: Gagal memuat antrian dari {QUEUE_FILE}: {e}")
            queue = deque() # Mulai dengan antrian kosong jika error

    if os.path.exists(PROCESSED_URLS_FILE):
        try:
            with open(PROCESSED_URLS_FILE, 'r') as f:
                urls_in_system = set(json.load(f))
            print(f"Resume: Data URL ({len(urls_in_system)}) dimuat dari {PROCESSED_URLS_FILE}")
        except Exception as e:
            print(f"Resume Error: Gagal memuat data URL dari {PROCESSED_URLS_FILE}: {e}")
            urls_in_system = set()

    if os.path.exists(SAVED_HASHES_FILE):
        try:
            with open(SAVED_HASHES_FILE, 'r') as f:
                saved_article_text_hashes = set(json.load(f))
            print(f"Resume: Hash artikel ({len(saved_article_text_hashes)}) dimuat dari {SAVED_HASHES_FILE}")
        except Exception as e:
            print(f"Resume Error: Gagal memuat hash dari {SAVED_HASHES_FILE}: {e}")
            saved_article_text_hashes = set()
    
    # Tambahkan START_URLS jika belum ada di sistem (untuk crawl baru atau jika antrian kosong)
    initial_urls_added_to_queue = 0
    for start_url in START_URLS:
        if start_url not in urls_in_system:
            queue.appendleft(start_url) # Tambahkan ke depan untuk diproses dulu jika crawl baru
            urls_in_system.add(start_url)
            initial_urls_added_to_queue +=1
    if initial_urls_added_to_queue > 0:
        print(f"Resume: Menambahkan {initial_urls_added_to_queue} URL awal baru ke antrian.")


def save_state():
    global queue, urls_in_system, saved_article_text_hashes
    print("\nMenyimpan state crawler...")
    try:
        with open(QUEUE_FILE, 'w') as f:
            json.dump(list(queue), f)
        print(f"Antrian ({len(queue)} URL) disimpan ke {QUEUE_FILE}")
    except Exception as e:
        print(f"Error menyimpan antrian: {e}")

    try:
        with open(PROCESSED_URLS_FILE, 'w') as f:
            json.dump(list(urls_in_system), f)
        print(f"Data URL ({len(urls_in_system)}) disimpan ke {PROCESSED_URLS_FILE}")
    except Exception as e:
        print(f"Error menyimpan data URL: {e}")

    try:
        with open(SAVED_HASHES_FILE, 'w') as f:
            json.dump(list(saved_article_text_hashes), f)
        print(f"Hash artikel ({len(saved_article_text_hashes)}) disimpan ke {SAVED_HASHES_FILE}")
    except Exception as e:
        print(f"Error menyimpan hash: {e}")

def signal_handler(sig, frame):
    print('\nInterupsi diterima (Ctrl+C), menghentikan crawler dan menyimpan state...')
    save_state()
    exit(0)

signal.signal(signal.SIGINT, signal_handler)


# --- FUNGSI CRAWLER UTAMA ---
def main_crawler():
    global queue, urls_in_system, saved_article_text_hashes, articles_saved_count # Gunakan global

    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)

    load_state() # Muat state sebelumnya

    robot_parsers = {}
    newspaper_config = NewspaperConfig()
    newspaper_config.browser_user_agent = USER_AGENT
    newspaper_config.request_timeout = 15
    newspaper_config.fetch_images = False
    newspaper_config.memoize_articles = False
    newspaper_config.verbose = False

    print(f"Memulai/Melanjutkan crawling dari: {START_URLS if not queue else 'antrian yang ada'}")
    print(f"Target menyimpan {MAX_ARTICLES_TO_SAVE} artikel (total, termasuk yang sudah ada).")
    print(f"Artikel saat ini tersimpan: {articles_saved_count}")
    print(f"Dokumen akan disimpan di '{DOCS_DIR}'")

    # Pengisian otomatis ALLOWED_DOMAINS dengan varian www dan non-www
    # Ini dijalankan setiap kali, tapi tidak masalah karena set akan menangani duplikasi
    final_allowed_domains = set()
    # Jika ALLOWED_DOMAINS di konfigurasi kosong, gunakan domain dari START_URLS
    domains_to_process_for_allowed_list = ALLOWED_DOMAINS if ALLOWED_DOMAINS else [urlparse(s).netloc for s in START_URLS if urlparse(s).netloc]
    
    for domain_item in domains_to_process_for_allowed_list:
        normalized_domain = get_normalized_domain(domain_item)
        if normalized_domain:
            final_allowed_domains.add(normalized_domain)
            final_allowed_domains.add("www." + normalized_domain) # Pastikan www juga ada
    
    effective_allowed_domains = list(final_allowed_domains)
    if effective_allowed_domains:
        print(f"Domain yang diizinkan: {effective_allowed_domains}")
    else:
        print("PERINGATAN: Tidak ada domain yang diizinkan, crawler mungkin tidak akan mengambil apapun.")


    processed_url_this_session_counter = 0
    try:
        while queue and articles_saved_count < MAX_ARTICLES_TO_SAVE:
            current_url = queue.popleft()
            processed_url_this_session_counter +=1
            
            print(f"\n[Sesi URL ke-{processed_url_this_session_counter}][Tersimpan: {articles_saved_count}/{MAX_ARTICLES_TO_SAVE}] Proses: {current_url}")

            parsed_url = urlparse(current_url)
            current_netloc = parsed_url.netloc.lower()

            is_domain_allowed = False
            if not effective_allowed_domains: # Jika tidak ada batasan domain, semua diizinkan
                is_domain_allowed = True
            else:
                for allowed_dom_pattern in effective_allowed_domains:
                    # Cek apakah current_netloc sama dengan allowed_dom_pattern atau variannya
                    if get_normalized_domain(current_netloc) == get_normalized_domain(allowed_dom_pattern):
                        is_domain_allowed = True
                        break
            
            if not is_domain_allowed:
                print(f"  -> Skip (luar domain): {current_netloc}")
                continue

            # Normalisasi domain untuk kunci cache robots.txt dan path robots.txt
            robots_domain_key = get_normalized_domain(current_netloc)
            if robots_domain_key not in robot_parsers:
                # Selalu coba ambil robots.txt dari domain root (tanpa www) jika memungkinkan,
                # atau biarkan RobotFileParser menanganinya. Untuk simpel:
                robots_url_path = f"{parsed_url.scheme}://{robots_domain_key}/robots.txt"
                rp = RobotFileParser()
                rp.set_url(robots_url_path)
                try:
                    rp.read()
                    robot_parsers[robots_domain_key] = rp
                except Exception as e:
                    print(f"  -> Peringatan: Gagal baca robots.txt dari {robots_url_path}: {e}")
                    robot_parsers[robots_domain_key] = None # Anggap boleh jika gagal baca
            
            robot_parser_for_domain = robot_parsers.get(robots_domain_key)
            if robot_parser_for_domain and not robot_parser_for_domain.can_fetch(USER_AGENT, current_url):
                print(f"  -> Dilarang robots.txt")
                continue
            
            try:
                article = Article(current_url, config=newspaper_config, language=ARTICLE_LANGUAGE)
                article.download()
                article.parse()

                page_title = article.title
                page_text = article.text.strip()
                
                if page_title and page_text and len(page_text) >= MIN_TEXT_LENGTH:
                    text_hash = hashlib.md5(page_text.encode('utf-8')).hexdigest()
                    if text_hash in saved_article_text_hashes:
                        print(f"  -> Artikel duplikat (hash sama): '{page_title}'. Dilewati.")
                    else:
                        print(f"  -> Artikel kandidat: '{page_title}' (Teks: {len(page_text)} char)")
                        filename_base = sanitize_filename_with_spaces(page_title)
                        filepath = os.path.join(DOCS_DIR, filename_base)
                        
                        counter = 1
                        original_filename_base = filename_base
                        while os.path.exists(filepath):
                            name_part, ext_part = os.path.splitext(original_filename_base)
                            name_part = re.sub(r'_\d+$', '', name_part)
                            filepath = os.path.join(DOCS_DIR, f"{name_part}_{counter}{ext_part}")
                            counter += 1
                        
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(current_url + "\n")
                            f.write(page_title + "\n\n")
                            f.write(page_text)
                        
                        articles_saved_count += 1
                        saved_article_text_hashes.add(text_hash)
                        print(f"  -> SUKSES disimpan: {filepath}")
                else:
                    if not page_title: print(f"  -> Bukan artikel (tidak ada judul).")
                    elif not page_text: print(f"  -> Bukan artikel (tidak ada teks).")
                    else: print(f"  -> Bukan artikel (teks terlalu pendek: {len(page_text)} char).")

                if article.html:
                    soup = BeautifulSoup(article.html, 'html.parser')
                    links_added_this_iteration = 0
                    for link_tag in soup.find_all('a', href=True):
                        href = link_tag['href']
                        absolute_url = urljoin(current_url, href)
                        parsed_absolute_url = urlparse(absolute_url)
                        abs_netloc = parsed_absolute_url.netloc.lower()

                        is_abs_link_domain_allowed = False
                        if not effective_allowed_domains: is_abs_link_domain_allowed = True
                        else:
                            for allowed_dom_pattern in effective_allowed_domains:
                                if get_normalized_domain(abs_netloc) == get_normalized_domain(allowed_dom_pattern):
                                    is_abs_link_domain_allowed = True
                                    break
                        
                        if parsed_absolute_url.scheme in ['http', 'https'] and \
                           not parsed_absolute_url.fragment and \
                           absolute_url not in urls_in_system and \
                           is_abs_link_domain_allowed:
                            
                            queue.append(absolute_url)
                            urls_in_system.add(absolute_url)
                            links_added_this_iteration +=1
                    if links_added_this_iteration > 0:
                        print(f"  -> Menambah {links_added_this_iteration} link baru ke antrian.")
                else:
                    print(f"  -> Gagal dapat HTML dari {current_url} untuk cari link.")
                
                time.sleep(POLITENESS_DELAY)

            except ArticleException as ae: # Tangani error spesifik dari newspaper3k
                print(f"  -> newspaper3k ArticleException saat proses {current_url}: {str(ae)}")
            except Exception as e:
                print(f"  -> ERROR proses {current_url}: {str(e)}")
                # traceback.print_exc() 
    
    finally: # Pastikan state disimpan apapun yang terjadi (selesai normal atau error tak tertangani)
        save_state()

    print(f"\nCrawling sesi ini selesai.")
    if articles_saved_count < MAX_ARTICLES_TO_SAVE and not queue :
        print(f"Antrian URL habis sebelum target {MAX_ARTICLES_TO_SAVE} artikel (total) tercapai.")
    print(f"Total {articles_saved_count} artikel (kumulatif) tersimpan di '{DOCS_DIR}'.")
    print(f"Total URL diproses sesi ini: {processed_url_this_session_counter}. Sisa di antrian: {len(queue)}.")

if __name__ == '__main__':
    main_crawler()