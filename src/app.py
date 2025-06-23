from flask import Flask, render_template, request, jsonify, redirect
import os
import re
import time
import math
import pickle
import json
import threading
import numpy as np
import jellyfish
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi, BM25Plus
from spellchecker import SpellChecker
from collections import Counter

app = Flask(__name__)

# --- Inisialisasi & Konfigurasi Global ---
doc_folder = '../docs'
CACHE_FILE = 'data/doc_cache.pkl'
RELEVANCE_FILE = 'data/relevan.json' # Untuk boosting skor dari feedback
MANUAL_RELEVANCE_FILE = 'data/query_relevan.json' # Untuk menampilkan di kotak relevan manual
IRRELEVANCE_FILE = 'data/tidak_relevan.json'
QUERY_LOG_FILE = 'data/query_log.txt'
file_lock = threading.Lock()

# --- Muat stopwords dan stemmer ---
stopwords = set(open('data/stopwords.txt', encoding='utf-8').read().splitlines())
stemmer = StemmerFactory().create_stemmer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stopwords]
    return ' '.join(words)

# --- Muat data relevansi & riwayat query ---
def load_json_data(filepath):
    data_map = {}
    if not os.path.exists(filepath):
        with open(filepath, 'w', encoding='utf-8') as f: json.dump([], f)
    with open(filepath, encoding='utf-8') as f:
        try:
            raw_data = json.load(f)
            for item in raw_data:
                data_map[item['query'].lower()] = item['dokumen_relevan']
        except json.JSONDecodeError: pass
    return data_map

human_relevance = load_json_data(RELEVANCE_FILE) # Untuk boosting skor
manual_relevance = load_json_data(MANUAL_RELEVANCE_FILE) # Untuk tampilan di UI
human_irrelevance = load_json_data(IRRELEVANCE_FILE)

logged_queries = []
if os.path.exists(QUERY_LOG_FILE):
    with open(QUERY_LOG_FILE, 'r', encoding='utf-8') as f:
        logged_queries = [line.strip() for line in f.readlines() if line.strip()]

# --- Caching Dokumen ---
def get_doc_files(folder):
    return sorted([f for f in os.listdir(folder) if f.endswith('.txt') and f != 'urls.txt'])

cache_valid = False
if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, 'rb') as f:
            # Tambahkan top_ngrams saat unpack pickle
            documents, doc_names, doc_urls, tfidf_matrix, bm25, bm25_plus, all_words, top_ngrams = pickle.load(f)
        if set(doc_names) == set(get_doc_files(doc_folder)):
            print("Cache valid.")
            cache_valid = True
        else: print("Cache usang. Membuat ulang cache...")
    except Exception: print("Gagal memuat cache. Membuat ulang cache...")

if not cache_valid:
    documents, doc_names, doc_urls, all_words = [], [], {}, set()
    for filename in get_doc_files(doc_folder):
        with open(os.path.join(doc_folder, filename), encoding='utf-8') as f:
            lines = f.readlines()
            if lines:
                first_line = lines[0].strip()
                doc_urls[filename] = first_line if first_line.startswith('http') else None
                content = ''.join(lines[1:]) if first_line.startswith('http') else ''.join(lines)
                processed = preprocess(content)
                documents.append(processed)
                doc_names.append(filename)
                all_words.update(processed.split())

    # --- MENGGUNAKAN TF-IDF UNTUK N-GRAM ---
    ngram_tfidf_vectorizer = TfidfVectorizer(ngram_range=(2, 5), stop_words=list(stopwords))
    X_ngrams_tfidf = ngram_tfidf_vectorizer.fit_transform(documents)

    # Jumlahkan skor TF-IDF untuk setiap N-gram di semua dokumen sebagai indikator kepentingan
    ngram_scores = X_ngrams_tfidf.sum(axis=0).getA1()
    ngram_features = ngram_tfidf_vectorizer.get_feature_names_out()

    # Gabungkan N-gram dengan skornya dan urutkan
    ngram_tfidf_scores = []
    for i, ngram in enumerate(ngram_features):
        ngram_tfidf_scores.append((ngram, ngram_scores[i]))

    ngram_tfidf_scores.sort(key=lambda x: x[1], reverse=True)

    # Ambil 5000 frasa dengan skor TF-IDF tertinggi
    top_ngrams = [ngram[0] for ngram in ngram_tfidf_scores[:5000]]

    # Bagian ini tetap sama
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(documents)
    tokenized_corpus = [doc.split() for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_plus = BM25Plus(tokenized_corpus)

    with open(CACHE_FILE, 'wb') as f:
        # Pastikan top_ngrams dari TF-IDF yang disimpan
        pickle.dump((documents, doc_names, doc_urls, tfidf_matrix, bm25, bm25_plus, all_words, top_ngrams), f)

spell = SpellChecker(language=None)
spell.word_frequency.load_words(all_words)
tfidf = TfidfVectorizer()
tfidf.fit(documents)

RESULTS_PER_PAGE = 5

def highlight_keywords(text, keywords):
    for kw in keywords:
        if kw.strip():
            pattern = re.compile(f'\\b{re.escape(kw)}\\b', re.IGNORECASE)
            text = pattern.sub(f'<mark>{kw}</mark>', text)
    return text

@app.route('/')
def home():
    return render_template('index.html', results=None, no_results=False)

@app.route('/autocomplete')
def autocomplete():
    query = request.args.get('q', '').lower()
    response_data = {'history': [], 'suggestions': []}
    seen_suggestions = set()
    q_stripped = query.strip()

    # --- 1. Ambil dari Riwayat Pencarian (berdasarkan frekuensi) ---
    if logged_queries:
        query_counts = Counter(logged_queries)
        
        # Ambil semua query unik dan urutkan berdasarkan frekuensi (paling sering duluan)
        all_sorted_queries = sorted(query_counts.keys(), key=lambda q: query_counts[q], reverse=True)

        # Filter berdasarkan inputan user (jika ada)
        if q_stripped:
            candidate_queries = [q for q in all_sorted_queries if q.startswith(q_stripped) and q != q_stripped]
        else: # Jika input kosong, tampilkan riwayat teratas
            candidate_queries = all_sorted_queries

        # Tambahkan ke hasil
        for logged_query in candidate_queries:
            if logged_query not in seen_suggestions:
                response_data['history'].append(logged_query)
                seen_suggestions.add(logged_query)
                # Batasi jumlah riwayat: 5 jika input kosong, 3 jika ada input
                limit = 5 if not q_stripped else 3
                if len(response_data['history']) >= limit:
                    break
    
    # Jika input kosong, hanya tampilkan riwayat dan berhenti
    if not q_stripped:
        return jsonify(response_data)

    # --- 2. Saran Berdasarkan Frasa Utuh (N-gram) ---
    for ngram in top_ngrams:
        if ngram.startswith(q_stripped) and ngram != q_stripped and ngram not in seen_suggestions:
            response_data['suggestions'].append(ngram)
            seen_suggestions.add(ngram)
            if len(response_data['suggestions']) >= 4:
                break
    
    # --- 3. Saran Pelengkapan Kata Terakhir ---
    if query and not query.endswith(' '):
        words = query.split()
        if words:
            last_word = words[-1]
            prefix = ' '.join(words[:-1])
            for word in all_words:
                if word.startswith(last_word):
                    suggestion_candidate = (prefix + ' ' + word).strip()
                    if suggestion_candidate != q_stripped and suggestion_candidate not in seen_suggestions:
                        response_data['suggestions'].append(suggestion_candidate)
                        seen_suggestions.add(suggestion_candidate)
                        if len(response_data['suggestions']) >= 8:
                            break
    
    response_data['suggestions'] = sorted(list(set(response_data['suggestions'])))[:8]
    return jsonify(response_data)

@app.route('/delete_query', methods=['POST'])
def delete_query():
    data = request.get_json()
    query_to_delete = data.get('query', '').lower()
    if not query_to_delete:
        return jsonify({'status': 'error', 'message': 'Query tidak ada'}), 400
    
    with file_lock:
        if os.path.exists(QUERY_LOG_FILE):
            temp_queries = [line.strip() for line in open(QUERY_LOG_FILE, 'r', encoding='utf-8') if line.strip() != query_to_delete]
            with open(QUERY_LOG_FILE, 'w', encoding='utf-8') as f:
                for q in temp_queries:
                    f.write(q + '\n')
            global logged_queries
            logged_queries = temp_queries
    
    return jsonify({'status': 'sukses'})


@app.route('/open_doc/<doc_name>')
def open_doc(doc_name):
    url = doc_urls.get(doc_name)
    if url: return redirect(url)
    doc_path = os.path.join(doc_folder, doc_name)
    if os.path.exists(doc_path):
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read().replace('\n', '<br>')
        return f"<h1>{doc_name[:-4]}</h1><hr><p>{content}</p>"
    return redirect('/')


@app.route('/search', methods=['POST', 'GET'])
def search():
    original_query = request.form.get('query', request.args.get('query', '')).strip()
    page = int(request.form.get('page', request.args.get('page', 1)))
    sort_by = request.form.get('sort_by', request.args.get('sort_by', 'bm25'))
    sort_order = request.form.get('sort_order', request.args.get('sort_order', 'desc'))

    if not original_query: return redirect('/')

    start_time = time.time()
    
    # Log kueri asli
    if original_query.lower() not in logged_queries:
        with file_lock:
            with open(QUERY_LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(original_query.lower() + '\n')
            logged_queries.append(original_query.lower())

    # SELALU proses kueri asli untuk pencarian
    processed_query = preprocess(original_query)
    tokenized_query = processed_query.split()
    
    correction = None # Variabel baru untuk saran

    # Batas maksimal kesalahan ketik yang bisa ditoleransi
    max_distance_threshold = math.floor(len(original_query) / 4) + 1
    best_match = None
    min_distance = float('inf')

    # Iterasi melalui kamus frasa kita (top_ngrams) untuk menemukan yang paling mirip
    for ngram in top_ngrams:
        distance = jellyfish.levenshtein_distance(original_query.lower(), ngram)
        if distance < min_distance:
            min_distance = distance
            best_match = ngram

    # Setelah loop, cek apakah kita menemukan saran yang bagus
    if best_match and 0 < min_distance <= max_distance_threshold and best_match.lower() != original_query.lower():
        correction = best_match # Simpan frasa saran
    
    tfidf_query = tfidf.transform([processed_query])
    cosine_scores = cosine_similarity(tfidf_query, tfidf_matrix).flatten()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_plus_scores = bm25_plus.get_scores(tokenized_query)
    
    results = []
    for idx, name in enumerate(doc_names):
        snippet = ' '.join(documents[idx].split()[:40]) + '...'
        results.append({
            'doc': name, 'url': f'/open_doc/{name}', 'snippet': snippet,
            'cosine': cosine_scores[idx], 'bm25': bm25_scores[idx], 
            'bm25_plus': bm25_plus_scores[idx], 'full_doc': documents[idx]
        })

    # Feedback boosting (dari relevan.json) dan penalti tetap menggunakan original_query
    docs_to_penalize = human_irrelevance.get(original_query.lower(), [])
    docs_to_boost = human_relevance.get(original_query.lower(), [])
    for r in results:
        if r['doc'] in docs_to_boost:
            r['cosine']+=1000; r['bm25']+=1000; r['bm25_plus']+=1000
        if r['doc'] in docs_to_penalize:
            r['cosine']-=1000; r['bm25']-=1000; r['bm25_plus']-=1000

    results = [r for r in results if r['bm25'] > -500 or r['cosine'] > -500]
    
    # Ambil dokumen relevan manual HANYA untuk ditampilkan (dari query_relevan.json)
    manual_docs_for_display = manual_relevance.get(original_query.lower(), [])

    if not results:
        return render_template('index.html', query=original_query, results=None, page=1, total_pages=1,
                               exec_time=round(time.time() - start_time, 4), 
                               correction=correction,
                               no_results=True, sort_by=sort_by, sort_order=sort_order, 
                               human_docs=manual_docs_for_display,
                               query_display=original_query)
    
    results.sort(key=lambda x: x[sort_by], reverse=(sort_order == 'desc'))

    total_results = len(results)
    total_pages = math.ceil(total_results / RESULTS_PER_PAGE)
    page = max(1, min(page, total_pages))
    start_idx, end_idx = (page - 1) * RESULTS_PER_PAGE, page * RESULTS_PER_PAGE
    paged_results = results[start_idx:end_idx]
    
    # Highlighting menggunakan tokenized_query asli
    for r in paged_results:
        r['snippet'] = highlight_keywords(r['snippet'], tokenized_query)
        r['full_doc_highlight'] = highlight_keywords(r['full_doc'], tokenized_query)

    return render_template('index.html', query=original_query, results=paged_results, page=page,
                           total_pages=total_pages, exec_time=round(time.time() - start_time, 4), 
                           correction=correction,
                           no_results=False, sort_by=sort_by, sort_order=sort_order, 
                           human_docs=manual_docs_for_display, 
                           query_display=original_query)


def update_feedback_file(filepath, query, doc_name):
    with file_lock:
        data = []
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                try: data = json.load(f)
                except json.JSONDecodeError: pass
        
        query_entry = next((item for item in data if item['query'] == query), None)
        if query_entry:
            if doc_name not in query_entry['dokumen_relevan']:
                query_entry['dokumen_relevan'].append(doc_name)
        else:
            data.append({'query': query, 'dokumen_relevan': [doc_name]})
            
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    query, doc_name = data.get('query', '').lower(), data.get('doc_name')
    if not query or not doc_name: return jsonify({'status': 'error'}), 400
    # Feedback (jempol ke atas) disimpan ke relevan.json untuk boosting
    update_feedback_file(RELEVANCE_FILE, query, doc_name)
    human_relevance[query] = human_relevance.get(query, []) + [doc_name]
    return jsonify({'status': 'sukses'})

@app.route('/feedback_irrelevant', methods=['POST'])
def feedback_irrelevant():
    data = request.get_json()
    query, doc_name = data.get('query', '').lower(), data.get('doc_name')
    if not query or not doc_name: return jsonify({'status': 'error'}), 400
    update_feedback_file(IRRELEVANCE_FILE, query, doc_name)
    human_irrelevance[query] = human_irrelevance.get(query, []) + [doc_name]
    return jsonify({'status': 'sukses'})

@app.route('/refresh_cache')
def refresh_cache():
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=False)