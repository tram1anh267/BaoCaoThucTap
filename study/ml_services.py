"""
ML Services – Phân tích điểm yếu học sinh bằng Machine Learning
Sử dụng: TF-IDF + K-Means Clustering + Gemini AI labeling
"""

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from collections import Counter

from .models import ExamSession


def collect_wrong_questions(user, subject=None):
    """
    Thu thập tất cả câu hỏi trả lời sai từ ExamSession.
    Trả về list of dict: {question, options, correct_index, user_answer, explanation, session_id}
    """
    filters = {'user': user, 'is_submitted': True}
    if subject:
        filters['subject'] = subject

    sessions = ExamSession.objects.filter(**filters)
    wrong_questions = []

    for session in sessions:
        questions = json.loads(session.questions_json)
        answers = json.loads(session.answers_json)

        for i, q in enumerate(questions):
            user_ans = answers[i] if i < len(answers) else None
            correct_idx = q['correct_index']
            if user_ans != correct_idx:
                wrong_questions.append({
                    'question': q['question'],
                    'options': q.get('options', []),
                    'correct_index': correct_idx,
                    'user_answer': user_ans,
                    'explanation': q.get('explanation', ''),
                    'session_id': session.id,
                    'subject_name': session.subject.name,
                })
    return wrong_questions


def cluster_wrong_questions(wrong_questions, n_clusters=None):
    """
    Nhóm các câu sai theo chủ đề bằng TF-IDF + K-Means.
    Tự động chọn số cluster tối ưu nếu không chỉ định.
    
    Returns: {
        'clusters': [{
            'cluster_id': int,
            'topic_keywords': [str],        # top TF-IDF keywords
            'questions': [wrong_question],   # câu hỏi trong cluster
            'count': int,
            'percentage': float,
        }],
        'total_wrong': int,
        'n_clusters': int,
        'silhouette_score': float,
        'tfidf_feature_names': [str],
    }
    """
    if len(wrong_questions) < 3:
        return {
            'clusters': [],
            'total_wrong': len(wrong_questions),
            'n_clusters': 0,
            'silhouette_score': 0,
            'message': 'Cần ít nhất 3 câu sai để phân tích. Hãy thi thêm!'
        }

    # Bước 1: TF-IDF Vectorization
    # Danh sách từ dừng kết hợp tiếng Anh và các từ phổ biến/từ đề thi tiếng Việt
    custom_stop_words = list(ENGLISH_STOP_WORDS) + [
        # Từ nối, đại từ cơ bản TV
        "là", "của", "và", "trong", "các", "khi", "có", "không", "để", "với", "cho", "thì", 
        "được", "những", "đã", "này", "đó", "tại", "vào", "ra", "làm", "sự", "như", "hay", 
        "hoặc", "nên", "ở", "từ", "một", "bị", "bởi", "thấy", "rằng", "rất",
        # Từ ngữ đặc thù trong đề thi trắc nghiệm (cả Anh & Việt)
        "câu", "hỏi", "đáp", "án", "đúng", "sai", "chọn", "phương", "nào", "sau", "đây", 
        "ý", "phát", "biểu", "mệnh", "đề", "trắc", "nghiệm", "dưới", "nhất", "điền", "trống",
        "thế", "người", "ta", "việc", "nhận", "định", "đặc", "điểm",
        "correct", "incorrect", "answer", "question", "options", "option", "choose", 
        "following", "statement", "statements", "true", "false", "blank", "best", 
        "which", "what", "where", "when", "how", "why", "who"
    ]

    texts = [q['question'] + ' ' + q.get('explanation', '') for q in wrong_questions]

    vectorizer = TfidfVectorizer(
        max_features=200,
        stop_words=custom_stop_words,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    # Bước 2: Chọn số cluster tối ưu (Elbow method + Silhouette)
    if n_clusters is None:
        max_k = min(len(wrong_questions) - 1, 8)
        max_k = max(max_k, 2)
        best_k = 2
        best_score = -1

        for k in range(2, max_k + 1):
            try:
                km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
                labels = km.fit_predict(tfidf_matrix)
                if len(set(labels)) < 2:
                    continue
                score = silhouette_score(tfidf_matrix, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception:
                continue
        n_clusters = best_k

    # Bước 3: K-Means Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(tfidf_matrix)

    # Silhouette score
    sil_score = 0
    if len(set(labels)) >= 2:
        sil_score = round(silhouette_score(tfidf_matrix, labels), 3)

    # Bước 4: Trích xuất top keywords cho mỗi cluster
    clusters = []
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_questions = [wrong_questions[i] for i in cluster_indices]

        # Top TF-IDF keywords cho cluster
        cluster_tfidf = tfidf_matrix[cluster_indices].toarray().mean(axis=0)
        top_keyword_indices = cluster_tfidf.argsort()[-5:][::-1]
        top_keywords = [feature_names[i] for i in top_keyword_indices if cluster_tfidf[i] > 0]

        clusters.append({
            'cluster_id': cluster_id,
            'topic_keywords': top_keywords,
            'topic_name': '',  # Sẽ được Gemini điền sau
            'questions': cluster_questions,
            'count': len(cluster_questions),
            'percentage': round(len(cluster_questions) / len(wrong_questions) * 100, 1),
        })

    # Chỉ dùng các từ khoá TF-IDF để định dạng tên chủ đề
    for c in clusters:
        # Lấy tối đa 3 từ khoá đầu tiên nối lại làm tên chủ đề, viết hoa chữ cái đầu
        kw_str = ", ".join(c['topic_keywords'][:3])
        c['topic_name'] = f"Chủ đề: {kw_str.capitalize()}"

    # Sắp xếp theo số câu sai giảm dần (điểm yếu lớn nhất trước)
    clusters.sort(key=lambda c: c['count'], reverse=True)

    return {
        'clusters': clusters,
        'total_wrong': len(wrong_questions),
        'n_clusters': n_clusters,
        'silhouette_score': sil_score,
    }


def generate_weakness_report(user, subject=None):
    """
    Tạo báo cáo tổng hợp điểm yếu cho user.
    Returns dict sẵn sàng trả về JSON cho frontend.
    """
    # Thu thập câu sai
    wrong_qs = collect_wrong_questions(user, subject)

    if not wrong_qs:
        return {
            'status': 'no_data',
            'message': 'Chưa có dữ liệu. Bạn cần thi thử ít nhất 1 lần!',
            'total_wrong': 0,
        }

    # Clustering
    result = cluster_wrong_questions(wrong_qs)

    if not result['clusters']:
        return {
            'status': 'insufficient',
            'message': result.get('message', 'Không đủ dữ liệu để phân tích.'),
            'total_wrong': result['total_wrong'],
        }

    # Thống kê tổng quan
    sessions = ExamSession.objects.filter(user=user, is_submitted=True)
    if subject:
        sessions = sessions.filter(subject=subject)

    total_questions_attempted = sum(s.total_questions for s in sessions)
    total_correct = sum(s.correct_count for s in sessions)
    avg_score = round(
        sum(s.score for s in sessions if s.score is not None) / max(sessions.count(), 1), 2
    )

    # Build report
    weakness_topics = []
    for cluster in result['clusters']:
        # Lấy mẫu câu hỏi sai (tối đa 3 câu)
        sample_questions = [
            {
                'question': q['question'],
                'correct_answer': q['options'][q['correct_index']] if q['options'] and q['correct_index'] < len(q['options']) else '',
                'user_answer': q['options'][q['user_answer']] if q['options'] and q['user_answer'] is not None and q['user_answer'] < len(q['options']) else 'Chưa trả lời',
                'explanation': q.get('explanation', ''),
            }
            for q in cluster['questions'][:3]
        ]

        weakness_topics.append({
            'topic_name': cluster['topic_name'],
            'topic_keywords': cluster['topic_keywords'],
            'wrong_count': cluster['count'],
            'percentage': cluster['percentage'],
            'sample_questions': sample_questions,
            'severity': 'high' if cluster['percentage'] >= 40 else ('medium' if cluster['percentage'] >= 20 else 'low'),
            # Thu thập toàn bộ ID/nội dung câu hỏi của cụm này gửi xuống Frontend luôn để luyện tập
            'cluster_questions': [q['question'] for q in cluster['questions']]
        })

    return {
        'status': 'success',
        'summary': {
            'total_exams': sessions.count(),
            'total_questions': total_questions_attempted,
            'total_correct': total_correct,
            'total_wrong': result['total_wrong'],
            'accuracy': round(total_correct / max(total_questions_attempted, 1) * 100, 1),
            'avg_score': avg_score,
        },
        'ml_info': {
            'algorithm': 'TF-IDF + K-Means Clustering',
            'n_clusters': result['n_clusters'],
            'silhouette_score': result['silhouette_score'],
        },
        'weakness_topics': weakness_topics,
    }

# ─────────────────────────────────────────────────────────
# Text Summarization – Extractive (TF-IDF + TextRank) + Abstractive (Gemini)
# ─────────────────────────────────────────────────────────

import re
import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
_summary_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
_SUMMARY_MODEL = os.getenv("MODEL_NAME", "gemini-2.0-flash-lite")


def _split_sentences(text):
    """Tach text thanh danh sach cau (loai bo ky tu nhieu tu slide/PDF)."""
    # Loai bo email va URL, loai cac day so dai
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\b\d{8,}\b', '', text)  # day so > 8 ky tu (SDT, footer)
    
    # Noi cac cau bi ngat dong giua chung trong PDF
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    
    # Giam khoang trang
    text = re.sub(r'\s+', ' ', text)
    
    # Tach bang dau cau
    raw = re.split(r'(?<=[.!?])\s+', text)
    
    sentences = []
    seen = set()
    for s in raw:
        s = s.strip()
        # Filter 1: do dai toi thieu 30 ky tu, it nhat 6 tu tieng Viet
        if len(s) > 30 and len(s.split()) >= 6:
            # Filter 2: khong de lot cau toan so hay ki tu dac biet
            if not re.match(r'^[\d\W_]+$', s):
                if s not in seen:
                    sentences.append(s)
                    seen.add(s)
                    
    # Fallback neu lo loc het sach (vd slide chi chua phrase ngan)
    if not sentences:
        raw_fallback = re.split(r'\n+', text)
        sentences = [s.strip() for s in raw_fallback if len(s.strip()) > 15]
        
    return sentences


def textrank_extractive_summary(text, num_sentences=5):
    """
    Extractive Summarization bang TF-IDF + TextRank (graph-based).
    
    Quy trinh:
    1. Tach van ban thanh cau
    2. TF-IDF vectorization cho tung cau
    3. Xay dung ma tran cosine similarity (do thi)
    4. Tinh TextRank score cho moi cau (tuong tu PageRank)
    5. Chon top-N cau quan trong nhat
    """
    sentences = _split_sentences(text)

    if len(sentences) <= num_sentences:
        return {
            'extractive_summary': ' '.join(sentences),
            'top_sentences': [{'sentence': s, 'score': 1.0, 'original_index': i}
                              for i, s in enumerate(sentences)],
            'total_sentences': len(sentences),
            'tfidf_features': 0,
        }

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
    )
    tfidf_matrix = vectorizer.fit_transform(sentences)
    n_features = len(vectorizer.get_feature_names_out())

    # Cosine Similarity Matrix (Graph adjacency)
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(tfidf_matrix)

    # TextRank Algorithm (Power iteration)
    n = len(sentences)
    damping = 0.85
    scores = np.ones(n) / n
    
    # Normalize similarity matrix (row-wise -> transition matrix)
    row_sums = sim_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    transition = sim_matrix / row_sums

    # Power iteration (30 rounds)
    for _ in range(30):
        new_scores = (1 - damping) / n + damping * transition.T @ scores
        if np.linalg.norm(new_scores - scores) < 1e-6:
            break
        scores = new_scores

    # Top-N sentences
    ranked_indices = scores.argsort()[::-1][:num_sentences]
    ranked_indices_sorted = sorted(ranked_indices)

    top_sentences = []
    for idx in ranked_indices_sorted:
        top_sentences.append({
            'sentence': sentences[idx],
            'score': round(float(scores[idx]), 4),
            'original_index': int(idx),
        })

    extractive_summary = ' '.join([sentences[i] for i in ranked_indices_sorted])

    return {
        'extractive_summary': extractive_summary,
        'top_sentences': top_sentences,
        'total_sentences': len(sentences),
        'tfidf_features': n_features,
    }


def abstractive_summary_gemini(text, max_length=500):
    """
    Abstractive Summarization bang Gemini AI.
    AI doc toan bo van ban va viet tom tat moi bang ngon ngu tu nhien.
    """
    truncated = text[:8000] if len(text) > 8000 else text

    prompt = f"""Ban la chuyen gia tom tat tai lieu hoc thuat. 
Hay tom tat noi dung sau day bang tieng Viet, ngan gon va co cau truc:

NGUYEN TAC:
1. Tom tat trong khoang {max_length} tu.
2. Giu nguyen cac thuat ngu chuyen nganh.
3. Chia thanh cac muc chinh voi icon emoji.
4. Liet ke cac y quan trong bang bullet points.
5. Cuoi cung, ghi "Tu khoa chinh: ..." liet ke 5-8 tu khoa.

NOI DUNG TAI LIEU:
{truncated}

TOM TAT:"""

    try:
        response = _summary_client.models.generate_content(
            model=_SUMMARY_MODEL, contents=prompt
        )
        return {
            'abstractive_summary': response.text.strip(),
            'model_used': _SUMMARY_MODEL,
            'input_length': len(text),
            'truncated': len(text) > 8000,
        }
    except Exception as e:
        print(f"Abstractive summary error: {e}")
        return {
            'abstractive_summary': f'Loi khi tom tat: {str(e)}',
            'model_used': _SUMMARY_MODEL,
            'input_length': len(text),
            'truncated': False,
        }


def summarize_document(text, num_extractive=5):
    """
    Hybrid Summarization: ket hop Extractive (ML) + Abstractive (AI).
    Returns full report cho frontend.
    """
    if not text or len(text.strip()) < 50:
        return {
            'status': 'error',
            'message': 'Noi dung tai lieu qua ngan de tom tat.',
        }

    # Buoc 1: Extractive - TF-IDF + TextRank
    extractive = textrank_extractive_summary(text, num_extractive)

    # Buoc 2: Abstractive - Gemini AI
    abstractive = abstractive_summary_gemini(text)

    # Thong ke van ban
    words = text.split()
    word_count = len(words)
    char_count = len(text)

    return {
        'status': 'success',
        'document_stats': {
            'word_count': word_count,
            'char_count': char_count,
            'sentence_count': extractive['total_sentences'],
            'tfidf_features': extractive['tfidf_features'],
        },
        'extractive': {
            'algorithm': 'TF-IDF + TextRank (Cosine Similarity Graph)',
            'summary': extractive['extractive_summary'],
            'top_sentences': extractive['top_sentences'],
            'num_selected': len(extractive['top_sentences']),
        },
        'abstractive': {
            'algorithm': f'Gemini AI ({abstractive["model_used"]})',
            'summary': abstractive['abstractive_summary'],
            'truncated_input': abstractive['truncated'],
        },
    }

