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


def _join_vietnamese_compounds(text):
    import re
    compound_words = [
        "máy tính", "cơ sở dữ liệu", "cơ sở", "dữ liệu", "trí tuệ nhân tạo",
        "trí tuệ", "nhân tạo", "mạng máy tính", "hệ điều hành", "điều hành",
        "phần mềm", "phần cứng", "lập trình", "ngôn ngữ lập trình", "ngôn ngữ",
        "thuật toán", "truy vấn", "truy xuất", "xử lý", "tính toán",
        "ứng dụng", "hệ thống", "dịch vụ", "nền tảng", "giao diện",
        "mã nguồn", "mã hóa", "giải mã", "bảo mật", "xác thực",
        "tập tin", "thư mục", "đám mây", "điện toán", "điện toán đám mây",
        "công cụ", "công nghệ", "kỹ thuật", "học giám sát", "thị giác máy tính",
        "học không giám sát", "học tăng cường", "machine learning",
        "khả năng", "lợi ích", "ưu điểm", "nhược điểm", "đặc trưng",
        "tác vụ", "nhiệm vụ", "chức năng", "mục tiêu", "kết quả",
        "phương pháp", "quy trình", "mô hình", "kiến trúc", "cấu trúc",
        "quản trị", "hệ quản trị", "phân tích", "tổng hợp", "đánh giá",
        "hiệu suất", "độ chính xác", "tối ưu", "mở rộng", "triển khai",
        "song song", "phân tán", "phân cụm", "phân loại", "dự đoán",
        "học máy", "huấn luyện", "tập dữ liệu",
        "chịu lỗi", "khả năng chịu lỗi", "khả năng mở rộng",
    ]
    # Sắp xếp theo độ dài giảm dần để ưu tiên cụm dài hơn trước
    compound_words.sort(key=len, reverse=True)

    for cw in compound_words:
        joined = cw.replace(" ", "_")
        pattern = re.compile(re.escape(cw), re.IGNORECASE)
        text = pattern.sub(joined, text)
    return text


def _deduplicate_keywords(keywords):
    """
    Loại bỏ các unigram trùng lặp khi bigram/compound tương ứng đã tồn tại trong danh sách.
    Ví dụ: ["khả_năng", "khả", "năng", "map", "hadoop"] → ["khả_năng", "map", "hadoop"]
    """
    bigrams = [kw for kw in keywords if '_' in kw or ' ' in kw]
    filtered = []
    for kw in keywords:
        if '_' not in kw and ' ' not in kw:
            is_part_of_bigram = False
            for bg in bigrams:
                parts = bg.replace('_', ' ').split()
                if kw.lower() in [p.lower() for p in parts]:
                    is_part_of_bigram = True
                    break
            if is_part_of_bigram:
                continue
        filtered.append(kw)
    return filtered


def cluster_wrong_questions(wrong_questions, n_clusters=None):
    if len(wrong_questions) < 3:
        return {
            'clusters': [],
            'total_wrong': len(wrong_questions),
            'n_clusters': 0,
            'silhouette_score': 0,
            'message': 'Cần ít nhất 3 câu sai để phân tích. Hãy thi thêm!'
        }

    # Bước 1: TF-IDF Vectorization
    import os
    github_stopwords = []
    stopwords_path = os.path.join(os.path.dirname(__file__), 'vietnamese-stopwords.txt')
    if os.path.exists(stopwords_path):
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            github_stopwords = [line.strip() for line in f if line.strip()]

    custom_stop_words = list(ENGLISH_STOP_WORDS) + github_stopwords + [
        "câu", "hỏi", "đáp", "án", "đúng", "sai", "chọn", "phương", "nào", "sau", "đây", 
        "ý", "phát", "biểu", "mệnh", "đề", "trắc", "nghiệm", "dưới", "nhất", "điền", "trống",
        "thế", "người", "ta", "việc", "quan", "nhận", "định", "hiểu", "nghĩa", "đặc", "điểm", "đoạn",
        "correct", "incorrect", "answer", "question", "options", "option", "choose", 
        "following", "statement", "statements", "true", "false", "blank", "best", 
        "which", "what", "where", "when", "how", "why", "who",
        "pdf", "file", "nguồn", "chapter", "lấy", "theo", "tài", "liệu",
        "nêu", "rõ", "dựa", "trên", "trích", "midterm", "practice", "exam", "lecture"]
    import re
    def _clean_source_metadata(text):
        text = re.sub(r'\[Nguồn:\s*[^\]]*\]', '', text)
        text = re.sub(r'\b\w+\.(pdf|docx|doc|txt|pptx|xlsx)\b', '', text, flags=re.IGNORECASE)
        return text

    texts = [
        _join_vietnamese_compounds(
            _clean_source_metadata(q['question'] + ' ' + q.get('explanation', ''))
        )
        for q in wrong_questions
    ]

    vectorizer = TfidfVectorizer(
        max_features=200,
        stop_words=custom_stop_words,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        token_pattern=r'(?u)\b[\w_]+\b',  
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    # Bước 2: Chọn số cluster tối ưu bằng Silhouette Score
    if n_clusters is None:
        max_k = min(len(wrong_questions) - 1, 5)  
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
        raw_keywords = [feature_names[i] for i in top_keyword_indices if cluster_tfidf[i] > 0]

        # Lọc trùng: loại unigram khi compound/bigram đã tồn tại
        top_keywords = _deduplicate_keywords(raw_keywords)[:5]

        # Thay dấu gạch dưới về dấu cách cho hiển thị
        top_keywords = [kw.replace('_', ' ') for kw in top_keywords]

        clusters.append({
            'cluster_id': cluster_id,
            'topic_keywords': top_keywords,
            'topic_name': '',
            'questions': cluster_questions,
            'count': len(cluster_questions),
            'percentage': round(len(cluster_questions) / len(wrong_questions) * 100, 1),
        })

    # Đặt tên cụm bằng Top 3 từ khóa TF-IDF
    for c in clusters:
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
# Text Summarization – Abstractive (Gemini)
# ─────────────────────────────────────────────────────────

import re
import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
_summary_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
_SUMMARY_MODEL = os.getenv("MODEL_NAME", "gemini-2.0-flash-lite")


def abstractive_summary_gemini(text, max_length=500):
    """
    Abstractive Summarization bằng Gemini AI.
    AI đọc toàn bộ văn bản và viết tóm tắt mới bằng ngôn ngữ tự nhiên.
    """
    truncated = text[:12000] if len(text) > 12000 else text

    prompt = f"""Bạn là chuyên gia tóm tắt tài liệu học thuật. 
Hay tóm tắt nội dung sau đây bằng tiếng Việt, ngắn gọn và có cấu trúc:

NGUYÊN TẮC:
1. Tóm tắt trong khoảng {max_length} từ.
2. Giữ nguyên các thuật ngữ chuyên ngành.
3. Chia thành các mục chính với icon emoji.
4. Liệt kê các ý quan trọng bằng bullet points.
5. Cuối cùng, ghi "Từ khóa chính: ..." liệt kê 5-8 từ khóa.

NỘI DUNG TÀI LIỆU:
{truncated}

TÓM TẮT:"""

    try:
        response = _summary_client.models.generate_content(
            model=_SUMMARY_MODEL, contents=prompt
        )
        return {
            'summary': response.text.strip(),
            'model': _SUMMARY_MODEL,
        }
    except Exception as e:
        print(f"Summarize error: {e}")
        return {
            'summary': f'Lỗi khi tóm tắt: {str(e)}',
            'model': _SUMMARY_MODEL,
        }


def summarize_document(text):
    """
    Tóm tắt tài liệu bằng Gemini AI.
    """
    if not text or len(text.strip()) < 50:
        return {
            'status': 'error',
            'message': 'Nội dung tài liệu quá ngắn để tóm tắt.',
        }

    # Gọi Gemini AI để tóm tắt
    result = abstractive_summary_gemini(text)

    # Trả về kết quả tinh gọn
    return {
        'status': 'success',
        'document_stats': {
            'word_count': len(text.split()),
            'char_count': len(text),
        },
        'summary': result['summary'],
        'model': result['model'],
    }

