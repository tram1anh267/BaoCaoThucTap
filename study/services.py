import os
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from django.conf import settings
from google import genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

# ─────────────────────────────────────────────────────────
# Khởi tạo AI Client & Embeddings
# ─────────────────────────────────────────────────────────
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash-lite")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
persist_directory = os.path.join(settings.BASE_DIR, "data", "chroma_db")

# ─────────────────────────────────────────────────────────
# SYSTEM PROMPT – Prompt Engineering nâng cao
# ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Bạn là EduFlow AI – trợ lý học tập cho sinh viên.

QUY TẮC QUAN TRỌNG:
- Trả lời NGẮN GỌN, tối đa 3-5 câu cho mỗi ý.
- Dùng bullet points ngắn, không viết đoạn văn dài.
- Ưu tiên tài liệu tham khảo nếu có.
- Trả lời bằng tiếng Việt.
- Không lặp lại câu hỏi trong câu trả lời.
"""


# ─────────────────────────────────────────────────────────
# OCR & TEXT EXTRACTION
# ─────────────────────────────────────────────────────────

def extract_text(file_path):
    """Trích xuất text từ file ảnh hoặc PDF."""
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            return pytesseract.image_to_string(Image.open(file_path), lang='vie+eng')
        elif ext == '.pdf':
            # Ưu tiên extract text trực tiếp
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            
            # Fallback sang OCR nếu PDF là scan
            if not text.strip():
                print("PDF text empty, falling back to OCR...")
                pages = convert_from_path(file_path)
                for page in pages:
                    text += pytesseract.image_to_string(page, lang='vie+eng')
            return text
    except Exception as e:
        print(f"Extraction Error: {e}")
    return ""


def classify_text(text):
    """Phân loại tài liệu dựa trên nội dung."""
    text_lower = text.lower()
    if any(kw in text_lower for kw in ['slide', 'trình chiếu', 'powerpoint', 'bài giảng', 'lecture']):
        return "Slide"
    elif any(kw in text_lower for kw in ['giáo trình', 'textbook', 'sách giáo khoa', 'chương trình đào tạo']):
        return "Textbook"
    elif any(kw in text_lower for kw in ['đề thi', 'kỳ thi', 'exam', 'năm học', 'học kỳ', 'đề kiểm tra']):
        return "PastExam"
    elif any(kw in text_lower for kw in ['ví dụ', 'example', 'ex:', 'minh họa']):
        return "Examples"
    elif any(kw in text_lower for kw in ['bài tập', 'luyện tập', 'exercise', 'practice', 'đề bài']):
        return "Exercises"
    elif any(kw in text_lower for kw in ['lý thuyết', 'khái niệm', 'định nghĩa', 'theory', 'chương']):
        return "Theory"
    return "Other"


# ─────────────────────────────────────────────────────────
# RAG – INDEX & RETRIEVE
# ─────────────────────────────────────────────────────────

def index_document(text, metadata):
    """Chia nhỏ văn bản và lưu vào ChromaDB vector store."""
    if not text.strip():
        print("Empty text, skipping indexing.")
        return
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    
    if not chunks:
        return

    # Lowercase subject để đồng bộ với retrieval
    if 'subject' in metadata:
        metadata['subject'] = metadata['subject'].lower()

    Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        metadatas=[metadata] * len(chunks)
    )
    print(f"Indexed {len(chunks)} chunks for subject: {metadata.get('subject')}")


def retrieve_context(subject_name: str, user_id: str, query: str) -> str:
    """Tìm kiếm các đoạn tài liệu liên quan từ ChromaDB."""
    if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
        return ""
    
    subject_lower = subject_name.lower()
    
    try:
        vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        
        # 1. Filter theo user + subject (dùng $and cho multi-filter)
        docs = vector_db.as_retriever(
            search_kwargs={
                "filter": {"$and": [{"user_id": user_id}, {"subject": subject_lower}]},
                "k": 2
            }
        ).invoke(query)
        
        # 2. Fallback: chỉ filter subject
        if not docs:
            docs = vector_db.as_retriever(
                search_kwargs={"filter": {"subject": subject_lower}, "k": 2}
            ).invoke(query)
        
        # 3. Fallback cuối: không filter, tìm theo semantic similarity
        if not docs:
            docs = vector_db.as_retriever(
                search_kwargs={"k": 2}
            ).invoke(query)
        
        if docs:
            context = "\n\n---\n\n".join([doc.page_content for doc in docs])
            print(f"RAG found {len(docs)} chunks for '{subject_name}'")
            return context
    except Exception as e:
        print(f"Vector DB Error (skipped): {e}")
    return ""


# ─────────────────────────────────────────────────────────
# LLM GENERATION với Prompt Engineering nâng cao
# ─────────────────────────────────────────────────────────

def get_answer(subject: str, query: str, history: str = "", user_id: str = "") -> str:
    """
    Sinh câu trả lời từ Gemini AI với:
    - RAG context từ tài liệu của user
    - Conversation history (memory)
    - Prompt Engineering nâng cao
    """
    try:
        # Bước 1: RAG – Retrieve relevant context
        context = retrieve_context(subject, user_id, query)

        # === LOG: Hiện chi tiết trên terminal ===
        print("\n" + "="*60)
        print(f"🔍 CÂU HỎI: {query}")
        print(f"📚 MÔN: {subject} | USER: {user_id}")
        if context:
            print(f"✅ RAG CONTEXT ({len(context)} chars):")
            print(f"   {context[:200]}...")
        else:
            print("❌ Không tìm thấy context → dùng kiến thức chung")
        print("="*60 + "\n")

        # Bước 2: Xây dựng prompt (Prompt Engineering)
        prompt_parts = [SYSTEM_PROMPT, "\n\n"]

        # Context từ tài liệu (nếu có)
        if context:
            prompt_parts.append(f"📚 TÀI LIỆU CỦA SINH VIÊN (đã upload cho môn {subject}):\n{context}\n\n")
            prompt_parts.append("ƯU TIÊN trả lời dựa trên tài liệu trên. Khi trích dẫn, ghi tự nhiên như 'Theo slide bài giảng,...' hoặc 'Trong tài liệu môn học,...'. Nếu tài liệu không đủ, bổ sung từ kiến thức chung và ghi rõ nguồn tham khảo uy tín.\n\n")
        else:
            prompt_parts.append(f"[Lưu ý: Chưa có tài liệu cho môn {subject}. Trả lời dựa trên kiến thức chung.]\n\n")

        # Conversation history (Memory)
        if history:
            prompt_parts.append(f"💬 LỊCH SỬ HỘI THOẠI GẦN ĐÂY:\n{history}\n\n")

        # Câu hỏi hiện tại
        prompt_parts.append(f"❓ CÂU HỎI: {query}")

        full_prompt = "".join(prompt_parts)

        # Bước 3: Gọi Gemini API (có retry khi 503)
        import time
        from google.genai import types
        
        last_error = None
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=500,
                        temperature=0.7,
                    )
                )
                # Xử lý response
                if hasattr(response, 'text') and response.text:
                    return response.text
                else:
                    return "AI không phản hồi nội dung. Có thể do chính sách an toàn."
            except Exception as api_err:
                last_error = api_err
                if '503' in str(api_err) or 'UNAVAILABLE' in str(api_err):
                    print(f"Gemini 503, retry {attempt+1}/3...")
                    time.sleep(3)
                    continue
                raise

        return "⏳ Gemini AI đang quá tải. Vui lòng thử lại sau 30 giây."

    except Exception as e:
        print(f"General AI Error: {e}")
        return "⏳ AI đang bận, vui lòng thử lại sau ít giây."


def generate_mock_exam(subject: str) -> str:
    """Sinh đề thi ôn tập tự động cho môn học."""
    exam_prompt = f"""Hãy tạo một đề thi ôn tập cho môn {subject} gồm:
- 5 câu hỏi trắc nghiệm (A, B, C, D) với đáp án
- 2 câu hỏi tự luận ngắn với gợi ý trả lời

Định dạng đề thi rõ ràng, có đánh số câu hỏi."""
    return get_answer(subject, exam_prompt)


import json as _json
import re as _re

def generate_exam_json(subject: str, num_questions: int = 10, user_id: str = "") -> list:
    """
    Sinh đề thi trắc nghiệm dạng JSON để chấm điểm tự động.
    Trả về list các dict: {question, options:[A,B,C,D], correct_index:0-3, explanation}
    """
    context = retrieve_context(subject, user_id, f"câu hỏi trắc nghiệm {subject}")

    context_block = ""
    if context:
        context_block = f"\nDựa vào tài liệu sau:\n{context}\n"

    prompt = f"""Bạn là giáo viên ra đề thi trắc nghiệm môn {subject}.
{context_block}
Hãy tạo đúng {num_questions} câu hỏi trắc nghiệm.
Trả về CHỈ một mảng JSON hợp lệ (không có markdown, không có ```json), theo đúng định dạng:
[
  {{
    "question": "Nội dung câu hỏi?",
    "options": ["Đáp án A", "Đáp án B", "Đáp án C", "Đáp án D"],
    "correct_index": 0,
    "explanation": "Giải thích tại sao đáp án đúng."
  }}
]
- correct_index là số nguyên 0, 1, 2, hoặc 3 (tương ứng A, B, C, D).
- Câu hỏi phải rõ ràng, phù hợp trình độ đại học.
- Chỉ trả về JSON, không có văn bản thêm vào."""

    try:
        response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        raw = response.text.strip()

        # Strip markdown code fences nếu có
        raw = _re.sub(r'^```[a-z]*\n?', '', raw, flags=_re.MULTILINE)
        raw = _re.sub(r'```$', '', raw.strip())

        questions = _json.loads(raw.strip())

        # Validate cấu trúc
        validated = []
        for q in questions:
            if all(k in q for k in ['question', 'options', 'correct_index', 'explanation']):
                if isinstance(q['options'], list) and len(q['options']) == 4:
                    if isinstance(q['correct_index'], int) and 0 <= q['correct_index'] <= 3:
                        validated.append(q)
        return validated
    except Exception as e:
        print(f"Exam JSON generation error: {e}")
        return []

