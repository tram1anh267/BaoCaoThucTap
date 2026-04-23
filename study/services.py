import os
import pytesseract
import numpy as np
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

# Cấu hình pytesseract cho tiếng Việt
_TESSERACT_LANG = 'vie+eng'

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash-lite")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
persist_directory = os.path.join(settings.BASE_DIR, "data", "chroma_db")

# SYSTEM PROMPT
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
            img = Image.open(file_path)
            return pytesseract.image_to_string(img, lang=_TESSERACT_LANG)
        elif ext == '.pdf':
            # Ưu tiên extract text trực tiếp
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""

            # Fallback sang OCR nếu PDF là scan hoặc slide image-based
            words_per_page = len(text.split()) / max(len(reader.pages), 1)
            if not text.strip() or words_per_page < 20:
                print(f"PDF text quá ít ({len(text.split())} từ / {len(reader.pages)} trang), fallback sang pytesseract OCR...")

                pages = convert_from_path(file_path, dpi=150, thread_count=4)

                from concurrent.futures import ThreadPoolExecutor

                def ocr_page(page_img):
                    return pytesseract.image_to_string(page_img, lang=_TESSERACT_LANG)

                with ThreadPoolExecutor(max_workers=max(1, os.cpu_count() // 2)) as executor:
                    results = list(executor.map(ocr_page, pages))

                ocr_text = '\n'.join(results)
                if ocr_text.strip():
                    text = ocr_text
            return text
    except Exception as e:
        print(f"Extraction Error: {e}")
    return ""
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
    # Gói Free Tier Google Gemini chặn khoảng 100-150 requests/phút. 
    # Ta tăng batch_size lên 100 (vì embed_documents của LangChain sẽ gộp các chunk này vào ít request hơn)
    # và giảm thời gian chờ xuống để tăng tốc độ.
    import time
    batch_size = 100
    total_indexed = 0

    print(f"Bắt đầu xử lý Text dài: Tổng cộng {len(chunks)} chunks.")
    
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_metadatas = [metadata] * len(batch_chunks)
        
        try:
            Chroma.from_texts(
                texts=batch_chunks,
                embedding=embeddings,
                persist_directory=persist_directory,
                metadatas=batch_metadatas
            )
            total_indexed += len(batch_chunks)
            print(f" Đã nhét thành công đợt {i//batch_size + 1} ({len(batch_chunks)} chunks).")
            
            # Chỉ nghỉ 2 giây thay vì 15 giây để tăng tốc, vẫn tránh spam API quá đà
            if i + batch_size < len(chunks):
                time.sleep(2)
                
        except Exception as e:
            print(f" Lỗi ở lô {i//batch_size + 1}: {e}")
            if "429" in str(e):
                print(f" Bị giới hạn tốc độ (429), nghỉ 20s...")
                time.sleep(20)
            continue

    print(f" Hoàn thành: Đã lưu {total_indexed}/{len(chunks)} chunks cho môn: {metadata.get('subject')}")


def retrieve_context(subject_name: str, user_id: str, query: str) -> tuple[str, list]:
    """Tìm kiếm các đoạn tài liệu liên quan từ ChromaDB. Trả về (context_string, chunks_list)."""
    if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
        return "", []
    
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
            chunks = [doc.page_content for doc in docs]
            context = "\n\n---\n\n".join(chunks)
            print(f"RAG found {len(docs)} chunks for '{subject_name}'")
            return context, chunks
    except Exception as e:
        print(f"Vector DB Error (skipped): {e}")
    return "", []


def retrieve_exam_context(subject_name: str, user_id: str, k: int = 8) -> tuple[str, list]:
    """Tìm kiếm các đoạn tài liệu ngẫu nhiên đa dạng (MMR) để làm ngữ cảnh sinh đề thi."""
    if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
        return "", []
    
    subject_lower = subject_name.lower()
    
    try:
        vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        
        # Dùng MMR để đa dạng hóa các đoạn văn bản được chọn
        docs = vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "filter": {"$and": [{"user_id": user_id}, {"subject": subject_lower}]},
                "k": k,
                "fetch_k": 20
            }
        ).invoke(f"Kiến thức cốt lõi và quan trọng nhất của môn {subject_name}")
        
        if not docs:
            docs = vector_db.as_retriever(
                search_kwargs={"filter": {"subject": subject_lower}, "k": k}
            ).invoke(f"Kiến thức cốt lõi {subject_name}")
            
        if docs:
            context_parts = []
            sources_used = set()
            for doc in docs:
                fname = doc.metadata.get('filename', 'Tài liệu')
                sources_used.add(fname)
                context_parts.append(f"--- Nguồn rút trích: {fname} ---\n{doc.page_content}")
            
            context = "\n\n".join(context_parts)
            print(f"Exam RAG found {len(docs)} chunks from {len(sources_used)} files for '{subject_name}'")
            return context, list(sources_used)
    except Exception as e:
        print(f"Exam Context DB Error (skipped): {e}")
    return "", []

# LLM GENERATION với Prompt Engineering nâng cao
def get_answer(subject: str, query: str, history: str = "", user_id: str = "") -> tuple[str, list]:
    """
    Sinh câu trả lời từ Gemini AI với:
    - RAG context từ tài liệu của user
    - Conversation history (memory)
    - Prompt Engineering nâng cao
    Trả về (answer_string, retrieved_chunks_list)
    """
    try:
        # Bước 1: RAG – Retrieve relevant context
        context, chunks = retrieve_context(subject, user_id, query)

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
                    return response.text, chunks
                else:
                    return "AI không phản hồi nội dung. Có thể do chính sách an toàn.", chunks
            except Exception as api_err:
                last_error = api_err
                if '503' in str(api_err) or 'UNAVAILABLE' in str(api_err):
                    print(f"Gemini 503, retry {attempt+1}/3...")
                    time.sleep(3)
                    continue
                raise

        return "Gemini AI đang quá tải. Vui lòng thử lại sau 30 giây.", chunks

    except Exception as e:
        print(f"General AI Error: {e}")
        return "AI đang bận rùiii, vui lòng thử lại sau ít giây nha tình yêu", []


def generate_mock_exam(subject: str) -> str:
    """Sinh đề thi ôn tập tự động cho môn học."""
    exam_prompt = f"""Hãy tạo một đề thi ôn tập cho môn {subject} gồm:
- 5 câu hỏi trắc nghiệm (A, B, C, D) với đáp án
- 2 câu hỏi tự luận ngắn với gợi ý trả lời

Định dạng đề thi rõ ràng, có đánh số câu hỏi."""
    answer, _ = get_answer(subject, exam_prompt)
    return answer


import json as _json
import re as _re

def parse_exam_from_text(raw_text: str, subject: str) -> list:
    """
    Preprocessing layer: Dùng LLM để chuẩn hóa đề thi từ text thô (bất kỳ định dạng nào)
    thành JSON chuẩn [{question, options:[A,B,C,D], correct_index:0-3, explanation}].
    
    Xử lý được các định dạng lộn xộn: đề thi scan PDF, đề thi gõ tay, v.v.
    """
    if not raw_text or not raw_text.strip():
        return []

    # Giới hạn text để tránh token quá dài
    truncated = raw_text[:12000]

    prompt = f"""Bạn là chuyên gia phân tích đề thi môn {subject}. Tôi cung cấp TEXT thô của một đề thi (có thể bị OCR lộn xộn, thiếu dấu, định dạng hỗn hợp).

NHIỆM VỤ: Trích xuất TẤT CẢ câu hỏi trắc nghiệm và xác định đáp án đúng.

NGUYÊN TẮc XỬ LÝ:
1. Mỗi câu hỏi phải có đúng 4 lựa chọn (A, B, C, D).
2. Về đáp án (correct_index):
   - Nếu đề THI CÓ đáp án sẵn → dùng đáp án đó (0=A, 1=B, 2=C, 3=D), đặt has_answer=true.
   - Nếu đề THI KHÔNG CÓ đáp án → dùng KIẾN THỨC CHUYÊN MÔN của bạn về môn {subject} để suy luận đáp án ĐÚNG NHẤT, đặt has_answer=false.
3. Về giải thích (explanation):
   - Nếu có trong đề → dùng.
   - Nếu không có → VIẾT GIẢI THÍCH NGẪN (1-2 câu) tại sao đáp án đó đúng dựa trên kiến thức môn {subject}.
4. Bỏ qua: tiêu đề đề thi, thông tin trường, tên giáo viên, câu tự luận.

TEXT ĐỀ THI:
\"\"\"
{truncated}
\"\"\"

Trả về CHỈ một mảng JSON hợp lệ (không có markdown, không có ```json):
[
  {{
    "question": "Nội dung câu hỏi?",
    "options": ["Đáp án A", "Đáp án B", "Đáp án C", "Đáp án D"],
    "correct_index": 0,
    "explanation": "Giải thích tại sao đáp án này đúng.",
    "has_answer": true
  }}
]
Chỉ trả về JSON array, không thêm bất kỳ văn bản nào khác."""

    try:
        import time
        from google.genai import types
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=8000,
                        temperature=0.1,  # Low temperature for precise extraction
                    )
                )
                raw = response.text.strip()
                # Strip markdown fences
                raw = _re.sub(r'^```[a-z]*\n?', '', raw, flags=_re.MULTILINE)
                raw = _re.sub(r'```$', '', raw.strip())

                questions = _json.loads(raw.strip())

                validated = []
                for q in questions:
                    if all(k in q for k in ['question', 'options', 'correct_index']):
                        if isinstance(q['options'], list) and len(q['options']) == 4:
                            if isinstance(q['correct_index'], int) and 0 <= q['correct_index'] <= 3:
                                validated.append({
                                    'question': str(q['question']),
                                    'options': [str(o) for o in q['options']],
                                    'correct_index': q['correct_index'],
                                    'explanation': str(q.get('explanation', '')),
                                    'has_answer': bool(q.get('has_answer', True)),
                                })
                print(f"[parse_exam] Extracted {len(validated)} questions from uploaded exam")
                return validated
            except Exception as e:
                if '503' in str(e) or 'UNAVAILABLE' in str(e):
                    print(f"Gemini 503, retry {attempt+1}/3...")
                    time.sleep(3)
                    continue
                print(f"[parse_exam] Error: {e}")
                return []
    except Exception as e:
        print(f"[parse_exam] Fatal error: {e}")
    return []


def generate_exam_json(subject: str, num_questions: int = 10, user_id: str = "") -> list:
    """
    Sinh đề thi trắc nghiệm dạng JSON để chấm điểm tự động.
    Trả về list các dict: {question, options:[A,B,C,D], correct_index:0-3, explanation, source}
    """
    context, sources = retrieve_exam_context(subject, user_id, k=8)

    has_doc_context = bool(context and context.strip())
    
    if has_doc_context:
        sources_str = ", ".join(sources)
        context_block = f"\nDưới đây là các trích đoạn từ tài liệu sinh viên đã tải lên (trích xuất từ các file: {sources_str}):\n{context}\n"
        source_label = f"📚 Dựa trên tài liệu môn {subject} đã upload"
        
        prompt_instruction = f"""YÊU CẦU NGHIÊM NGẶT:
Hãy ƯU TIÊN tối đa việc sử dụng các trích đoạn tài liệu trên để tạo ra {num_questions} câu hỏi.
Nếu nội dung tài liệu dồi dào, BẮT BUỘC toàn bộ câu hỏi phải lấy từ tài liệu.
Trong trường hợp sinh viên yêu cầu số lượng câu lớn ({num_questions} câu), nhưng nội dung từ tài liệu không cugn cấp đủ ý để làm đủ số lượng trên, bạn mới ĐƯỢC PHÉP mix thêm bằng kiến thức tổng hợp bên ngoài (Gemini AI) của bạn về môn học này để bổ sung cho đủ {num_questions} câu.

Trong json trả về, có trường "question_source" để chỉ rõ nguồn gốc câu hỏi (Ví dụ: "Lấy từ file: bai_giang_tuần_1.pdf" hoặc "Kiến thức AI tổng hợp")."""

    else:
        context_block = ""
        source_label = "🤖 Kiến thức AI tổng hợp (chưa có tài liệu upload)"
        prompt_instruction = f"""Sinh viên CHƯA upload tài liệu nào.
Hãy sử dụng kiến thức AI sâu rộng của bạn về môn {subject} để tạo đủ {num_questions} câu hỏi.
Trong json trả về, ghi trường "question_source" là "Kiến thức AI tổng hợp" cho tất cả các câu."""

    prompt = f"""Bạn là giảng viên đại học tâm huyết ra đề thi trắc nghiệm môn {subject}.
{context_block}
{prompt_instruction}

Trả về CHỈ một mảng JSON hợp lệ (không có markdown, không có ```json), theo đúng định dạng:
[
  {{
    "question": "Nội dung câu hỏi?",
    "options": ["Đáp án A", "Đáp án B", "Đáp án C", "Đáp án D"],
    "correct_index": 0,
    "explanation": "Giải thích tại sao đáp án lại đúng.",
    "question_source": "Trích rõ nguồn câu hỏi (tên file hoặc AI)"
  }}
]
- correct_index luôn là 0, 1, 2, hoặc 3 (tương ứng A, B, C, D).
- Câu hỏi phải rõ ràng, bám sát chuyên môn học thuật.
- Chỉ trả về chuỗi JSON mảng."""

    try:
        import time
        from google.genai import types
        
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME, 
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=8000,
                        temperature=0.7,
                    )
                )
                raw = response.text.strip()

                # Strip markdown code fences nếu có
                raw = _re.sub(r'^```[a-z]*\n?', '', raw, flags=_re.MULTILINE)
                raw = _re.sub(r'```$', '', raw.strip())

                questions = _json.loads(raw.strip())

                # Validate cấu trúc + gắn source metadata
                validated = []
                for q in questions:
                    if all(k in q for k in ['question', 'options', 'correct_index', 'explanation']):
                        if isinstance(q['options'], list) and len(q['options']) == 4:
                            if isinstance(q['correct_index'], int) and 0 <= q['correct_index'] <= 3:
                                q['source'] = source_label
                                q['has_doc_context'] = has_doc_context
                                # Nhúng nguồn trực tiếp vào lời giải thích để UI có thể show ra
                                if 'question_source' in q and q['question_source']:
                                    q['explanation'] = f"[Nguồn: {q['question_source']}] " + q['explanation']
                                validated.append(q)
                return validated
            except Exception as api_err:
                if '503' in str(api_err) or 'UNAVAILABLE' in str(api_err):
                    print(f"Gemini 503 in generate_exam_json, retry {attempt+1}/3...")
                    time.sleep(3)
                    continue
                print(f"Exam JSON API error: {api_err}")
                return []
        
        return []

    except Exception as e:
        print(f"Exam JSON generation error: {e}")
        return []


def generate_weakness_exam_json(subject: str, topic_name: str, wrong_questions: list, num_questions: int = 5) -> list:
    """
    Sinh đề thi tập trung vào điểm yếu cụ thể (cluster).
    """
    questions_list = "\n".join([f"- {q}" for q in wrong_questions[:5]])  # Lấy tối đa 5 câu đại diện
    
    prompt = f"""Bạn là giáo viên ra đề thi trắc nghiệm môn {subject}.
Học sinh đang yếu và hay làm sai ở dạng bài/chủ đề: "{topic_name}".
Dưới đây là một số câu hỏi mà học sinh ĐÃ LÀM SAI trước đó thuộc chủ đề này:
{questions_list}

Hãy XÂY DỰNG MỚI {num_questions} câu hỏi trắc nghiệm TƯƠNG TỰ (cùng cấu trúc, mức độ khó, hoặc mở rộng nhẹ) để học sinh luyện tập khắc phục điểm yếu này.
Trả về CHỈ một mảng JSON hợp lệ (không có markdown, không có ```json), theo định dạng:
[
  {{
    "question": "Nội dung câu hỏi?",
    "options": ["Đáp án A", "Đáp án B", "Đáp án C", "Đáp án D"],
    "correct_index": 0,
    "explanation": "Giải thích chi tiết tại sao đáp án đúng."
  }}
]"""

    try:
        import time
        from google.genai import types
        
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME, 
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=4000,
                        temperature=0.7,
                    )
                )
                raw = response.text.strip()
                raw = _re.sub(r'^```[a-z]*\n?', '', raw, flags=_re.MULTILINE)
                raw = _re.sub(r'```$', '', raw.strip())
                questions = _json.loads(raw.strip())
                
                validated = []
                for q in questions:
                    if all(k in q for k in ['question', 'options', 'correct_index', 'explanation']):
                        if isinstance(q['options'], list) and len(q['options']) == 4:
                            if isinstance(q['correct_index'], int) and 0 <= q['correct_index'] <= 3:
                                q['source'] = f"🎯 Ôn tập điểm yếu: {topic_name}"
                                validated.append(q)
                return validated
            except Exception as api_err:
                if '503' in str(api_err) or 'UNAVAILABLE' in str(api_err):
                    print(f"Gemini 503 in generate_weakness_exam, retry {attempt+1}/3...")
                    time.sleep(3)
                    continue
                print(f"Weakness Exam API error: {api_err}")
                return []
        
        return []

    except Exception as e:
        print(f"Weakness Exam JSON gen error: {e}")
        return []
