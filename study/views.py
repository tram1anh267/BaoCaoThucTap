from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
from django.conf import settings
from .services import extract_text, classify_text, index_document, get_answer, generate_mock_exam, generate_exam_json
from .models import Subject, Document, ChatMessage, ExamResult, ExamSession
from .ml_services import generate_weakness_report, summarize_document, generate_oral_question, grade_oral_answer
from .forms import RegisterForm, LoginForm, SubjectForm
import os
import shutil
import json
from django.utils import timezone


# ─────────────────────────────────────────────────────────
# AUTH VIEWS
# ─────────────────────────────────────────────────────────

def register_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, f'Chào mừng {user.first_name}! Tài khoản đã được tạo.')
            return redirect('dashboard')
    else:
        form = RegisterForm()
    return render(request, 'study/register.html', {'form': form})


def login_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    if request.method == 'POST':
        form = LoginForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            next_url = request.GET.get('next', 'dashboard')
            return redirect(next_url)
        else:
            messages.error(request, 'Tên đăng nhập hoặc mật khẩu không đúng.')
    else:
        form = LoginForm()
    return render(request, 'study/login.html', {'form': form})


def logout_view(request):
    logout(request)
    return redirect('login')


# ─────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────

@login_required
def dashboard(request):
    user = request.user
    subjects = Subject.objects.filter(owner=user)
    subject_form = SubjectForm()

    # Stats
    total_docs  = Document.objects.filter(owner=user).count()
    total_chats = ChatMessage.objects.filter(user=user, role='user').count()
    total_exams = ExamResult.objects.filter(user=user).count()

    # Pre-compute display values to avoid complex template filters
    full_name    = user.get_full_name() or user.username
    display_name = user.first_name or user.username
    fn = (user.first_name or user.username)[:1].upper()
    ln = (user.last_name or '')[:1].upper()
    initials = (fn + ln) if ln else fn

    return render(request, 'study/dashboard.html', {
        'subjects':      subjects,
        'subject_form':  subject_form,
        'total_docs':    total_docs,
        'total_chats':   total_chats,
        'total_exams':   total_exams,
        'full_name':     full_name,
        'display_name':  display_name,
        'initials':      initials,
    })


# ─────────────────────────────────────────────────────────
# SUBJECT MANAGEMENT
# ─────────────────────────────────────────────────────────

@login_required
def add_subject(request):
    if request.method == 'POST':
        form = SubjectForm(request.POST)
        if form.is_valid():
            subject = form.save(commit=False)
            subject.owner = request.user
            subject.save()
            return JsonResponse({'status': 'success', 'id': subject.id, 'name': subject.name, 'icon': subject.icon})
        return JsonResponse({'status': 'error', 'message': str(form.errors)})
    return JsonResponse({'status': 'error', 'message': 'Invalid method'})


@login_required
def delete_subject(request, subject_id):
    subject = get_object_or_404(Subject, id=subject_id, owner=request.user)
    subject.delete()
    return JsonResponse({'status': 'success'})


# ─────────────────────────────────────────────────────────
# DOCUMENT UPLOAD
# ─────────────────────────────────────────────────────────

@login_required
def upload_file(request):
    if request.method == 'POST' and request.FILES.get('file'):
        try:
            file = request.FILES['file']
            subject_id = request.POST.get('subject_id')
            subject = get_object_or_404(Subject, id=subject_id, owner=request.user)

            fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'temp'))
            filename = fs.save(file.name, file)
            file_path = fs.path(filename)

            print(f"Processing file: {file_path}")
            text = extract_text(file_path)
            category = classify_text(text)

            # Cho phép user override category (nếu họ chọn thủ công)
            user_category = request.POST.get('category', '').strip()
            if user_category in ['Theory', 'Slide', 'Textbook', 'Examples', 'Exercises', 'PastExam', 'Other']:
                category = user_category

            # Move to final folder
            final_dir = os.path.join(settings.MEDIA_ROOT, 'subjects', str(request.user.id), subject.name, category)
            os.makedirs(final_dir, exist_ok=True)
            final_path = os.path.join(final_dir, filename)
            shutil.move(file_path, final_path)

            # Index with user-scoped metadata
            index_document(text, {
                "subject": subject.name,
                "subject_id": str(subject.id),
                "user_id": str(request.user.id),
                "category": category,
                "filename": filename
            })

            # Save to DB
            Document.objects.create(
                subject=subject,
                owner=request.user,
                filename=filename,
                category=category,
                file_path=final_path,
            )

            return JsonResponse({
                'status': 'success',
                'category': category,
                'message': f'File đã được lưu vào thư mục {category}'
            })
        except Exception as e:
            print(f"Upload error: {str(e)}")
            return JsonResponse({'status': 'error', 'message': str(e)})
    return JsonResponse({'status': 'error', 'message': 'Invalid request'})


# ─────────────────────────────────────────────────────────
# CHAT
# ─────────────────────────────────────────────────────────

@login_required
@csrf_exempt
def chat(request):
    if request.method == 'POST':
        try:
            query = request.POST.get('query', '').strip()
            subject_id = request.POST.get('subject_id')

            if not query:
                return JsonResponse({'answer': 'Vui lòng nhập câu hỏi.'})

            subject = get_object_or_404(Subject, id=subject_id, owner=request.user)

            # Save user message
            ChatMessage.objects.create(
                user=request.user,
                subject=subject,
                role='user',
                content=query
            )

            # Get recent conversation history (last 6 messages for context)
            recent_msgs = ChatMessage.objects.filter(
                user=request.user,
                subject=subject
            ).order_by('-created_at')[:6]
            
            history = list(reversed(recent_msgs))
            history_text = "\n".join([f"{'Người học' if m.role == 'user' else 'AI'}: {m.content}" for m in history[:-1]])

            answer = get_answer(subject.name, query, history_text, str(request.user.id))

            # Save AI response
            ChatMessage.objects.create(
                user=request.user,
                subject=subject,
                role='ai',
                content=answer
            )

            return JsonResponse({'answer': answer})
        except Exception as e:
            print(f"Chat error: {str(e)}")
            return JsonResponse({'answer': f'Lỗi hệ thống AI: {str(e)}'})
    return JsonResponse({'status': 'error', 'message': 'Invalid request'})


@login_required
def chat_history(request, subject_id):
    subject = get_object_or_404(Subject, id=subject_id, owner=request.user)
    messages_qs = ChatMessage.objects.filter(user=request.user, subject=subject).order_by('created_at')
    history = [{'role': m.role, 'content': m.content, 'time': m.created_at.strftime('%H:%M')} for m in messages_qs]
    return JsonResponse({'history': history})


# ─────────────────────────────────────────────────────────
# EXAM
# ─────────────────────────────────────────────────────────

@login_required
def get_exam(request, subject_id):
    try:
        subject = get_object_or_404(Subject, id=subject_id, owner=request.user)
        print(f"Generating exam for {subject.name}")
        exam = generate_mock_exam(subject.name)

        # Save exam result
        ExamResult.objects.create(
            user=request.user,
            subject=subject,
            exam_content=exam,
        )
        return JsonResponse({'exam': exam, 'subject': subject.name})
    except Exception as e:
        print(f"Exam gen error: {str(e)}")
        return JsonResponse({'status': 'error', 'exam': f'Lỗi sinh đề: {str(e)}'})


@login_required
def documents_list(request, subject_id):
    subject = get_object_or_404(Subject, id=subject_id, owner=request.user)
    docs = Document.objects.filter(subject=subject, owner=request.user)
    data = [{'id': d.id, 'filename': d.filename, 'category': d.category, 'uploaded_at': d.uploaded_at.strftime('%Y-%m-%d %H:%M')} for d in docs]
    return JsonResponse({'documents': data})


# ─────────────────────────────────────────────────────────
# EXAM SESSION – Thi thử & Chấm điểm
# ─────────────────────────────────────────────────────────

@login_required
def start_exam_session(request, subject_id):
    """Tạo phiên thi thử mới, sinh đề JSON từ AI."""
    subject = get_object_or_404(Subject, id=subject_id, owner=request.user)
    try:
        num_q = int(request.GET.get('num', 10))
        num_q = max(5, min(num_q, 20))  # giới hạn 5-20 câu
        questions = generate_exam_json(subject.name, num_q, str(request.user.id))
        if not questions:
            return JsonResponse({'status': 'error', 'message': 'AI không sinh được đề thi. Thử lại sau.'})

        session = ExamSession.objects.create(
            user=request.user,
            subject=subject,
            questions_json=json.dumps(questions, ensure_ascii=False),
            answers_json='[]',
            total_questions=len(questions),
        )
        # Trả về câu hỏi KHÔNG kèm đáp án cho client
        safe_questions = [
            {'question': q['question'], 'options': q['options']}
            for q in questions
        ]
        return JsonResponse({
            'status': 'success',
            'session_id': session.id,
            'questions': safe_questions,
            'total': len(questions),
        })
    except Exception as e:
        print(f"Start exam error: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)})


@login_required
def submit_exam_session(request, session_id):
    """Nhận bài làm, chấm điểm, lưu kết quả chi tiết."""
    session = get_object_or_404(ExamSession, id=session_id, user=request.user)
    if session.is_submitted:
        return JsonResponse({'status': 'error', 'message': 'Bài thi đã được nộp rồi.'})

    try:
        body = json.loads(request.body)
        user_answers = body.get('answers', [])  # list of int (correct_index) or None
    except Exception:
        return JsonResponse({'status': 'error', 'message': 'Dữ liệu không hợp lệ.'})

    questions = json.loads(session.questions_json)
    total = len(questions)
    correct_count = 0
    result_detail = []

    for i, q in enumerate(questions):
        user_ans = user_answers[i] if i < len(user_answers) else None
        correct_idx = q['correct_index']
        is_correct = (user_ans == correct_idx)
        if is_correct:
            correct_count += 1
        result_detail.append({
            'question': q['question'],
            'options': q['options'],
            'correct_index': correct_idx,
            'user_answer': user_ans,
            'is_correct': is_correct,
            'explanation': q.get('explanation', ''),
        })

    score = round((correct_count / total) * 10, 2) if total > 0 else 0

    session.answers_json = json.dumps(user_answers)
    session.correct_count = correct_count
    session.score = score
    session.is_submitted = True
    session.submitted_at = timezone.now()
    session.save()

    return JsonResponse({
        'status': 'success',
        'score': score,
        'correct_count': correct_count,
        'total': total,
        'result_detail': result_detail,
    })


@login_required
def exam_history(request, subject_id):
    """Lịch sử các lần thi thử của user cho 1 môn."""
    subject = get_object_or_404(Subject, id=subject_id, owner=request.user)
    sessions = ExamSession.objects.filter(user=request.user, subject=subject, is_submitted=True)[:10]
    data = [{
        'id': s.id,
        'score': s.score,
        'correct_count': s.correct_count,
        'total_questions': s.total_questions,
        'submitted_at': s.submitted_at.strftime('%d/%m/%Y %H:%M') if s.submitted_at else '',
    } for s in sessions]
    return JsonResponse({'history': data})


# ─────────────────────────────────────────────────────────
# ML – Phân tích điểm yếu
# ─────────────────────────────────────────────────────────

@login_required
def weakness_analysis(request, subject_id=None):
    """Phân tích điểm yếu bằng ML (TF-IDF + K-Means)."""
    try:
        subject = None
        if subject_id:
            subject = get_object_or_404(Subject, id=subject_id, owner=request.user)
        report = generate_weakness_report(request.user, subject)
        return JsonResponse(report)
    except Exception as e:
        print(f"Weakness analysis error: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)})


# ─────────────────────────────────────────────────────────
# Text Summarization
# ─────────────────────────────────────────────────────────

@login_required
def summarize_doc_view(request, doc_id):
    """Tóm tắt tài liệu bằng Extractive (TF-IDF+TextRank) + Abstractive (Gemini)."""
    try:
        doc = get_object_or_404(Document, id=doc_id, subject__owner=request.user)
        # Đọc text từ file gốc
        file_path = doc.file_path
        if not os.path.exists(file_path):
            file_path = os.path.join(settings.MEDIA_ROOT, doc.file_path)
        text = extract_text(file_path)
        if not text or not text.strip():
            return JsonResponse({'status': 'error', 'message': 'Tài liệu không có nội dung text.'})

        result = summarize_document(text)
        result['document_name'] = doc.filename
        result['document_category'] = doc.category
        return JsonResponse(result)
    except Exception as e:
        print(f"Summarize error: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)})


# ─────────────────────────────────────────────────────────
# Oral Exam – Vấn đáp AI + Voice
# ─────────────────────────────────────────────────────────

@login_required
def oral_get_question(request, subject_id):
    """Sinh câu hỏi vấn đáp cho môn học."""
    subject = get_object_or_404(Subject, id=subject_id, owner=request.user)
    try:
        # Lấy context từ tài liệu nếu có
        docs = Document.objects.filter(subject=subject, owner=request.user)[:3]
        context = ""
        for doc in docs:
            try:
                fp = doc.file_path
                if not os.path.exists(fp):
                    fp = os.path.join(settings.MEDIA_ROOT, doc.file_path)
                context += extract_text(fp)[:1000] + "\n"
            except Exception:
                pass

        data = generate_oral_question(subject.name, context)
        if not data:
            return JsonResponse({'status': 'error', 'message': 'AI không sinh được câu hỏi.'})

        return JsonResponse({
            'status': 'success',
            'question': data['question'],
            'reference_answer': data['reference_answer'],
            'keywords': data.get('keywords', []),
        })
    except Exception as e:
        print(f"Oral question error: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)})


@login_required
def oral_grade(request):
    """Chấm điểm câu trả lời vấn đáp bằng ML + AI."""
    try:
        body = json.loads(request.body)
        question = body.get('question', '')
        student_answer = body.get('student_answer', '')
        reference_answer = body.get('reference_answer', '')
        keywords = body.get('keywords', [])

        if not student_answer.strip():
            return JsonResponse({'status': 'error', 'message': 'Chưa có câu trả lời.'})

        result = grade_oral_answer(question, student_answer, reference_answer, keywords)
        result['status'] = 'success'
        return JsonResponse(result)
    except Exception as e:
        print(f"Oral grade error: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)})
