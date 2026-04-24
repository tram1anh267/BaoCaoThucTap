from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
from django.conf import settings
from .services import extract_text, index_document, get_answer, generate_mock_exam, generate_exam_json, parse_exam_from_text, generate_weakness_exam_json
from .models import Subject, Document, ChatMessage, ExamResult, ExamSession, UploadedExam
from .ml_services import generate_weakness_report, summarize_document
from .forms import RegisterForm, LoginForm, SubjectForm
import os
import shutil
import json
from django.utils import timezone

@csrf_exempt
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


@csrf_exempt
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
            # Cho phép user chọn thư mục lưu trữ thủ công
            user_category = request.POST.get('category', '').strip()
            if user_category in ['Theory', 'Slide', 'Textbook', 'Examples', 'Exercises', 'PastExam', 'Other']:
                category = user_category
            else:
                category = 'Other'

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
            doc = Document.objects.create(
                subject=subject,
                owner=request.user,
                filename=filename,
                category=category,
                file_path=final_path,
            )

            # Nếu là đề thi → auto-parse thành JSON chuẩn (preprocessing layer)
            if category == 'PastExam':
                try:
                    questions = parse_exam_from_text(text, subject.name)
                    display_name = os.path.splitext(filename)[0]  # bỏ extension
                    UploadedExam.objects.create(
                        document=doc,
                        subject=subject,
                        owner=request.user,
                        display_name=display_name,
                        questions_json=json.dumps(questions, ensure_ascii=False),
                        total_questions=len(questions),
                        parse_status='done' if questions else 'failed',
                    )
                    print(f"[upload] Parsed PastExam: {display_name} → {len(questions)} câu")
                except Exception as ex:
                    print(f"[upload] PastExam parse error: {ex}")

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

            # Save AI response (kèm theo các chunks RAG đã dùng)
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


@login_required
def delete_document(request, doc_id):
    if request.method == 'POST':
        doc = get_object_or_404(Document, id=doc_id, owner=request.user)
        try:
            # Xóa file trong ổ cứng
            if doc.file_path and os.path.exists(doc.file_path):
                os.remove(doc.file_path)
            # Xóa trong cơ sở dữ liệu (tự động xóa các UploadedExam liên quan)
            doc.delete()
            return JsonResponse({'status': 'success', 'message': 'Đã xóa tài liệu.'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': f'Lỗi khi xóa: {str(e)}'})
    return JsonResponse({'status': 'error', 'message': 'Dữ liệu không hợp lệ.'})


# ─────────────────────────────────────────────────────────
# EXAM SESSION – Thi thử & Chấm điểm
# ─────────────────────────────────────────────────────────

@login_required
def start_exam_session(request, subject_id):
    """
    Tạo phiên thi thử mới.
    
    Modes:
    - mode=new (default): AI sinh đề mới từ tài liệu.
    - mode=retake&exam_id=<id>: Thi lại y chang đề đã upload (UploadedExam).
    """
    subject = get_object_or_404(Subject, id=subject_id, owner=request.user)
    mode = request.GET.get('mode', 'new')

    try:
        if mode == 'retake':
            # ── Chế độ thi lại đề đã upload ──────────────────────────────
            exam_id = request.GET.get('exam_id')
            if not exam_id:
                return JsonResponse({'status': 'error', 'message': 'Thiếu exam_id để thi lại.'})

            uploaded_exam = get_object_or_404(
                UploadedExam,
                id=exam_id,
                subject=subject,
                owner=request.user,
                parse_status='done',
            )
            questions = json.loads(uploaded_exam.questions_json)
            if not questions:
                return JsonResponse({'status': 'error', 'message': 'Đề thi này chưa được parse hoặc không có câu hỏi.'})

            session = ExamSession.objects.create(
                user=request.user,
                subject=subject,
                questions_json=uploaded_exam.questions_json,
                answers_json='[]',
                total_questions=len(questions),
            )
            safe_questions = [
                {'question': q['question'], 'options': q['options']}
                for q in questions
            ]
            return JsonResponse({
                'status': 'success',
                'mode': 'retake',
                'exam_name': uploaded_exam.display_name,
                'exam_source': f'📜 Thi lại đề đã upload: {uploaded_exam.display_name}',
                'session_id': session.id,
                'questions': safe_questions,
                'total': len(questions),
            })

        else:
            # ── Chế độ sinh đề mới từ AI ──────────────────────────────────
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
            safe_questions = [
                {'question': q['question'], 'options': q['options']}
                for q in questions
            ]
            # Lấy source từ câu hỏi đầu tiên
            exam_source = questions[0].get('source', '🤖 AI sinh đề mới') if questions else ''
            return JsonResponse({
                'status': 'success',
                'mode': 'new',
                'exam_source': exam_source,
                'session_id': session.id,
                'questions': safe_questions,
                'total': len(questions),
            })

    except Exception as e:
        print(f"Start exam error: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)})


@login_required
def submit_exam_session(request, session_id):
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


@login_required
def uploaded_exams_list(request, subject_id):
    """Danh sách đề thi đã upload & parse cho môn học."""
    subject = get_object_or_404(Subject, id=subject_id, owner=request.user)
    exams = UploadedExam.objects.filter(subject=subject, owner=request.user).order_by('-created_at')
    data = [{
        'id': e.id,
        'name': e.display_name,            # JS reads: e.name
        'num_questions': e.total_questions, # JS reads: e.num_questions
        'status': e.parse_status,           # JS reads: e.status
        'created_at': e.created_at.strftime('%d/%m/%Y %H:%M'),
    } for e in exams]
    return JsonResponse({'exams': data})   # JS reads: data.exams


# ─────────────────────────────────────────────────────────
# ML – Phân tích điểm yếu
# ─────────────────────────────────────────────────────────

@login_required
def weakness_analysis(request, subject_id=None):
    try:
        subject = None
        if subject_id:
            subject = get_object_or_404(Subject, id=subject_id, owner=request.user)
        report = generate_weakness_report(request.user, subject)
        return JsonResponse(report)
    except Exception as e:
        print(f"Weakness analysis error: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)})


@login_required
@csrf_exempt
def practice_weakness(request, subject_id):
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Invalid request method.'})

    subject = get_object_or_404(Subject, id=subject_id, owner=request.user)
    try:
        data = json.loads(request.body)
        topic_name = data.get('topic_name', 'Chủ đề hỗn hợp')
        wrong_questions = data.get('questions', [])
        
        if not wrong_questions:
            return JsonResponse({'status': 'error', 'message': 'Không có dữ liệu câu hỏi lỗi để luyện tập.'})

        questions = generate_weakness_exam_json(subject.name, topic_name, wrong_questions, num_questions=5)
        
        if not questions:
            return JsonResponse({'status': 'error', 'message': 'Không thể sinh câu hỏi luyện tập. Thử lại sau.'})

        session = ExamSession.objects.create(
            user=request.user,
            subject=subject,
            questions_json=json.dumps(questions, ensure_ascii=False),
            answers_json='[]',
            total_questions=len(questions),
        )
        safe_questions = [
            {'question': q['question'], 'options': q['options']}
            for q in questions
        ]
        
        return JsonResponse({
            'status': 'success',
            'mode': 'practice',
            'exam_source': f'🎯 Ôn tập điểm yếu: {topic_name}',
            'session_id': session.id,
            'questions': safe_questions,
            'total': len(questions),
        })

    except Exception as e:
        print(f"Practice weakness error: {e}")
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
# DASHBOARD STATS – Dữ liệu cho Charts
# ─────────────────────────────────────────────────────────

@login_required
def dashboard_stats(request):
    """
    Trả về tất cả data cần thiết để render Dashboard Charts:
    - activity_days: số buổi học/ngày (14 ngày gần nhất)
    - score_trend: điểm thi theo thời gian, chia theo môn
    - docs_by_category: số tài liệu theo loại
    - subject_avg_scores: điểm trung bình mỗi môn
    - overall: tổng hợp nhanh
    """
    from datetime import timedelta
    from django.utils import timezone
    from django.db.models import Avg, Count
    from django.db.models.functions import TruncDate

    user = request.user
    now = timezone.now()
    days_30 = now - timedelta(days=30)

    # ── 1. Activity heatmap: số phiên thi + chat mỗi ngày (30 ngày) ──
    exam_by_day = (
        ExamSession.objects
        .filter(user=user, started_at__gte=days_30, is_submitted=True)
        .annotate(day=TruncDate('started_at'))
        .values('day')
        .annotate(cnt=Count('id'))
    )
    chat_by_day = (
        ChatMessage.objects
        .filter(user=user, role='user', created_at__gte=days_30)
        .annotate(day=TruncDate('created_at'))
        .values('day')
        .annotate(cnt=Count('id'))
    )
    activity_map = {}
    for r in exam_by_day:
        key = r['day'].strftime('%Y-%m-%d')
        activity_map[key] = activity_map.get(key, 0) + r['cnt']
    for r in chat_by_day:
        key = r['day'].strftime('%Y-%m-%d')
        activity_map[key] = activity_map.get(key, 0) + r['cnt']

    activity_days = []
    for i in range(30):
        day = (now - timedelta(days=29-i)).date()
        key = day.strftime('%Y-%m-%d')
        activity_days.append({
            'date': key,
            'label': day.strftime('%d/%m'),
            'count': activity_map.get(key, 0),
        })

    # ── 2. Score trend: điểm theo ngày (30 ngày), group by môn ──
    sessions = (
        ExamSession.objects
        .filter(user=user, is_submitted=True, submitted_at__gte=days_30, score__isnull=False)
        .select_related('subject')
        .order_by('submitted_at')
        .values('subject__name', 'subject__icon', 'score', 'submitted_at')
    )
    score_by_subject = {}
    for s in sessions:
        name = f"{s['subject__icon']} {s['subject__name']}"
        if name not in score_by_subject:
            score_by_subject[name] = []
        score_by_subject[name].append({
            'date': s['submitted_at'].strftime('%d/%m'),
            'score': round(s['score'], 1),
        })

    # ── 3. Documents by category ──
    doc_cats = (
        Document.objects
        .filter(owner=user)
        .values('category')
        .annotate(cnt=Count('id'))
        .order_by('-cnt')
    )
    cat_labels_vi = {
        'Theory': 'Lý thuyết', 'Slide': 'Slide',
        'Textbook': 'Giáo trình', 'Examples': 'Ví dụ',
        'Exercises': 'Bài tập', 'PastExam': 'Đề thi', 'Other': 'Khác',
    }
    docs_by_category = [
        {'category': cat_labels_vi.get(d['category'], d['category']), 'count': d['cnt']}
        for d in doc_cats
    ]

    # ── 4. Subject avg scores ──
    subject_avgs = (
        ExamSession.objects
        .filter(user=user, is_submitted=True, score__isnull=False)
        .values('subject__name', 'subject__icon')
        .annotate(avg_score=Avg('score'), total=Count('id'))
        .order_by('-avg_score')
    )
    subject_avg_scores = [
        {
            'subject': f"{s['subject__icon']} {s['subject__name']}",
            'avg': round(s['avg_score'], 1),
            'total': s['total'],
        }
        for s in subject_avgs
    ]

    # ── 5. Overall stats ──
    total_sessions = ExamSession.objects.filter(user=user, is_submitted=True).count()
    avg_score_all = ExamSession.objects.filter(
        user=user, is_submitted=True, score__isnull=False
    ).aggregate(avg=Avg('score'))['avg'] or 0
    best_score = ExamSession.objects.filter(
        user=user, is_submitted=True, score__isnull=False
    ).order_by('-score').values_list('score', flat=True).first() or 0
    active_days = len([d for d in activity_days if d['count'] > 0])

    return JsonResponse({
        'activity_days': activity_days,
        'score_trend': score_by_subject,
        'docs_by_category': docs_by_category,
        'subject_avg_scores': subject_avg_scores,
        'overall': {
            'total_sessions': total_sessions,
            'avg_score': round(avg_score_all, 1),
            'best_score': round(best_score, 1),
            'active_days_30': active_days,
        },
    })
