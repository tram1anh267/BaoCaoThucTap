from django.db import models
from django.contrib.auth.models import User


class Subject(models.Model):
    ICON_CHOICES = [
        ('�', 'Đại cương'),
        ('🎓', 'Chuyên Ngành'),
        ('⭐', 'Tự chọn'),
    ]
    name = models.CharField(max_length=100)
    icon = models.CharField(max_length=10, default='📚')
    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name='subjects')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('name', 'owner')
        ordering = ['name']

    def __str__(self):
        return f"{self.icon} {self.name} ({self.owner.username})"


class Document(models.Model):
    CATEGORY_CHOICES = [
        ('Theory', 'Lý thuyết'),
        ('Slide', 'Slide bài giảng'),
        ('Textbook', 'Giáo trình'),
        ('Examples', 'Ví dụ minh họa'),
        ('Exercises', 'Bài tập'),
        ('PastExam', 'Đề thi các năm'),
        ('Other', 'Khác'),
    ]
    subject = models.ForeignKey(Subject, on_delete=models.CASCADE, related_name='documents')
    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name='documents')
    filename = models.CharField(max_length=255)
    category = models.CharField(max_length=50, choices=CATEGORY_CHOICES, default='Other')
    file_path = models.CharField(max_length=500)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-uploaded_at']

    def __str__(self):
        return f"{self.filename} [{self.category}] - {self.subject.name}"


class ChatMessage(models.Model):
    ROLE_CHOICES = [
        ('user', 'User'),
        ('ai', 'AI'),
    ]
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='chat_messages')
    subject = models.ForeignKey(Subject, on_delete=models.SET_NULL, null=True, blank=True, related_name='chat_messages')
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    retrieved_context = models.JSONField(null=True, blank=True)  # Lưu list chunks RAG đã dùng
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"[{self.role.upper()}] {self.user.username}: {self.content[:60]}"


class ExamResult(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='exam_results')
    subject = models.ForeignKey(Subject, on_delete=models.CASCADE, related_name='exam_results')
    exam_content = models.TextField()
    score = models.FloatField(null=True, blank=True)
    taken_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-taken_at']

    def __str__(self):
        return f"{self.user.username} - {self.subject.name} - {self.taken_at.strftime('%Y-%m-%d')}"


class ExamSession(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='exam_sessions')
    subject = models.ForeignKey(Subject, on_delete=models.CASCADE, related_name='exam_sessions')
    # JSON: list of {question, options: [A,B,C,D], correct_answer, explanation}
    questions_json = models.TextField()
    # JSON: list of user answers (index 0-3 or None)
    answers_json = models.TextField(default='[]')
    score = models.FloatField(null=True, blank=True)          # 0-10
    total_questions = models.IntegerField(default=0)
    correct_count = models.IntegerField(default=0)
    is_submitted = models.BooleanField(default=False)
    started_at = models.DateTimeField(auto_now_add=True)
    submitted_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-started_at']

    def __str__(self):
        return f"{self.user.username} - {self.subject.name} - {self.score}/10"


class UploadedExam(models.Model):
    """Lưu đề thi đã upload và được parse thành JSON chuẩn bởi LLM."""
    document = models.OneToOneField(Document, on_delete=models.CASCADE, related_name='uploaded_exam')
    subject = models.ForeignKey(Subject, on_delete=models.CASCADE, related_name='uploaded_exams')
    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name='uploaded_exams')
    # Tên hiển thị (lấy từ filename, không có extension)
    display_name = models.CharField(max_length=255)
    # JSON chuẩn: list of {question, options:[A,B,C,D], correct_index:0-3, explanation}
    questions_json = models.TextField()
    total_questions = models.IntegerField(default=0)
    # Trạng thái parse: pending / done / failed
    parse_status = models.CharField(max_length=20, default='pending')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.display_name} [{self.subject.name}] - {self.total_questions} câu"

