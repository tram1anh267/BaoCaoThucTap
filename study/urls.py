from django.urls import path
from . import views

urlpatterns = [
    # Auth
    path('register/', views.register_view, name='register'),
    path('login/',    views.login_view,    name='login'),
    path('logout/',   views.logout_view,   name='logout'),

    # Dashboard
    path('', views.dashboard, name='dashboard'),

    # Subject management
    path('api/subject/add/',            views.add_subject,    name='add_subject'),
    path('api/subject/<int:subject_id>/delete/', views.delete_subject, name='delete_subject'),

    # Documents
    path('api/upload/',                          views.upload_file,    name='upload_file'),
    path('api/subject/<int:subject_id>/docs/',   views.documents_list, name='documents_list'),

    # Chat
    path('api/chat/',                            views.chat,          name='chat'),
    path('api/subject/<int:subject_id>/history/', views.chat_history,  name='chat_history'),

    # Exam (old format)
    path('api/exam/<int:subject_id>/', views.get_exam, name='get_exam'),

    # Exam Session – Thi thử & Chấm điểm
    path('api/exam-session/start/<int:subject_id>/', views.start_exam_session, name='start_exam_session'),
    path('api/exam-session/submit/<int:session_id>/', views.submit_exam_session, name='submit_exam_session'),
    path('api/exam-session/history/<int:subject_id>/', views.exam_history, name='exam_history'),

    # ML – Phân tích điểm yếu
    path('api/weakness/', views.weakness_analysis, name='weakness_all'),
    path('api/weakness/<int:subject_id>/', views.weakness_analysis, name='weakness_subject'),

    # ML – Tóm tắt tài liệu
    path('api/summarize/<int:doc_id>/', views.summarize_doc_view, name='summarize_doc'),

    # Oral Exam – Vấn đáp AI + Voice
    path('api/oral/question/<int:subject_id>/', views.oral_get_question, name='oral_question'),
    path('api/oral/grade/', views.oral_grade, name='oral_grade'),
]
