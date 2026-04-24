import os
import django
import csv

# Kết nối với cài đặt của dự án Django hiện tại
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'eduflow.settings')
django.setup()

from study.services import get_answer
from django.contrib.auth.models import User

def generate_rag_evaluation_data():
    input_file = '/Users/hoangtramanh/Documents/BAO CAO THUC TAP/input_eval.csv'
    output_file = 'RAG_Full_Eval_Data.csv'
    
    print(f"Đang đọc file đầu vào: {input_file}...")
    
    # Đọc dữ liệu từ file CSV cũ
    with open(input_file, mode='r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)
        # Thêm 2 cột mới
        fieldnames = reader.fieldnames + ['Chat-bot-answer', 'Contexts']
        
    # Lấy đại 1 user (để backend không bị lỗi phân quyền)
    user = User.objects.first()
    admin_user_id = str(user.id) if user else "1"

    # Xử lý từng câu hỏi trong file
    for i, row in enumerate(rows):
        subject_name = row.get('Subject', '')
        query = row.get('Question', '')
        
        print(f"[{i + 1}/{len(rows)}] Đang hỏi Chatbot: {query}")
        try:
            # GỌI HÀM GET_ANSWER (BẮT CẢ CÂU TRẢ LỜI LẪN CONTEXT)
            answer, retrieved_chunks = get_answer(subject_name, query, "", admin_user_id)
            
            # Gắn vào data
            row['Chat-bot-answer'] = answer
            row['Contexts'] = str(retrieved_chunks)
            print(f"   -> Đã lấy được Text & Context!")
            
        except Exception as e:
            print(f"   -> Lỗi bỏ qua: {e}")
            row['Chat-bot-answer'] = f"LỖI: {e}"
            row['Contexts'] = "LỖI"
            
    # Lưu ra file mới hoàn chỉnh
    print(f"\n=======================")
    print(f"Đang xuất file hoàn chỉnh ra {output_file}...")
    with open(output_file, mode='w', encoding='utf-8-sig', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"THÀNH CÔNG! Giờ bạn có thể mở file {output_file} lên xem!")

if __name__ == '__main__':
    generate_rag_evaluation_data()
