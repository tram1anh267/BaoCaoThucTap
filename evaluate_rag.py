import json
import csv
import time
import os
from openai import OpenAI


GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

client = OpenAI(
    api_key=GITHUB_TOKEN,
    base_url="https://models.inference.ai.azure.com",
    timeout=60.0 # Timeout sau 60 giây
)
MODEL_NAME = "gpt-4o-mini"

EVAL_METRICS = {
    "Answer_Relevancy": {
        "definition": "Đo lường độ trực tiếp và đúng trọng tâm của câu trả lời. Câu trả lời có giải quyết trực tiếp câu hỏi mà không bị lan man hay thừa thãi không?",
        "evaluation_steps": [
            "Tính điểm dựa trên tỷ lệ thông tin hữu ích: Score = (Độ dài phần thông tin trực tiếp giải quyết câu hỏi) / (Tổng độ dài câu trả lời).",
            "1.0 = Hoàn toàn đi thẳng vào vấn đề, không có thông tin thừa, không né tránh.",
            "0.5 = Trả lời đúng trọng tâm nhưng kèm theo nhiều thông tin lạc đề hoặc rườm rà.",
            "0.0 = Hoàn toàn lạc đề hoặc chỉ trả lời chung chung, khuôn mẫu không hữu ích."
        ],
        "scoring_rule": "Chấm một số thực (float) từ 0.0 đến 1.0 phản ánh tỷ lệ độ liên quan của câu trả lời.",
        "inputs_needed": ["Question", "Chat-bot-answer"]
    },
    "Faithfulness": {
        "definition": "Đo lường tỷ lệ các mệnh đề (claims) trong câu trả lời có thể được suy ra trực tiếp từ tài liệu (Context) được cung cấp. Đánh giá mức độ 'ảo giác' (hallucination).",
        "evaluation_steps": [
            "Cách tính: Score = (Số mệnh đề suy ra được từ Context) / (Tổng số mệnh đề trong câu trả lời).",
            "1. Yêu cầu AI tách câu trả lời thành các ý/mệnh đề (claims) độc lập.",
            "2. Đối chiếu từng mệnh đề với Context.",
            "1.0 = Không có ảo giác (100% mệnh đề đều đúng theo Context).",
            "0.5 = Độ trung thực đạt một nửa (ví dụ: 2 ý lấy từ Context, 2 ý tự bịa).",
            "0.0 = Dựa hoàn toàn vào kiến thức ngoài hoặc bịa đặt (0% lấy từ Context)."
        ],
        "scoring_rule": "Chấm một số thực (float) từ 0.0 đến 1.0 phản ánh tỷ lệ mệnh đề được hỗ trợ bởi Context.",
        "inputs_needed": ["Chat-bot-answer", "Contexts"]
    },
    "Context_Precision": {
        "definition": "Đánh giá chất lượng xếp hạng của hệ thống truy xuất (Retriever). Chunk chứa đáp án thực sự (Ground Truth) có được xếp ở thứ hạng cao nhất (Rank 1) hay không?",
        "evaluation_steps": [
            "Tính điểm dựa trên công thức MRR (Mean Reciprocal Rank): Score = 1 / Vị_trí_của_chunk_chứa_đáp_án_đầu_tiên.",
            "1.0 = Chunk chứa Ground Truth đứng ở vị trí số 1 (Rank 1).",
            "0.5 = Chunk chứa Ground Truth đứng ở vị trí số 2 (Rank 2).",
            "0.33 = Chunk chứa Ground Truth đứng ở vị trí số 3 (Rank 3).",
            "0.0 = Không có chunk nào trong Context chứa thông tin của Ground Truth."
        ],
        "scoring_rule": "Chấm một số thực (float) từ 0.0 đến 1.0 (ví dụ 1.0, 0.5, 0.33, 0.25, 0.0) dựa trên thứ hạng (rank) của chunk đúng.",
        "inputs_needed": ["Question", "Contexts", "Ground Truth"]
    },
    "Context_Recall": {
        "definition": "Đo lường khả năng thu thập đầy đủ thông tin của Retriever. Context lấy ra có chứa được bao nhiêu phần trăm thông tin cần thiết để tạo thành một đáp án hoàn chỉnh (Ground Truth)?",
        "evaluation_steps": [
            "Cách tính: Score = (Số lượng ý/facts trong Ground Truth có mặt trong Contexts) / (Tổng số ý/facts trong Ground Truth).",
            "1. Tách Ground Truth thành các facts/ý chính.",
            "2. Kiểm tra xem Contexts bao phủ được bao nhiêu facts đó.",
            "1.0 = Contexts cung cấp đúng và đủ 100% thông tin để trả lời được như Ground Truth.",
            "0.5 = Contexts chỉ chứa một nửa (50%) thông tin của Ground Truth.",
            "0.0 = Contexts không chứa bất kỳ thông tin nào liên quan đến Ground Truth."
        ],
        "scoring_rule": "Chấm một số thực (float) từ 0.0 đến 1.0 phản ánh tỷ lệ facts của Ground Truth được tìm thấy trong Contexts.",
        "inputs_needed": ["Ground Truth", "Contexts"]
    }
}

def evaluate_single_metric(metric_name, row):
    metric_info = EVAL_METRICS[metric_name]
    evaluation_data = {key: row.get(key, "") for key in metric_info['inputs_needed']}
    
    steps_str = "\n    ".join([f"- {step}" for step in metric_info['evaluation_steps']])

    prompt = f"""Bạn là Giám khảo AI nghiêm túc chuyên đánh giá hệ thống RAG theo thang điểm từ 0.0 đến 1.0 dựa trên phương pháp định lượng nghiêm ngặt của G-Eval.
    Tiêu chí Đánh giá:
    - Tên Metric: {metric_name}
    - Định nghĩa: {metric_info['definition']}
    
    HƯỚNG DẪN ĐO LƯỜNG VÀ TÍNH ĐIỂM:
    {steps_str}
    
    Quy tắc chấm điểm: {metric_info['scoring_rule']}
    
    DỮ LIỆU ĐÁNH GIÁ:
    {json.dumps(evaluation_data, ensure_ascii=False, indent=2)}
    
    YÊU CẦU:
    1. Phân tích từng bước theo Hướng dẫn đo lường ở trên (Ví dụ: đếm xem có bao nhiêu ý/mệnh đề, cái nào đúng/sai, chunk nằm ở vị trí thứ mấy...).
    2. Chốt điểm (score) là MỘT SỐ THỰC (FLOAT) TỪ 0.0 ĐẾN 1.0 (ví dụ: 0.0, 0.25, 0.33, 0.5, 0.75, 1.0) dựa trên phân tích toán học tỷ lệ phần trăm hoặc công thức MRR. Tuyệt đối KHÔNG gò bó điểm bắt buộc phải là 1.0 hay 0.0 nếu câu trả lời nằm ở mức giữa.
    
    CHỈ TRẢ VỀ ĐÚNG 1 JSON NÀY (KHÔNG DƯ KÝ TỰ NÀO KHÁC):
    {{
        "reason": "Giải thích chi tiết các bước tính toán theo hướng dẫn...",
        "score": <điểm_từ_0.0_đến_1.0>
    }}
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Bạn phải luôn trả về dữ liệu định dạng JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" },
            temperature=0.0
        )
        result = json.loads(response.choices[0].message.content)
        score = result.get('score', 0.0)
        # Đảm bảo score là số thực trong khoảng 0.0-1.0
        try:
            score = max(0.0, min(1.0, float(score)))
        except:
            score = 0.0
        return score, result.get('reason', '')
    except Exception as e:
        print(f"Lỗi: {e}")
        return 0, str(e)

def run_evaluation():
    input_file = 'RAG_Full_Eval_Data.csv'
    output_file = 'RAG_Final_Scores.csv'
    
    print("===========================================")
    print("🤖 HỆ THỐNG ĐÁNH GIÁ RAG AUTO 4 METRICS 🤖")
    print("===========================================")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)
        # Strip tên cột để tránh lỗi dấu cách thừa trong CSV header (ví dụ: "Ground Truth " → "Ground Truth")
        rows = [{k.strip(): v for k, v in row.items()} for row in raw_rows]
        fieldnames = [col.strip() for col in reader.fieldnames]
        
    for metric_name in EVAL_METRICS.keys():
        col_score = f"{metric_name}_score"
        col_reason = f"{metric_name}_reason"
        if col_score not in fieldnames:
            fieldnames.append(col_score)
        if col_reason not in fieldnames:
            fieldnames.append(col_reason)

    print(f"🚀 Bắt đầu chấm điểm {len(rows)} mẫu câu theo thang điểm 0.0-1.0...\n")
    
    score_sums = {metric: 0.0 for metric in EVAL_METRICS.keys()}
    valid_counts = {metric: 0 for metric in EVAL_METRICS.keys()}
    
    try:
        for i, row in enumerate(rows):
            print(f"\n--- 📝 Câu {i+1}/{len(rows)}: {row.get('Question', '')[:80]} ---")
            for metric in EVAL_METRICS.keys():
                score, reason = evaluate_single_metric(metric, row)
                row[f"{metric}_score"] = score
                row[f"{metric}_reason"] = reason
                
                print(f"[{metric}] Điểm: {score:.2f}/1.0")
                print(f"   > Lý do: {reason[:200]}...") # In ngắn gọn reason
                
                score_sums[metric] += score
                valid_counts[metric] += 1
                
                time.sleep(2) # Tránh Rate Limit Github (Tăng lên 2s để an toàn hơn)
    except Exception as e:
        print(f"\n⚠️ Dừng đột ngột: {e}")

    # Lưu file
    with open(output_file, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
    # IN KẾT QUẢ TỔNG KẾT
    print("\n📊 ========================================================")
    print("             BẢNG TỔNG KẾT RAG THANG ĐIỂM 1.0")
    print("===========================================================")
    
    for metric, info in EVAL_METRICS.items():
        if valid_counts[metric] > 0:
            avg_score = score_sums[metric] / valid_counts[metric]
            percentage = avg_score * 100
            
            print(f"\n📌 {metric.upper()}:")
            print(f"   - Điểm trung bình: {avg_score:.3f}/1.0")
            print(f"   - Hiệu suất (Accuracy %): {percentage:.1f}%")
            
            # Đánh giá chất lượng
            stat = "Rất tốt" if avg_score >= 0.8 else "Khá" if avg_score >= 0.65 else "Đạt yêu cầu" if avg_score >= 0.5 else "Cần cải thiện"
            print(f"   - Xếp loại: {stat}")
        else:
            print(f"\n📌 {metric.upper()}: Không có dữ liệu")
            
    print("\n===========================================================")
    print(f"📁 Dữ liệu lưu vết lý do chấm điểm (Reason) nằm tại: {output_file}")

if __name__ == "__main__":
    run_evaluation()
