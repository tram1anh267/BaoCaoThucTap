import os, sys, jiwer
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "eduflow.settings")
import django; django.setup()
from study.services import extract_text

PREVIEW_MODE = False  # True = xem text, False = tính WER/CER
PREVIEW_FILE = "media/subjects/1/Phân tích dữ liệu bằng Python/Exercises/download (8)(1).jpg"

test_cases = [
    {
        "label": "PDF text thuần (Slide)",
        "file": "/Users/hoangtramanh/Documents/BAO CAO THUC TAP/Project/media/subjects/1/AI for business/Slide/midterm-practice.pdf",
        "truth": """TRÍ TUỆ NHÂN TẠO TRONG KINH DOANH
+ Học Máy (Machine learning) , Các loại học máy
a) PP Học máy được đặc trưng bởi khả năng học từ dữ liệu mà không cần lập trình rõ ràng. Học máy thường được dùng cho việc ra quyết định.
b) 3 loại học máy:
- Học giám sát: Phát triển mô hình dự đoán dựa trên dữ liệu đầu vào và đầu ra
Học từ dữ liệu đã được dán nhãn rõ ràng đầu ra và đầu vào EX: phân loại email (chính/xã hội/spam/...)
+ Phân loại
+ Hồi quy
- Học không giám sát: Nhóm và giải thích các quan sát chỉ dựa trên dữ liệu đầu vào
Không có một bộ đầu ra cố định, phải học từ đầu vào. Mục tiêu là phân nhóm và nhận dạng những đặc điểm quan trọng.
EX: LDA: VD như đối với chủ đề chính trị thì sẽ thường xuất hiện những từ như Obama, speech,...
+ Phát hiện bất thường
+ Phân cụm
- Học tăng cường: Thu thập dữ liệu mới bằng cách thực hiện các hành động và từ các
phản hồi trong quá trình thực hiện
Để thuật toán học bằng cách thử nhiều hành động/chiến lược và chọn ra cái tốt nhất i/ Exploration: going to new restaurant
ii/ Exploitation: going to favorite restaurant
+ Bandit Algorithm: multi-armed bandit về sự đánh đổi trong quá trình thăm dò và khai thác (exploration–exploitation trade-offs) của thuật toán để thông qua đó giảm thiểu chi phí thăm dò và tăng hiệu quả khai thác được
+ Q-learning
c) Lợi ích của ML: Dữ liệu và sự tính toán có thể thay thế cho chuyên gia
- Lợi ích: sự ổn định, tốc độ, làm việc rất tốt với 1 số nhiệm vụ, nhanh và rẻ hơn để xây dựng
d) Những yếu tố ảnh hưởng đến sự chính xác của ML:
- Khối lượng của dữ liệu (số hàng)
- Số lượng đặc điểm (số cột)
- Sự phù hợp của dữ liệu
- Tính phức tạp của mô hình
- Feature engineering: có kiến thức để tạo ra những đặc điểm mới từ dữ liệu
đầu vào
e) Đánh giá hiệu quả của ML:
+ Học sâu ( Deep learning)
- Không cần feature extraction như ML mà thay vào đó tích hợp bước feature extraction với classification
- DL ưu việt hơn:
+ DL có thể cải thiện rất nhiều về hiệu quả
+ Việc tính toán trở nên rẻ hơn, khiến DL trở nên khả thi: chúng ta có thể dự đoán từ
lượng lớn DL phi cấu trúc
- Ứng dụng: nhận diện hình ảnh, phân biệt tin giả, hàng giả từ hàng hiệu (detecting knockoffs from luxury products)
+ Mạng Nơ-ron nhân tạo:
- Neural Networks (NNs) được lấy cảm hứng từ mạng nơ-ron tế bào sinh học của con người. Mỗi một nơ ron trong mạng nơ ron nhân tạo là hàm toán học với chức năng thu thập và phân loại các thông tin dựa theo cấu trúc cụ thể. Nơ-ron lấy đầu vào từ các nơ-ron khác, chuyển đổi và truyền tín hiệu đi
- Kiến trúc của deep NNs: gồm 3 lớp
+ Tầng input layer (tầng vào): Tầng này nằm bên trái cùng của mạng, thể hiện cho các đầu vào của mạng.
+ Tầng output layer (tầng ra): Là tầng bên phải cùng và nó thể hiện cho những đầu ra của mạng.
+ Tầng hidden layer (tầng ẩn): Tầng này nằm giữa tầng vào và tầng ra nó thể hiện cho quá trình suy luận logic của mạng.
Lưu ý: Mỗi một Neural Network chỉ có duy nhất một tầng vào và 1 tầng ra nhưng lại có rất nhiều tầng ẩn.
- Ứng dụng: Trong lĩnh vực tài chính, mạng nơ ron nhân tạo hỗ trợ cho quá trình phát triển các quy trình như: giao dịch thuật toán, dự báo chuỗi thời gian, phân loại chứng khoán, mô hình rủi ro tín dụng và xây dựng chỉ báo độc quyền và công cụ phát sinh giá cả.
Deep Learning giúp Google theo dõi nguy cơ đau tim. - NNs đang rất thành công những năm gần đây, bởi:
+ Có nhiều thông số -> có thể xây dựng những mô hình phức tạp
+ Nhiều cải tiến trong tính toán và thuật toán cho phép nhiều tầng (layers) hơn - Có nên dùng DL thay cho ML? -> Không, vì chi phí rất đắt đỏ.
- NNs rất khó để hiểu và diễn giải
+ Big data - các đặc điểm:
- Volume (khối lượng)
VD: Vào năm 2016, lưu lượng di động toàn cầu ước tính là 6,2 Exabyte (6,2 tỷ GB) mỗi tháng. Trong năm 2020, chúng ta sẽ có gần 40000 ExaByte dữ liệu.
- Velocity (vận tốc): Vận tốc đề cập tích lũy dữ liệu tốc độ cao.
VD: Có hơn 3,5 tỷ lượt tìm kiếm mỗi ngày trên Google. Ngoài ra, người dùng FaceBook đang tăng khoảng 22% hàng năm.
- Variety (đa dạng): đề cập đến bản chất của dữ liệu là dữ liệu có cấu trúc, bán cấu trúc và dữ liệu phi cấu trúc; cũng đề cập đến các nguồn không đồng nhất
- Veracity (tính xác thực): sự không nhất quán và không chắc chắn trong dữ liệu, tức là dữ liệu có sẵn đôi khi có thể lộn xộn, chất lượng và độ chính xác rất khó kiểm soát.
- Value (giá trị): ý nghĩa/giá trị sử dụng của dữ liệu đối với DN
+ Ứng dụng AI, quy trình AI *
Ứng dụng: Hiện nay ứng dụng AI được ứng dụng rất rộng rãi trên rất nhiều lĩnh vực
-
-
-
Trong ngành vận tải:
+ Ô Tô tự lái, nhận diện biển số, phát hiện hành vi vượt đèn đỏ, lấn làn, lấn tuyến, ngược chiều, ....
+
Trong y tế:
+ Chẩn đoán bệnh thông qua ảnh X- Ray hay điện tâm đồ
+ Các máy chụp ảnh nhiệt ở sân bay để phân tích bệnh nhân nhiễm bệnh do vi
trùng từ việc chụp cắt lớp vi phổi, giúp cung cấp dữ liệu và theo dõi sự lây lan
của dịch bệnh CV 19
Trong quân sự:
 Có thể cải thiện độ an toàn, tốc độ và hiệu quả của giao thông đường sắt bằng
 cách giảm thiểu ma sát bánh xe, tối đa hóa tốc độ và cho phép lái xe tự động.

+ Máy bay không người lái
+ Radar phát hiện máy bay địch đi vào lãnh thổ.
- Trong giáo dục:
+
- Trong kinh doanh:
+ Các phần mềm chống gian lận trong thi cử
 + +
+
Dựa trên lịch sử giao dịch AI có thể dự đoán các giao dịch như chứng khoán, bất động sản,.. khi nào nên mua khi nào nên bán.
Khi truy cập vào web amazon.com, ngay lập tức hệ thống AI được sử dụng. Nếu đã mua hàng trước đó ngay lập tức hệ thống sẽ gợi ý cho bạn những sản phẩm liên quan đến lịch sử giao dịch trước đó. Nhưng nếu chưa có thì chỉ cần bạn gõ vào thanh tìm kiếm gõ chữ cái đầu tiên nó sẽ xuất hiện một loạt gợi ý các sản phẩm có chữ cái đó
Chạy quảng cáo tự động vào đúng thời điểm, dựa vào những yếu tố như thông tin nhân khẩu học, thói quen trong hoạt động trực tuyến và những nội dung mà khách hàng xem khi quảng cáo xuất hiện.
Phần mềm kiểm tra lỗi chính tả và ngữ pháp tiếng Anh
 + Phần mềm học nói tiếng anh
  + Ở Walmart họ đã ứng dụng hệ thống AI để quản lí sản phẩm trên kệ hàng
 để giảm tình trạng thiếu hàng trên kệ hàng, giải quyết tình trạng hàng tồn
 kho tại cửa hàng.
- Trong lĩnh vực giải trí:
+
- Trong nông nghiệp:
+ Ứng dụng AI ứng dụng trong thời gian thực giúp người nông dân phát hiện và kiểm soát dịch bệnh, kiểm soát lượng nước, tình trạng của đất, dự đoán sức khỏe cây trồng, dự đoán thời gian thu hoạch một cách chính xác nhất.
+ Hệ thống tự động tưới nước hay dự báo thời tiết để điều tiết lượng nước phù hợp
- Trong an ninh mạng :
+ Giúp dự đoán và ngăn chặn các cuộc tấn công mạng với độ chính xác cao
+ Ngăn chặn các phần mềm độc hại rất tốt
+ Cảnh báo người dùng trong những hành vi có nguy cơ bị xâm nhập cao.
Quy trình:
Thông qua phương pháp tiếp cận hệ thống chuyên gia
+ Kỹ thuật tri thức hoặc hệ thống chuyên gia
+ Nắm bắt và chuyển giao kiến thức từ các chuyên gia vào hệ thống máy tính.
VD: Phần mềm chuẩn đoán bệnh: phỏng vấn bác sĩ và hệ thống hóa các quy tắc mà họ sử dụng để chẩn đoán bệnh.
Phần mềm dành cho ô tô tự lái: phỏng vấn các tài xế và hệ thống hóa các quy tắc mà họ sử dụng để lái ô tô
 AI giúp Netflix cung cấp trải nghiệm cá nhân hóa cho người dùng, cụ thể
 hơn là chỉ cần cung cấp nội dung mà bạn muốn thì nó sẽ xuất hiện một
 chuỗi các sản phẩm mà bạn muốn để bạn lựa chọn
 + Sản phẩm Apex Utility AI được sử dụng trong game bắn súng để đánh giá
 xem nên tải loại vũ khí nào, bắn, chạy, ẩn nấp hay tấn công.
 + Máy tính chơi cờ vua đã đánh bại nhà vô địch cờ vua thế giới

+ Có thể tạo ra các hệ thống thông minh hợp lý, nhưng theo thời gian, các hệ thống chuyên gia không thể đánh bại con người ở những nhiệm vụ phức tạp đòi hỏi trí thông minh
  + Mô hình tạo sinh * ( Generative models)
Trả lợi:
- Generative models là tạo ra dữ liệu mới từ dữ liệu đã có
VD:
+ Tạo ra một gương mặt mới từ bức ảnh đã có sẵn.
+ Tạo ra một bài hát mới cùng thể loại, giai điệu
+ Gợi ý từ tiếp theo trong chuỗi các từ ngữ khi bạn muốn nhắn tin
+ Tạo văn bản khi bạn cung cấp chủ đề, từ khóa, số câu, số dòng,...
+ Vấn đề khi khởi nghiệp AI*
Trả lời:
- Sự khác biệt về hiệu suất giữa các thuật toán (có thể tương đối nhỏ)
- Chicken and Egg Problems:

+ Không có dữ liệu thì không xây dựng được chương trình AI
+ Không có sản phẩm -> không có khách hàng -> không có dữ liệu
Câu hỏi ôn tập:
1) Allàgì?Vídụ?
AI là lý thuyết và sự phát triển của hệ thống máy tính có khả năng thực hiện những nhiệm vụ mà thường yêu cầu trí tuệ của con người
3 loại AI:
+ Artificial Narrow Intelligence - weak AI: thực hiện rất tốt 1 nhiệm vụ cụ thể (thuật toán chơi cờ vua, hệ thống đề xuất,...)
+ Artificial General Intelligence - strong AI: có khả năng thực hiện mọi công việc như con người một cách dễ dàng và nhanh chóng (Artificial Neural Networks)
+ Artificial Superintelligence: tự cải thiện chính nó nhanh chóng và thực hiện nhiệm vụ ở một mức độ đáng kể về tốc độ và sự hoàn thiện
2) Dữ liệu phi cấu trúc và có cấu trúc?
- DL có cấu trúc: dễ dàng sắp xếp và thường được hàm chứa trong các cột và
hàng
3) ML có thể trở thành GPTs không?
Trả lời:
Có. Nó giúp hầu hết các lĩnh vực công nghiệp có khả năng thay đổi. Nhưng trong quá trình chuyển đổi có rất nhiều sự chậm trễ có thể là bị động và chủ động. Tạo ra rất nhiều cơ hội và các nhà quản lí có khả năng sử dụng công nghệ này và ứng dụng của nó để làm thay đổi hoàn toàn “ Mô hình kinh doanh” của các bạn và cơ sở hạ tầng công nghệ cũng thay đổi theo. Nó dẫn ảnh hưởng đế văn hóa tiêu dùng và sử dụng và quá trình tổ chức của doanh nghiệp
Vd: Khi phát triển Uber và Grab thì mô hình kinh doanh taxi truyền thống và xe ôm truyền thống dần dần bị loại bỏ và hạ tầng của nó yêu cầu của nó là mọi người cần phải có một chiếc smartphone.
4) Sự khác nhau giữa ML và DL?
Trả lời:
Sự khác nhau của ML và DL:
Đó là ML phải trải qua quá trình feature extraction
Rồi mới tới classification. Còn ở DL thì 2 quá trình hắn kết hợp lại với nhau.
Ví dụ cụ thể hơn: Ở cái nhận diện hình ảnh con chó với con mèo. Ở DL Chỉ cần cung cấp cho hắn thật nhiều ảnh để hắn phân tích và nhận dạng. Còn ML thì mình phải nhập thuộc tính. Ví dụ mắt thì sao, lông thì sao, mũi sao càng nhiều thuộc tính độ chính xác càng cao.
  VD: các văn bản, dữ liệu được trình bày theo dạng cột và hàng,...
 - DL phi cấu trúc: không thể chứa trong CSDL dạng hàng và cột, và nó cũng
 không có mô hình dữ liệu nào liên quan
 VD: hình ảnh, phim và các tệp âm thanh, các tệp chứa chữ cái, các nội dung
 từ mạng xã hội, hình ảnh từ vệ tinh, các bài thuyết trình, tệp PDF, các câu trả
 lời từ bản khảo sát câu hỏi mở, các trang web và bản thu từ các cuộc gọi hỗ
 trợ khách hàng.

 5) GANs và VAEs ?
GANs
- Sử dụng để tạo ra nội dung nhân tạo, ngày càng khó phân biệt với nội dung thật.
- Sử dụng hai mạng “ cạnh tranh” với nhau
+ Sử dụng mạng chung tạo ra một nội dung mới
+ Sử dụng một mạng khác, một mạng phân biệt, được sử dụng một cách đơn
giản để phân biệt đầu ra của mạng đầu tiên là thật hay giả.
VD:
Phân biệt tiền thật giả: Training set: Cung cấp những dữ liệu về tiền thật, như hình dạng màu sắc, chất liệu, độ dài,... cho discriminator
+ random noise -> Generator ( phát hiện tiền giả) -> Discriminator -> tiền giả random noise -> Generator ( phát hiện tiền thật) -> Discriminator -> tiền thật
VAEs
- Sử dụng để tạo ra một nội dung mới mà chúng ta có thể kiểm soát được các thuộc
tính của nó.
VD: Input: là một bức ảnh tường đen và nhân vật có đeo kính -> encode -> decode -> bức ảnh mới tường trắng và nhân vật không đeo kính.
Mô hình AI đã chuẩn thực hiện trong báo cáo giữa kỳ. Mô tả hệ thống, quy trình, cách thức hoạt động của mô hình AI, etc.Độ chính xác càng cao thì tỉ lệ sai càng thấp, độ chính xác càng cao thì dẫn đến phân loại hoàn hảo (perfect classifier )""",
    },
    {
        "label": "PDF scan (ảnh chụp)",
        "file": "/Users/hoangtramanh/Documents/BAO CAO THUC TAP/Project/media/subjects/1/Phân tích dữ liệu bằng Python/Exercises/Screenshot 2026-04-13 at 01.41.15.png",  
        "truth": "Trường Đại học Kinh tế Khoa Thương mại Điện tử Đề thi kết thúc học phần bậc Đại học hệ chính quy Học kỳ 1 Năm học 2023-2024 Tên học phần Phân tích dữ liệu bằng Python Mã học phần MIS3041 Hình thức thi Vấn đáp Thời gian vấn đáp 5-8 phút Mã đề thi 04 Câu 1 Anh chị hãy nêu qui trình tạo ra quyết định dựa vào dữ liệu trong kinh doanh và cho ví dụ cụ thể để minh họa về qui trình này. Câu 2 Giải thích cú pháp numpy.dtype object và cho ví dụ minh họa. Lệnh np.random.seed 1 để làm gì? Sinh viên không được viết vào đề thi và nộp lại đề thi",
    },
]

def wer(ref, hyp):
    r = ref.lower()
    h_words = hyp.lower().split()[:len(r.split())]
    h = " ".join(h_words)
    return round(jiwer.wer(r, h) * 100, 2)

def cer(ref, hyp):
    r = ref.lower()
    h = hyp.lower()[:len(r)]
    return round(jiwer.cer(r, h) * 100, 2)

if PREVIEW_MODE:
    print(f"\n Đang trích xuất: {PREVIEW_FILE}\n" + "─"*60)
    print(extract_text(PREVIEW_FILE))
    print("─"*60 + "\n→ Copy 1 đoạn 50–100 từ vào 'truth', đổi PREVIEW_MODE=False")
else:
    print("="*60 + "\n  KẾT QUẢ ĐÁNH GIÁ OCR\n" + "="*60)
    rows = []
    for tc in test_cases:
        if not os.path.exists(tc["file"]):
            print(f"Bỏ qua (không tìm thấy): {tc['file']}"); continue
        ocr = extract_text(tc["file"])
        w, c = wer(tc["truth"], ocr), cer(tc["truth"], ocr)
        rows.append((tc["label"], w, c))
        print(f"  [{tc['label']}]  WER={w}%  CER={c}%")
    if rows:
        print("─"*60)
        print(f"  {'Loại':<30} {'WER':>6} {'CER':>6}")
        for label,w,c in rows: print(f"  {label:<30} {w:>5.1f}% {c:>5.1f}%")
        print(f"  {'Trung bình':<30} {sum(r[1] for r in rows)/len(rows):>5.1f}% {sum(r[2] for r in rows)/len(rows):>5.1f}%")
        print("="*60)
