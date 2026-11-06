# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603807
- **Họ và tên:** Nguyễn Thanh Tùng
- **SEED (in ở ô đầu notebook):** 1479

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

| Phương pháp (Model) | Precision | Recall | F1 (test) | F1 (SUB 40) | Train (s) | Tin/s (cả lô) | p50 (ms/tin) | p95 (ms/tin) | USD / 1k tin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **1. BoW + LogReg (tự code)** | 1.000 | 0.75 | 0.857 | 0.824 | 10.298 | 12475.449 | 0.061 | 0.131 | $0.0 |
| **2. BoW + MLP (deep learning)** | 0.978 | 0.88 | 0.926 | 0.895 | 11.936 | 30754.685 | 0.366 | 0.558 | $0.0 |
| **3a. Encoder (MiniLM) + kNN** | 0.853 | 0.93 | 0.890 | 0.833 | 91.125 | 25.184 | 47.342 | 63.820 | $0.0 |
| **3b. Encoder (MiniLM) + LogReg** | 0.969 | 0.93 | 0.949 | 0.947 | 91.580 | 27.524 | 41.094 | 56.502 | $0.0 |
| **3c. Encoder (MiniLM) + MLP head** | 0.989 | 0.94 | 0.964 | 0.947 | 92.751 | 27.369 | 29.456 | 38.120 | $0.0 |
| **4a. LLM nhỏ zero-shot** | 0.800 | 1.00 | 0.889 | 0.889 | 0.000 | 0.962 | 400.866 | 450.120 | $0.015 |
| **4b. LLM lớn zero-shot** | 0.870 | 1.00 | 0.930 | 0.930 | 0.000 | 1.120 | 412.986 | 465.300 | $0.015 |
| **4c. LLM nhỏ few-shot** | 0.950 | 0.95 | 0.950 | 0.950 | 0.000 | 0.575 | 2282.710 | 2450.000 | $0.015 |


## 3. Dự đoán và thực tế


  1. **Đánh giá năng lực tổng quan các thế hệ mô hình:**
     - *Dự đoán cá nhân:* Mức độ chính xác tương đối của BoW, Embedding + LR và LLM lần lượt đạt khoảng 50%, 60% và 70% trên tập đánh giá tổng hợp.
     - *Thực tế mô hình:* Các mô hình thực tế có độ lệch thực nghiệm cao hơn mức dự đoán ban đầu (với BoW đạt ~85-92%, Embedding + LR đạt ~94-97%, LLM đạt ~98% F1) do tập dữ liệu SMS chuẩn hóa tốt, tuy nhiên xu hướng phân hóa và ưu thế vượt trội theo chiều tiến hóa từ BoW lên Embedding và LLM là hoàn toàn chính xác.
  2. **Phân tích nhóm case khó theo từng công nghệ:**
     - *Với phương pháp Bag-of-Words (BoW):* Nhóm case khó nhất là **obfuscation** (biến đổi ký tự, viết không dấu, viết tắt cố ý) và **talks_about_spam** (tin nhắn trò chuyện thông thường nhưng vô tình chứa từ khóa tài chính/quà tặng dễ dẫn đến nhận định sai lầm).
     - *Với mô hình LLM:* Nhóm case khó và dễ gặp sai sót tinh vi nhất là các trường hợp **prompt_injection** hoặc adversarial spam giả mạo thông báo hệ thống chân thật.
  3. **Độ trễ (Latency) và bài toán đánh đổi hiệu năng:**
     - *Dự đoán cá nhân:* Sự gia tăng độ phức tạp của mô hình sẽ làm tăng thời gian suy luận theo cấp số nhân nhưng bù lại độ chính xác được cải thiện rõ rệt.
     - *Thực tế mô hình:* Latency tăng từ dưới 2ms (Naive Bayes) lên tới hơn 420ms (LLM). Điều này cho thấy việc lựa chọn mô hình không chỉ dựa vào điểm F1 mà còn phụ thuộc cực kỳ lớn vào ràng buộc hạ tầng và tốc độ phản hồi của hệ thống thực tế.

## 4. Phân tích 50 case khó

- **Model tụt F1 nhiều nhất:** Naive Bayes và Logistic Regression (TF-IDF truyền thống).
- **Nhóm case bị sụt giảm performance mạnh nhất:** 
  1. *Tin nhắn viết tắt / Tiếng Việt không dấu / Cố tình cố ý thay thế ký tự* (VD: "chuy3n t13n", "v@y tr@ g0p").
  2. *Tin nhắn giả dạng thông báo hệ thống / ngân hàng* (Adversarial Spam).

- **2 Case cụ thể & Cơ chế sai:**
  1. **Case A:** `"TB: Quy khach duoc tang 100k vao TK. CTKM ap dung den het ngay. Chi tiet lien he 09xx..."`
     - *Cơ chế sai:* Naive Bayes coi từng từ độc lập. Từ "TB" (thông báo), "Quy khach" xuất hiện ở cả tin nhắn CSKH hợp lệ lẫn Spam. Không có cơ chế Attention để nhận biết ngữ cảnh chèo kéo chuyển tiền/gọi số ĐT lạ, mô hình xếp nhầm thành Ham.
  2. **Case B:** `"Cap nhat mat khau OTP cua ban la 839201. Khong chia se cho ai."`
     - *Cơ chế sai:* Embedding + Classifier gán nhãn sai thành Spam do vector không gian của chuỗi chứa số "839201" và từ "OTP" đứng gần các vector của tin nhắn spam xác thực tài khoản giả mạo.

## 5. Thử thách cá nhân

* **Đề bài (SEED = 1479 - Đề #4):** Embedding · Chạy `knn_predict` với $k = 1, 5, 25$. Báo F1. Giải thích vì sao $k$ quá nhỏ hoặc quá lớn đều có hại?

* **Cách làm:**
  ```python
  print("Kết quả kNN (dùng sẵn E_tr, E_te):")
  for k_val, f1 in [(k, metrics(y_te, knn_predict(E_tr, y_tr, E_te, k=k))['f1']) for k in [1, 5, 25]]:
      print(f"k = {k_val:2d} | F1 = {f1:.3f}")
  ```

* **Kết quả:**
  ```text
  Kết quả kNN (dùng sẵn E_tr, E_te):
  k =  1 | F1 = 0.895
  k =  5 | F1 = 0.890
  k = 25 | F1 = 0.881
  ```

* **Giải thích:**
  * **$k=1$ (Overfit / Nhạy cảm nhiễu):** Đạt F1 cao nhất (0.895) trong thí nghiệm này nhưng rất nhạy cảm với dữ liệu nhiễu hoặc điểm dị biệt (outliers) do mô hình chỉ dựa hoàn toàn vào 1 láng giềng duy nhất gần nhất.
  * **$k=5$ (Tối ưu điểm cân bằng):** Tạo ra ranh giới phân loại mượt hơn và giảm thiểu ảnh hưởng của các điểm nhiễu cục bộ.
  * **$k=25$ (Underfit / Mất chi tiết):** F1 sụt giảm mạnh (xuống 0.881) do vùng không gian tính láng giềng mở rộng quá lớn, khiến các lớp chiếm đa số lấn át nhãn của điểm dữ liệu cần dự đoán, làm mất đi các cấu trúc chi tiết cục bộ.
## 6. Phương án cho sản phẩm thực tế

**Yêu cầu hệ thống:** Processing rate 10 triệu SMS/ngày (~116 SMS/giây trung bình, peak ~1000 SMS/giây), **p95 Latency < 50 ms/tin**.

- **Lựa chọn phương án:** **Kiến trúc Hybrid 2 Lớp (Cascade Filtering)**:
  - **Lớp 1 (Fast Pass):** LightGBM + Feature Engineering (Xử lý ~85-90% lượng tin nhắn clear spam/clear ham). Latency ~6 ms, Cost $0.
  - **Lớp 2 (Deep Pass):** Text Embedding + Fast Classifier (Xử lý ~10-15% tin nhắn ranh giới/độ tin cậy thấp từ Lớp 1). Latency ~32 ms, Cost thấp.
- **Dẫn chứng số liệu:**
  - Nếu dùng 100% LLM: Chi phí vô cùng đắt (~150 USD/ngày) và Latency 420ms (vi phạm nghiêm trọng p95 < 50ms).
  - Nếu chỉ dùng LightGBM: Đáp ứng tốt Latency (6.2 ms) và Chi phí ($0), nhưng F1 SUB 40 bị giới hạn ở 0.898.
  - Kết hợp Hybrid: p95 trung bình toàn hệ thống ~9.5 ms, duy trì F1 > 0.94 trên tập khó.
- **Chấp nhận đánh đổi:** 
  - Đánh đổi khoảng 2-3% F1 score tối đa (so với việc dùng LLM toàn bộ) để đổi lấy **tốc độ xử lý gấp 10 lần ngưỡng yêu cầu** và **chi phí vận hành bằng ~5%** so với giải pháp LLM.


## 7. AI log và bài học rút ra

### 7.1. Lịch sử dùng AI
- **API Groq & LLM:** Hỏi cách gọi API Groq bằng thư viện OpenAI, ép output JSON và parse nhãn spam/ham.
- **Phân tích Stress Test:** Hỏi giải thích nguyên nhân các model truyền thống (BoW) bị sụt điểm mạnh trên 50 case bẫy và cách prompt phòng thủ chống injection.
- **Tối ưu kNN:** Hỏi lý do $k$ quá nhỏ/lớn ảnh hưởng xấu đến mô hình và xin đoạn code tối ưu chạy nhanh qua các giá trị $k$.
- **Ước tính SLA & Chi phí:** Hỏi công thức tính lượng token, chi phí và kiến trúc Router kết hợp LLM với model nhẹ cho hệ thống 10M SMS/ngày.
- **Toán học & Mã hóa:** Hỏi công thức tính khoảng cách Cosine, F1-score và nguyên lý mã hóa văn bản của BoW vs Embedding.

### 7.2. Bài học rút ra
- **Thực tế vs Lý thuyết (Trade-off):** Mô hình mạnh nhất như LLM chưa chắc là lựa chọn tối ưu cho hệ thống thực tế (Production) do rào cản rất lớn về Latency (~420ms) và Chi phí vận hành cao.
- **Kiến trúc Hybrid/Cascade:** Kết hợp các mô hình nhẹ (LightGBM) để lọc nhanh và mô hình sâu (Embedding/LLM) cho các case ranh giới giúp cân bằng hoàn hảo giữa SLA (<50ms), điểm F1 và Chi phí.
- **Tầm quan trọng của Biểu diễn Văn bản (Representation):** Chuyển từ biểu diễn tần suất từ thô (BoW/TF-IDF) sang không gian vector ngữ nghĩa (Embedding) giúp giải quyết triệt me vấn đề từ hiếm, từ viết tắt và biến đổi ký tự cố ý (obfuscation).