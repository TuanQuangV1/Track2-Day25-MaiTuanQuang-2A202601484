# BÁO CÁO PHÂN TÍCH TỐI ƯU HÓA CHI PHÍ GPU (GPU FINOPS WRITE-UP)
**Dự án:** Lab 25 — GPU FinOps Workshop · **Học viên:** Mai Tuấn Quang  
**Vị trí:** Lead FinOps Engineer @ NimbusAI  

---

## 1. So sánh Baseline vs. Optimized & Hiệu quả Kinh tế Token

Tại NimbusAI, chi phí GPU ban đầu đang tiêu tốn **$27,133 / tháng** mà không có các cơ chế quản trị và tối ưu hóa hạ tầng. Bằng việc áp dụng toàn diện 4 đòn bẩy FinOps cốt lõi, chi phí vận hành đã giảm xuống còn **$14,626 / tháng**, mang lại mức tiết kiệm **$12,507 / tháng (tiết kiệm 46.1%)**.

| Chỉ số Kinh tế | Baseline (Ban đầu) | Optimized (Sau tối ưu) | Mức Cải thiện |
|---|---|---|---|
| **Tổng Chi phí Vận hành (Monthly Spend)** | **$27,133** | **$14,626** | **Giảm 46.1% (-$12,507/tháng)** |
| **Chi phí Inference Hàng ngày** | $45.54 / ngày | $7.90 / ngày | Giảm 82.6% |
| **Đơn giá Kinh tế Token (`$/1M-token`)** | **$6.488 / 1M tokens** | **$1.126 / 1M tokens** | **Cắt giảm 82.6% đơn giá phục vụ** |
| **Tag Coverage & Chargeback Readiness** | 92% (Coverage) | Sẵn sàng Chargeback | Mở cổng thu hồi chi phí theo team |

> **Bài học Cốt lõi về Thước đo:** Nếu chỉ nhìn vào đơn giá thuê `$/GPU-giờ`, ta không thể thấy được hiệu quả thực sự mà hệ thống mang lại. Khi chuẩn hóa về đơn vị **`$/1M-token`**, hiệu quả của việc phân tầng model (Cascading), Prompt Caching (-90%) và Batch API (-50%) được phản ánh rõ rệt với mức giảm chi phí token lên tới **82.6%**.

---

## 2. Phân tích Chuyên sâu từng Đòn bẩy Tối ưu (FinOps Levers)

| Đòn bẩy (Lever) | Tiết kiệm (USD/tháng) | % Tổng Tiết kiệm | Nguyên nhân Đóng góp & Cơ chế Hoạt động |
|---|---|---|---|
| **Purchasing Strategy (Spot / Reserved)** | **$10,040** | **80.3%** | **Đóng góp lớn nhất.** Các workload chạy 24/7 (như inference chat, RAG) chuyển sang cam kết Reserved 3-năm hưởng chiết khấu 45% (vượt xa điểm hòa vốn 55% duty cycle). Các job training/fine-tuning gián đoạn được chuyển sang Spot instance có checkpoint. |
| **Inference Levers (Cascade / Cache / Batch)** | **$1,212** | **9.7%** | **Đòn bẩy hiệu quả token cao nhất.** 80% truy vấn đơn giản được định tuyến sang model nhỏ (rẻ hơn 15×); prompt tĩnh của Chat/RAG được cache (-90% input); các tác vụ eval chạy offline chuyển sang Batch API (-50%). |
| **Right-size Util-Lies** | **$655** | **5.2%** | Phát hiện các GPU H100/A10G bị nghẽn memory-bound hoặc kernel stall (MFU thấp dù GPU-Util cao), hạ cấp sang A100/L4 để tiết kiệm chi phí phần cứng chênh lệch. |
| **Kill Idle GPUs** | **$600** | **4.8%** | Tự động phát hiện và ngắt các GPU để trống qua đêm (`gpu-h100-5` idle 8 tiếng/ngày với utilization <10%), loại bỏ $20/ngày lãng phí thuần túy. |

---

## 3. Bản chất Hiện tượng "GPU-Util Lie" & Tác động Tài chính

### 3.1 Cơ chế Vi kiến trúc
Chỉ số `gpu_util_pct` hiển thị trên `nvidia-smi` chỉ là tỷ lệ thời gian có ít nhất 1 kernel đang thực thi trên GPU (Clock Active Time). Nó **không phản ánh** tỷ lệ Streaming Multiprocessors (SM) hay Tensor Cores thực sự bận rộn tính toán ma trận (Compute Efficiency).

### 3.2 Các GPU bị "Lie" trong hệ thống NimbusAI:
1. **`gpu-h100-4`:** `gpu_util_pct = 98.2%`, nhưng **`MFU = 0.194` (chỉ 19.4%)** và `MBU = 0.207`.
2. **`gpu-a10g-1`:** `gpu_util_pct = 96.9%`, nhưng **`MFU = 0.268`**.

### 3.3 Nguyên nhân Kỹ thuật theo Roofline Model:
- **Memory Bandwidth Stall:** Trong giai đoạn Token Generation (Decode), Arithmetic Intensity cực thấp ($\approx 1-2\text{ FLOP/byte}$), nằm sâu trong vùng *Memory-Bound* (dưới Ridge Point $295\text{ FLOP/byte}$ của H100). GPU liên tục dừng chờ nạp weights từ HBM.
- **Kernel Launch & I/O Overhead:** Training phân tán bị nghẽn AllReduce đồng bộ mạng hoặc CPU nạp dữ liệu chậm, khiến GPU clock luôn active nhưng Tensor Cores rảnh rỗi.

### 3.4 Tác động Tài chính:
Với `gpu-h100-4`, công ty trả trọn vẹn **$2.50 / giờ ($1,800 / tháng)** nhưng chỉ nhận được **~$0.49 giá trị FLOPs thực tế**. Việc hạ cấp xuống A100 ($1.79/h) hoặc tái cấu trúc batching giúp NimbusAI thu hồi hàng ngàn USD mỗi tháng mà không suy giảm thông lượng thực tế.

---

## 4. Kết quả Thực hiện 5 Phần mở rộng "Your Turn"

### Extension 1: Nâng cấp Chính sách Mua GPU `recommend_tier()`
- **Triển khai:** Tích hợp ma trận đánh giá rủi ro gián đoạn (Interruption Rate) theo kiến trúc GPU (H100 ~3% vs A10G ~8%) và thời hạn công việc (Job Duration).
- **Kết quả:** Phân loại chính xác 100% các job: job ngắn hạn gián đoạn $\rightarrow$ Spot + Checkpoint; job 24/7 dài hạn $\rightarrow$ 3-Year Reserved. Tiết kiệm 39.1% chi phí mua phần cứng.

### Extension 2: Right-sizing Phần cứng dựa trên MBU & $/GB-VRAM
- **Triển khai:** Xây dựng bảng đơn vị kinh tế phần cứng (`$/GB-VRAM/h`, `$/(TB/s) BW/h`, `TFLOPs/$`). Đề xuất GPU thay thế đáp ứng đủ $1.2\times$ dung lượng VRAM và băng thông bộ nhớ thực tế.
- **Kết quả:** Nhận diện các tác vụ memory-bound trên H100 có thể chuyển sang A100/A10G, tiết kiệm thêm tới **$1,310 / tháng** nếu right-size toàn bộ fleet.

### Extension 3: Mô hình Kinh tế học Prompt Cache `cache_is_worth_it()`
- **Triển khai:** Thiết lập công thức điểm hòa vốn số lượt đọc lại ($N_{\text{reads}} = \frac{\text{Write Cost}}{(1 - \text{Read Discount}) \times \text{Base Price}}$).
- **Kết quả:** Với model nhỏ, điểm hòa vốn là **1.39 lượt đọc**; model lớn là **1.39 lượt đọc**. Vì các ứng dụng Chat và RAG có số lần tái sử dụng system prompt trung bình $\approx 4.5\text{ lần} > 1.39$, việc bật prompt caching mang lại ROI dương vượt trội (tiết kiệm ròng).

### Extension 4: Quản trị Ngân sách & Năng lượng của Reasoning Tokens
- **Triển khai:** Đo lường tác động của cờ `is_reasoning = 1`.
- **Kết quả:** Reasoning requests chỉ chiếm **8.4% tổng số truy vấn**, nhưng tiêu thụ tới **94.0% tổng năng lượng điện inference** và chiếm **16.5% chi phí inference** do sinh chuỗi CoT (Chain-of-Thought) dài và hệ số năng lượng $80\times$.
- **Khuyến nghị:** Áp dụng bộ lọc độ phức tạp prompt (Routing Threshold). Nếu giới hạn reasoning ở mức 5% traffic, công ty tiết kiệm thêm **$0.30 / ngày ($9 / tháng)** và **11.93 kWh điện / ngày**.

### Extension 5: Điều phối Tác vụ Nhận thức Carbon (Carbon-Aware Scheduling)
- **Triển khai:** Đánh giá 4,227 kWh năng lượng hàng tháng của các job training gián đoạn trên 5 khu vực đám mây.
- **Kết quả So sánh:**
  - `europe-north1` (Na Uy - Thủy điện): **30 gCO2/kWh** $\rightarrow$ Cắt giảm **1,479.5 kg CO2e (giảm 92.1% khí thải)** so với chạy tại `us-east-1` (380 gCO2/kWh).
  - `us-east-wa` (Washington): **$0.055/kWh** $\rightarrow$ Giảm **$274.75 / tháng tiền điện** so với `us-east-1` ($0.12/kWh).

---

## 5. Top 3 Khuyến nghị Hành động Dành cho Ban Giám đốc NimbusAI

Với tư cách là FinOps Lead, 3 hành động ưu tiên cao nhất cần thực thi ngay lập tức là:

1. **Khuyến nghị 1 (Thực thi trong 24h - Tiết kiệm ngay $10,640/tháng):**
   - **Ký cam kết Reserved 3-năm** cho 3 cụm GPU phục vụ inference 24/7 (`job-infer-chat`, `job-infer-rag`, `job-infer-search`).
   - **Bật cấu hình Auto-shutdown** tự động ngắt `gpu-h100-5` khi GPU-Util < 10% trong 30 phút.
2. **Khuyến nghị 2 (Thực thi trong Tuần 1 - Tối ưu 82.6% $/1M-token):**
   - **Bắt buộc triển khai Prompt Caching & Model Cascading** trên Gateway API: Định tuyến 80% prompt đơn giản sang model nhỏ và cache toàn bộ System Prompts của chatbot/RAG.
   - **Chuyển toàn bộ Nightly Eval sang Batch API** để tự động hưởng chiết khấu 50%.
3. **Khuyến nghị 3 (Thực thi trong Tháng 1 - Governance & Green FinOps):**
   - **Kích hoạt chính sách Chargeback:** Với tag coverage hiện tại đạt **92%** ($\ge 80\%$), tiến hành phân bổ hóa đơn định kỳ tới 4 team (`assistant`, `search`, `eval`, `rag`) thông qua xuất dữ liệu chuẩn **FOCUS 1.x**.
   - **Di chuyển toàn bộ Batch Training sang `europe-north1`**: Đạt mục tiêu kép vừa giảm phát thải carbon 92.1%, vừa tối ưu hóa chi phí vận hành xanh.
