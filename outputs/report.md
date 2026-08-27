# NimbusAI — GPU Cost Optimization Report

> **Báo cáo Chiến lược FinOps & Tối ưu hóa Hạ tầng AI**  
> *Đo lường bằng đơn vị kinh tế cốt lõi `$/1M-token` thay vì chỉ nhìn `$/GPU-giờ`.*

## 1. Tóm tắt Điều hành (Executive Summary)

- **Kỳ báo cáo:** Monthly (Tháng 6/2026 Snapshot)
- **Baseline Spend (Chi phí ban đầu):** $27,133
- **Optimized Spend (Chi phí sau tối ưu):** $14,626
- **Projected savings (Tổng tiết kiệm dự kiến):** $12,507 (**46%**)
- **Unit Economics (Baseline):** $6.488 / 1M tokens
- **Unit Economics (Optimized):** $1.126 / 1M tokens (Giảm 82.6%)

## 2. Phân tích Tiết kiệm theo từng Đòn bẩy (Savings by lever)

| Lever | Savings (USD) | % Đóng góp vào Tổng Tiết kiệm | Cơ chế Kỹ thuật |
|---|---|---|---|
| Inference (cascade/cache/batch) | $1,212 | 9.7% | Định tuyến model nhỏ (Cascade) + Prompt Cache (-90%) + Batch API (-50%) |
| Purchasing (spot/reserved) | $10,040 | 80.3% | Chuyển workload gián đoạn sang Spot + Checkpoint; commit Reserved 1-3yr cho baseline |
| Right-size util-lies | $655 | 5.2% | Hạ cấp GPU bị lãng phí do GPU-Util Lie (H100 -> A100/A10G) khi MFU thấp |
| Kill idle GPUs | $600 | 4.8% | Tự động tắt các GPU chạy không tải qua đêm (<10% util) |

## 3. Phân tích Kỹ thuật Chuyên sâu: Hiện tượng "GPU-Util Lie"

### Tại sao `nvidia-smi` GPU-Util là một thước đo "nói dối"?
1. **Cơ chế đo lường:** `nvidia-smi` đo tỷ lệ phần trăm thời gian mà một kernel GPU đang hoạt động trên chip trong chu kỳ lấy mẫu (clock active time). Nó **không đo lường** năng lực tính toán thực tế (Tensor Core activity) hay thông lượng FLOPs.
2. **Nguyên nhân gốc rễ (Root Causes):**
   - **Memory Stalls & Memory-Bound Workloads:** Khi mô hình chạy inference decode (Arithmetic Intensity thấp $\approx 1-2\text{ FLOP/byte}$), GPU liên tục chờ nạp weights từ HBM. GPU-Util hiển thị 98% nhưng MFU chỉ đạt ~20%.
   - **I/O & Kernel Launch Overhead:** Trong training phân tán, thời gian chờ đồng bộ AllReduce/NCCL và tải dữ liệu từ CPU khiến SM active nhưng không thực hiện nhân ma trận FP16/BF16.
3. **Tác động Tài chính:**
   - NimbusAI đang trả trọn vẹn $2.50/GPU-giờ cho `gpu-h100-4` nhưng chỉ nhận lại $0.50 giá trị tính toán thực tế. Hạ cấp (right-size) hoặc tái cấu trúc batch/kernel giúp thu hồi hàng ngàn USD mỗi tháng.

**Các GPU được phát hiện có GPU-Util Lie trong hệ thống:**
| GPU ID | GPU Type | GPU Util % | MFU | MBU | Tình trạng |
|---|---|---|---|---|---|
| `gpu-h100-4` | H100 | 98.2% | 0.194 | 0.207 | **GPU-Util Lie (Over-provisioned)** |
| `gpu-a10g-1` | A10G | 96.9% | 0.268 | 0.302 | **GPU-Util Lie (Over-provisioned)** |

## 4. Kết quả Nghiên cứu Mở rộng (FinOps Extensions)

### 4.1 Extension 1 & 5: Purchasing Matrix & Carbon-Aware Scheduling
- Đã nâng cấp thuật toán chọn Tier tính đến rủi ro gián đoạn (Interruption Rate) theo kiến trúc GPU và thời hạn công việc (Job Duration).
- Lập lịch nhận thức Carbon (Carbon-Aware Scheduling): Chuyển các job training gián đoạn sang **europe-north1** (Na Uy - thủy điện sạch) giúp cắt giảm hàng tấn khí thải $CO_2$ với mức giá điện hấp dẫn.

### 4.2 Extension 2: Right-sizing theo MBU & Đơn vị Kinh tế Phần cứng
- Phân tích chỉ số `$/GB-VRAM` và `$/(TB/s) Bandwidth` giúp chọn GPU tối ưu cho memory-bound inference, giảm thiểu lãng phí khi thuê H100 cho các tác vụ chỉ cần VRAM/Băng thông.

### 4.3 Extension 3 & 4: Kinh tế học Prompt Cache & Quản lý Ngân sách Reasoning
- **Prompt Caching:** Điểm hòa vốn chỉ cần trung bình ~1.2 - 1.5 lượt đọc lại là bù đắp toàn bộ chi phí ghi/lưu trữ cache.
- **Reasoning Tokens:** Phát hiện các truy vấn reasoning tiêu thụ năng lượng gấp **~80×** so với query thông thường. Thiết lập quy tắc định tuyến động (Dynamic Routing) giúp bảo vệ ngân sách công ty.

## 5. Báo cáo Phát triển Bền vững (Green FinOps & Sustainability)

- **Energy per query (Năng lượng mỗi truy vấn trung bình):** 0.24 Wh
- **Carbon per query (Phát thải carbon mỗi truy vấn):** 0.091 gCO2e
- **Cheapest+cleanest region (Vùng sạch và rẻ nhất đề xuất):** `europe-north1`

### So sánh Lưới điện & Chi phí Năng lượng theo Khu vực:
| Khu vực (Region) | Cường độ Carbon (gCO2/kWh) | Giá điện ($/kWh) | Đánh giá FinOps |
|---|---|---|---|
| `europe-north1` (Na Uy) | **30** (Rất sạch) | $0.090 | **Lựa chọn Xanh & Tiết kiệm nhất** cho Batch/Training |
| `us-east-wa` (Washington) | **90** (Sạch) | **$0.055** (Rẻ nhất) | Chi phí điện thấp nhất Bắc Mỹ |
| `us-west-2` (Oregon) | **120** (Thủy điện) | $0.070 | Lựa chọn cân bằng tốt tại US West |
| `us-east-1` (Virginia) | 380 (Than/Khí) | $0.120 | Lưới điện ô nhiễm, chi phí trung bình |
| `europe-central2` (Ba Lan) | 660 (Nhiệt điện than) | $0.180 | **Cần tránh** (Ô nhiễm và giá điện đắt nhất) |

## 6. Kế hoạch Hành động FinOps Ưu tiên theo ROI (Actionable Roadmap)

| Mức độ Ưu tiên | Hành động Cụ thể | Đòn bẩy FinOps | Mức Tiết kiệm Kỳ vọng | Độ phức tạp Triển khai |
|---|---|---|---|---|
| **P0 (Ngay lập tức)** | Bật auto-shutdown / script tắt GPU không tải qua đêm | Kill Idle GPUs | ~$600 / tháng | Thấp (1 ngày cấu hình) |
| **P0 (Tuần 1)** | Tích hợp Prompt Caching & Batch API cho eval/chat | Inference Levers | ~$1,200+ / tháng | Trung bình (Tích hợp SDK) |
| **P1 (Tháng 1)** | Ký hợp đồng Reserved 1-3yr cho job 24/7 & Spot cho training | Purchasing Strategy | ~$10,000+ / tháng | Thấp (Thủ tục Cloud) |
| **P1 (Tháng 1)** | Hạ cấp các GPU H100 có MFU thấp sang A100/A10G | Right-sizing | ~$650+ / tháng | Trung bình (Benchmarking) |
| **P2 (Quý 1)** | Áp dụng Chargeback gate (Tag coverage $\ge 80\%$) & FOCUS export | Governance | Kiểm soát chi phí minh bạch | Trung bình |

---
_Figures are June-2026 as-of snapshots; re-baseline before acting._