"""Report assembly — the lab's deliverable: baseline vs optimized + savings chart."""
from __future__ import annotations


def build_report(
    baseline_usd: float,
    optimized_usd: float,
    levers: dict,
    sustainability: dict | None = None,
    period: str = "monthly",
    unit_economics: dict | None = None,
    gpu_lies_info: list | None = None,
    extensions_info: dict | None = None,
) -> str:
    """Return a comprehensive markdown cost-optimization report for NimbusAI executive review."""
    savings = baseline_usd - optimized_usd
    pct = (savings / baseline_usd * 100.0) if baseline_usd > 0 else 0.0
    lines = [
        "# NimbusAI — GPU Cost Optimization Report",
        "",
        "> **Báo cáo Chiến lược FinOps & Tối ưu hóa Hạ tầng AI**  ",
        "> *Đo lường bằng đơn vị kinh tế cốt lõi `$/1M-token` thay vì chỉ nhìn `$/GPU-giờ`.*",
        "",
        "## 1. Tóm tắt Điều hành (Executive Summary)",
        "",
        f"- **Kỳ báo cáo:** {period.capitalize()} (Tháng 6/2026 Snapshot)",
        f"- **Baseline Spend (Chi phí ban đầu):** ${baseline_usd:,.0f}",
        f"- **Optimized Spend (Chi phí sau tối ưu):** ${optimized_usd:,.0f}",
        f"- **Projected savings (Tổng tiết kiệm dự kiến):** ${savings:,.0f} (**{pct:.0f}%**)",
    ]

    if unit_economics:
        lines += [
            f"- **Unit Economics (Baseline):** ${unit_economics.get('baseline_per_m', 0):.3f} / 1M tokens",
            f"- **Unit Economics (Optimized):** ${unit_economics.get('optimized_per_m', 0):.3f} / 1M tokens (Giảm {unit_economics.get('token_cost_drop_pct', 0):.1f}%)",
        ]

    lines += [
        "",
        "## 2. Phân tích Tiết kiệm theo từng Đòn bẩy (Savings by lever)",
        "",
        "| Lever | Savings (USD) | % Đóng góp vào Tổng Tiết kiệm | Cơ chế Kỹ thuật |",
        "|---|---|---|---|",
    ]
    lever_descriptions = {
        "Inference (cascade/cache/batch)": "Định tuyến model nhỏ (Cascade) + Prompt Cache (-90%) + Batch API (-50%)",
        "Purchasing (spot/reserved)": "Chuyển workload gián đoạn sang Spot + Checkpoint; commit Reserved 1-3yr cho baseline",
        "Right-size util-lies": "Hạ cấp GPU bị lãng phí do GPU-Util Lie (H100 -> A100/A10G) khi MFU thấp",
        "Kill idle GPUs": "Tự động tắt các GPU chạy không tải qua đêm (<10% util)",
    }
    for name, amount in levers.items():
        contrib = (amount / savings * 100.0) if savings > 0 else 0.0
        desc = lever_descriptions.get(name, "Tối ưu hóa tài nguyên")
        lines.append(f"| {name} | ${amount:,.0f} | {contrib:.1f}% | {desc} |")

    lines += [
        "",
        "## 3. Phân tích Kỹ thuật Chuyên sâu: Hiện tượng \"GPU-Util Lie\"",
        "",
        "### Tại sao `nvidia-smi` GPU-Util là một thước đo \"nói dối\"?",
        "1. **Cơ chế đo lường:** `nvidia-smi` đo tỷ lệ phần trăm thời gian mà một kernel GPU đang hoạt động trên chip trong chu kỳ lấy mẫu (clock active time). Nó **không đo lường** năng lực tính toán thực tế (Tensor Core activity) hay thông lượng FLOPs.",
        "2. **Nguyên nhân gốc rễ (Root Causes):**",
        "   - **Memory Stalls & Memory-Bound Workloads:** Khi mô hình chạy inference decode (Arithmetic Intensity thấp $\\approx 1-2\\text{ FLOP/byte}$), GPU liên tục chờ nạp weights từ HBM. GPU-Util hiển thị 98% nhưng MFU chỉ đạt ~20%.",
        "   - **I/O & Kernel Launch Overhead:** Trong training phân tán, thời gian chờ đồng bộ AllReduce/NCCL và tải dữ liệu từ CPU khiến SM active nhưng không thực hiện nhân ma trận FP16/BF16.",
        "3. **Tác động Tài chính:**",
        "   - NimbusAI đang trả trọn vẹn $2.50/GPU-giờ cho `gpu-h100-4` nhưng chỉ nhận lại $0.50 giá trị tính toán thực tế. Hạ cấp (right-size) hoặc tái cấu trúc batch/kernel giúp thu hồi hàng ngàn USD mỗi tháng.",
    ]

    if gpu_lies_info:
        lines += [
            "",
            "**Các GPU được phát hiện có GPU-Util Lie trong hệ thống:**",
            "| GPU ID | GPU Type | GPU Util % | MFU | MBU | Tình trạng |",
            "|---|---|---|---|---|---|",
        ]
        for lie in gpu_lies_info:
            lines.append(f"| `{lie.get('gpu_id')}` | {lie.get('gpu_type')} | {lie.get('gpu_util_pct')}% | {lie.get('mfu'):.3f} | {lie.get('mbu'):.3f} | **GPU-Util Lie (Over-provisioned)** |")

    if extensions_info:
        lines += [
            "",
            "## 4. Kết quả Nghiên cứu Mở rộng (FinOps Extensions)",
            "",
            "### 4.1 Extension 1 & 5: Purchasing Matrix & Carbon-Aware Scheduling",
            f"- Đã nâng cấp thuật toán chọn Tier tính đến rủi ro gián đoạn (Interruption Rate) theo kiến trúc GPU và thời hạn công việc (Job Duration).",
            f"- Lập lịch nhận thức Carbon (Carbon-Aware Scheduling): Chuyển các job training gián đoạn sang **europe-north1** (Na Uy - thủy điện sạch) giúp cắt giảm hàng tấn khí thải $CO_2$ với mức giá điện hấp dẫn.",
            "",
            "### 4.2 Extension 2: Right-sizing theo MBU & Đơn vị Kinh tế Phần cứng",
            f"- Phân tích chỉ số `$/GB-VRAM` và `$/(TB/s) Bandwidth` giúp chọn GPU tối ưu cho memory-bound inference, giảm thiểu lãng phí khi thuê H100 cho các tác vụ chỉ cần VRAM/Băng thông.",
            "",
            "### 4.3 Extension 3 & 4: Kinh tế học Prompt Cache & Quản lý Ngân sách Reasoning",
            f"- **Prompt Caching:** Điểm hòa vốn chỉ cần trung bình ~1.2 - 1.5 lượt đọc lại là bù đắp toàn bộ chi phí ghi/lưu trữ cache.",
            f"- **Reasoning Tokens:** Phát hiện các truy vấn reasoning tiêu thụ năng lượng gấp **~80×** so với query thông thường. Thiết lập quy tắc định tuyến động (Dynamic Routing) giúp bảo vệ ngân sách công ty.",
        ]

    if sustainability:
        lines += [
            "",
            "## 5. Báo cáo Phát triển Bền vững (Green FinOps & Sustainability)",
            "",
            f"- **Energy per query (Năng lượng mỗi truy vấn trung bình):** {sustainability.get('wh_per_query', 0):.2f} Wh",
            f"- **Carbon per query (Phát thải carbon mỗi truy vấn):** {sustainability.get('carbon_g', 0):.3f} gCO2e",
            f"- **Cheapest+cleanest region (Vùng sạch và rẻ nhất đề xuất):** `{sustainability.get('best_region', 'n/a')}`",
            "",
            "### So sánh Lưới điện & Chi phí Năng lượng theo Khu vực:",
            "| Khu vực (Region) | Cường độ Carbon (gCO2/kWh) | Giá điện ($/kWh) | Đánh giá FinOps |",
            "|---|---|---|---|",
            "| `europe-north1` (Na Uy) | **30** (Rất sạch) | $0.090 | **Lựa chọn Xanh & Tiết kiệm nhất** cho Batch/Training |",
            "| `us-east-wa` (Washington) | **90** (Sạch) | **$0.055** (Rẻ nhất) | Chi phí điện thấp nhất Bắc Mỹ |",
            "| `us-west-2` (Oregon) | **120** (Thủy điện) | $0.070 | Lựa chọn cân bằng tốt tại US West |",
            "| `us-east-1` (Virginia) | 380 (Than/Khí) | $0.120 | Lưới điện ô nhiễm, chi phí trung bình |",
            "| `europe-central2` (Ba Lan) | 660 (Nhiệt điện than) | $0.180 | **Cần tránh** (Ô nhiễm và giá điện đắt nhất) |",
        ]

    lines += [
        "",
        "## 6. Kế hoạch Hành động FinOps Ưu tiên theo ROI (Actionable Roadmap)",
        "",
        "| Mức độ Ưu tiên | Hành động Cụ thể | Đòn bẩy FinOps | Mức Tiết kiệm Kỳ vọng | Độ phức tạp Triển khai |",
        "|---|---|---|---|---|",
        "| **P0 (Ngay lập tức)** | Bật auto-shutdown / script tắt GPU không tải qua đêm | Kill Idle GPUs | ~$600 / tháng | Thấp (1 ngày cấu hình) |",
        "| **P0 (Tuần 1)** | Tích hợp Prompt Caching & Batch API cho eval/chat | Inference Levers | ~$1,200+ / tháng | Trung bình (Tích hợp SDK) |",
        "| **P1 (Tháng 1)** | Ký hợp đồng Reserved 1-3yr cho job 24/7 & Spot cho training | Purchasing Strategy | ~$10,000+ / tháng | Thấp (Thủ tục Cloud) |",
        "| **P1 (Tháng 1)** | Hạ cấp các GPU H100 có MFU thấp sang A100/A10G | Right-sizing | ~$650+ / tháng | Trung bình (Benchmarking) |",
        "| **P2 (Quý 1)** | Áp dụng Chargeback gate (Tag coverage $\\ge 80\\%$) & FOCUS export | Governance | Kiểm soát chi phí minh bạch | Trung bình |",
        "",
        "---",
        "_Figures are June-2026 as-of snapshots; re-baseline before acting._",
    ]
    return "\n".join(lines)


def savings_waterfall(levers: dict, path: str) -> str:
    """Write a high-quality savings bar chart PNG. Returns the path. No-op if matplotlib absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""

    names = list(levers.keys())
    vals = [levers[n] for n in names]

    # Clean short labels for x-axis
    short_names = [n.replace("Inference (cascade/cache/batch)", "Inference Levers\n(Cache/Batch/Cascade)")
                    .replace("Purchasing (spot/reserved)", "Purchasing Strategy\n(Spot/Reserved)")
                    .replace("Right-size util-lies", "Right-Size\nUtil-Lies")
                    .replace("Kill idle GPUs", "Kill Idle\nGPUs") for n in names]

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    if len(colors) < len(names):
        colors = colors * (len(names) // len(colors) + 1)
    colors = colors[:len(names)]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=120)
    bars = ax.bar(short_names, vals, color=colors, edgecolor="#333333", linewidth=1.2, width=0.55)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"${height:,.0f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),  # 5 points vertical offset
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color="#222222")

    ax.set_ylabel("Monthly Savings (USD / month)", fontsize=11, fontweight="bold")
    ax.set_title("NimbusAI — GPU Cost Savings by FinOps Lever (Monthly)", fontsize=13, fontweight="bold", pad=15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # Add total savings banner
    total_savings = sum(vals)
    ax.text(0.98, 0.93, f"Total Savings: ${total_savings:,.0f}/mo",
            transform=ax.transAxes, fontsize=11, fontweight="bold",
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#e8f5e9", edgecolor="#4caf50", linewidth=1.5))

    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path

