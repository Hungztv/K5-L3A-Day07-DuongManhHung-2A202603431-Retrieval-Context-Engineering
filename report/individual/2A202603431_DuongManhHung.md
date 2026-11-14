# Báo cáo Day 7 — Agentic RAG

- **Học viên:** Dương Mạnh Hùng | **MSSV:** `2A202603431` | **Lớp:** `L3A`
- **Môi trường:** Windows, PowerShell 5.1, Python 3.12.10, không dùng Docker.

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:**
  - Tài liệu: *OpenStax Introduction to Sociology 3e* (`openstax-introduction-sociology-3e.pdf`, CC BY 4.0, OpenStax).
  - Số trang: 664 trang vật lý / 662 trang nội dung (trang 3–664).
  - Số chunk: **2.726 chunks** (kế thừa từ Strategy B `by_paragraph`, `max_size=800` ở Day 6), export sang [chunks.json](../../starter/data/converted/chunks.json) bằng [src/export_day6.py](../../src/export_day6.py).
- **Vector store, embedding model, phiên bản và số chiều:**
  - Vector store: `qdrant-client==1.20.0` local `PersistentClient` tại `starter/qdrant_db`, collection `strategy_B`, metric `cosine`.
  - Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (384 chiều, L2-normalized, chạy CPU local).
- **Agent model, cách chạy và phân chia dev/test:**
  - Agent model: `openai/gpt-4o-mini` qua OpenRouter API (`base_url="https://openrouter.ai/api/v1"`), sử dụng tool calling `search_documents(query)` trong [src/openrouter_agent.py](../../src/openrouter_agent.py).
  - Phân chia benchmark ([questions_day7.json](../../starter/data/converted/questions_day7.json), 25 câu):
    - **Dev (17 câu):** 15 câu có đáp án từ Day 6 (`q01`–`q15`) + 2 câu thiếu bằng chứng (`q22`, `q23`).
    - **Test (8 câu):** 6 câu có đáp án từ Day 6 (`q16`–`q21`) + 2 câu thiếu bằng chứng (`q24`, `q25`). Tập `test` Day 6 không được dùng để chỉnh cấu hình Day 7; toàn bộ tham số được chốt trên `dev` trước khi chạy `test`.
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**
  - **Câu có đáp án:** evidence trích nguyên văn nằm trọn trong một chunk đúng `source_id` và `page` (chuẩn hóa khoảng trắng và dấu nháy khi đối chiếu).
  - **Câu thiếu bằng chứng (`q22`–`q25`):** đã chạy [src/verify_absence.py](../../src/verify_absence.py) tìm từ khóa và đồng xuất hiện trên **toàn bộ 2.726 chunks** (kết quả lưu tại `starter/results/absence_check.json`):
    - `q22` (dev): `"Vietnam"` (16 chunks), `"pension"` (1 chunk, p.383), `"social insurance"` (0), `"2024"` (0); đồng xuất hiện `(Vietnam + social insurance/pension/2024)` = 0.
    - `q23` (dev): `"Gini"` (4 chunks), `"World Bank"` (27 chunks), `"2025"` (1 chunk, p.383); đồng xuất hiện `(Gini + 2025)` = 0, `(World Bank + 2025)` = 0.
    - `q24` (test): `"AI Act"` (0), `"emotion recognition"` (0), `"2024"` (0), đồng xuất hiện `(European Union + artificial intelligence)` = 0.
    - `q25` (test): `"burnout"` (0), `"remote work"` (0), `"Ministry of Manpower"` (0), đồng xuất hiện `(Singapore + burnout/2025)` = 0.
  - **Chấm thủ công:** theo [BENCHMARK.md](../../BENCHMARK.md) (Grounded Answer 0/1/2, Context 0/1/2, Citation support, Abstention) và tổng hợp bằng [starter/agent_metrics.py](../../starter/agent_metrics.py).

## Retrieval và thử nghiệm

| Cấu hình | Phương pháp, top-k, filter, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline (`dense_day6_k3_b800`) | Dense (Qdrant Day 6), `top_k=3`, `budget=800` | Kế thừa nguyên Day 6 + budget mặc định starter | dev | 4/15 (26.7%) | 14/15 (93.3%) | 0/15 (0.0%) | 12.52 ms/q |
| Phương pháp 2 (`tfidf_k3_b800`) | Lexical TF-IDF (`baseline.py`), `top_k=3`, `budget=800` | So sánh lexical thuần với dense | dev | 9/15 (60.0%) | 11/15 (73.3%) | 0/15 (0.0%) | 8.35 ms/q |
| Phương pháp 3 (`bm25_k3_b800`) | Lexical BM25 (`k1=1.5, b=0.75`), `top_k=3`, `budget=800` | BM25 chuẩn hóa độ dài tốt hơn TF-IDF | dev | 11/15 (73.3%) | 11/15 (73.3%) | 0/15 (0.0%) | 7.84 ms/q |
| Phương pháp 4 (`hybrid_rrf_k3_b800`) | Hybrid RRF (Dense + BM25, `fetch_k=20`), `top_k=3`, `budget=800` | Kết hợp ngữ nghĩa (Dense) và từ khóa (BM25) | dev | 12/15 (80.0%) | 15/15 (100.0%) | 0/15 (0.0%) | 26.32 ms/q |
| Cải thiện 1 (`dense_day6_k3_b2400`) | Dense (Qdrant Day 6), `top_k=3`, `budget=2400` | Tăng `budget` từ 800 lên 2400 để không cắt chunk 800 ký tự | dev | 4/15 (26.7%) | 14/15 (93.3%) | 11/15 (73.3%) | 21.81 ms/q |
| Cải thiện 2 (`hybrid_rrf_k3_b2400`) | Hybrid RRF, `top_k=3`, `budget=2400` | Dùng Hybrid RRF + `budget=2400` | dev | 12/15 (80.0%) | 15/15 (100.0%) | 14/15 (93.3%) | 33.56 ms/q |
| Cải thiện 3 (`hybrid_rrf_k5_b3200`) | Hybrid RRF, `top_k=5`, `budget=3200` | Mở rộng `top_k=5`, `budget=3200` để giữ đủ evidence | dev | **12/15 (80.0%)** | **15/15 (100.0%)** | **15/15 (100.0%)** | 35.66 ms/q |
| Cấu hình đã chọn (`hybrid_rrf_k5_b3200`) | Hybrid RRF, `top_k=5`, `budget=3200` | Đánh giá độc lập trên test sau khi chốt | test | **3/6 (50.0%)** | **5/6 (83.3%)** | **5/6 (83.3%)** | 24.44 ms/q |
| Baseline (`dense_day6_k3_b2400`) | Dense (Qdrant Day 6), `top_k=3`, `budget=2400` | Đối chiếu baseline trên test | test | 2/6 (33.3%) | 4/6 (66.7%) | 3/6 (50.0%) | 13.00 ms/q |

**Phân tích 2 lỗi dev và lý do chọn cấu hình:**
1. **Lỗi `budget=800` và `q03` ở Dense:** Chunk Strategy B có `max_size=800` ký tự, cộng header nguồn (~70 ký tự) > 800 nên `pack_context(..., budget=800)` loại bỏ toàn bộ chunk (`CtxEvid=0%`). Khi tăng `budget=2400` ở Dense, `q03` vẫn bị từ chối nhầm ở cả `agent-dev-dense` và `agent-dev-dense-final` do chunk đúng (`p0031-paragraph-0000`) chỉ đứng hạng 3 và bị cắt khỏi `budget=2400` sau 2 chunk đầu. Chuyển sang **Hybrid RRF (`top_k=5, budget=3200`)** đã đưa `p0031-paragraph-0000` lên hạng 1 và sửa dứt điểm `q03`.
2. **Lỗi filter `q12`, `q15` trên dev (và `q19`, `q21` trên test):** Các câu này có `filter={"document_type": "textbook"}`, nhưng payload Qdrant `strategy_B` kế thừa từ Day 6 chỉ có `['char_count', 'chunk_strategy', 'has_text', 'strategy']`, không index `document_type` nên truy vấn lọc trả về rỗng (`[]`). Sau khi cập nhật [src/retrievers.py](../../src/retrievers.py) để bỏ qua filter key chưa được index trong corpus (vì toàn bộ corpus là sách giáo khoa), `Hybrid RRF (k=5, budget=3200)` tăng `EvidHit@3` dev từ `13/15 (86.7%)` lên `15/15 (100.0%)`. *Giới hạn:* nếu sau này corpus gộp nhiều loại tài liệu, cần re-index bổ sung trường `document_type` vào Qdrant payload.

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:**
  - Dùng 1 tool `search_documents(query)` kết hợp vòng lặp `run_agent` (`max_searches=4`), dừng khi model trả câu trả lời kèm `[chunk-id]` hợp lệ hoặc chạm giới hạn 4 lượt tìm (tự động `abstain`).
  - Context đóng gói qua `pack_context` (deduplicate theo `chunk_id`, giữ thứ tự score trong `budget`). Lỗi API/retriever ghi riêng với `status="error"`, không dùng mock.
  - Chi phí tính theo giá công bố của `openai/gpt-4o-mini` trên OpenRouter: `\$0.15 / 1M` input tokens và `\$0.60 / 1M` output tokens.

| Cấu hình / split | Tool use | Số lượt gọi (phân bố) | Grounded answer 0/1/2 (TB) | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm | Tokens / Chi phí / Thời gian |
|---|---|---|---|---|---|---|---|---|
| `agent-dev-dense-final` (`k=3, b=2400`) / `dev` | 17/17 | 1: 13; 2: 1; 4: 3 | `0:2, 1:0, 2:13` (**1.73/2**) | 20/20 (100%) | 13/13 | 2/2 (100%) | 2/15 (13.3%) | 29.219 tok / \$0.00546 / 72.70s |
| `agent-dev-hybrid-final` (`k=5, b=3200`) / `dev` | 17/17 | 1: 14; 2: 1; 4: 2 | `0:1, 1:0, 2:14` (**1.87/2**) | 20/20 (100%) | 14/14 | 2/2 (100%) | 1/15 (6.7%) | 32.384 tok / \$0.00589 / 64.60s |
| Baseline `agent-test-dense` (`k=3, b=2400`) / `test` | 8/8 | 1: 5; 4: 3 | `0:1, 1:1, 2:4` (**1.50/2**) | 5/5 (100%) | 5/5 | 2/2 (100%) | 1/6 (16.7%) | 17.098 tok / \$0.00300 / 39.44s |
| **Đã chọn `agent-test-hybrid` (`k=5, b=3200`) / `test`** | **8/8** | 1: 6; 4: 2 | `0:0, 1:1, 2:5` (**1.83/2**) | **7/7 (100%)** | **6/6** | **2/2 (100%)** | **0/6 (0.0%)** | 18.817 tok / \$0.00329 / 34.06s |

### Đánh giá Context (0/1/2) cho 6 câu `test` có đáp án

| Query ID | Baseline `agent-test-dense` (Context 0/1/2 & lý do) | Đã chọn `agent-test-hybrid` (Context 0/1/2 & lý do) |
|---|---|---|
| `q16` | **2** — Có `p0380-paragraph-0002` nêu rõ Millennials coi 59 tuổi là già + nguồn | **2** — Có `p0380-paragraph-0002` (đứng #1) + nguồn |
| `q17` | **1** — Có `p0380-paragraph-0002` (mốc 31/36), thiếu bảng `TABLE 13.1` (`p0381-paragraph-0000`) chứa số 50 | **1** — Có `p0380-paragraph-0002`, `p0381-paragraph-0000` đứng #4 nhưng bị cắt sau 3 chunk đầu bởi `budget=3200` |
| `q18` | **2** — Có `p0141-paragraph-0000` đầy đủ Nhật đô hộ Triều Tiên 1910–1945 + nguồn | **2** — Có `p0141-paragraph-0000` + nguồn |
| `q19` | **2** — Có `p0311-paragraph-0000` định nghĩa miscegenation và luật cấm đến 1967 (evidence thay thế hợp lệ) | **2** — Có cả `p0311-paragraph-0000` và `p0301-paragraph-0000` + nguồn |
| `q20` | **2** — Có `p0171-paragraph-0000` đầy đủ nghiên cứu Jerry Newman + nguồn | **2** — Có `p0171-paragraph-0000` + nguồn |
| `q21` | **0** — Chỉ lấy được `p0056-paragraph-0000`, `p0049-paragraph-0001` (không định nghĩa biến độc lập IV) | **2** — Có `p0050-paragraph-0002` & `p0050-paragraph-0004` định nghĩa đầy đủ vai trò IV/DV và ví dụ |
| **Tổng** | `0:1, 1:1, 2:4` (TB **1.50/2**) | `0:0, 1:1, 2:5` (TB **1.83/2**) |

### Chi tiết đánh giá câu trả lời 8 câu `test` (cấu hình đã chọn `agent-test-hybrid`)

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| `q16` | Có đáp án | 1 lượt search, lấy `p0380-paragraph-0002` | Trả lời đúng Millennials coi 59 tuổi là già `[p0380-paragraph-0002]` | `grade=2`, cite `[true]` |
| `q17` | Có đáp án (bảng p.381) | 1 lượt search, `p0381-paragraph-0000` bị cắt khỏi budget | Nêu rõ context không có tuổi cụ thể của Boomers, chỉ nêu mốc 31/36 `[p0380-paragraph-0002]` | `grade=1`, cite `[true]` (không bịa số, nhưng thiếu đáp án 50) |
| `q18` | Có đáp án | 1 lượt search, lấy `p0141-paragraph-0000` | Trả lời đúng Nhật đô hộ Triều Tiên 1910–1945 và tranh cãi SGK `[p0141-paragraph-0000]` | `grade=2`, cite `[true]` |
| `q19` | Có đáp án | 1 lượt search, lấy `p0311-paragraph-0000`, `p0301-paragraph-0000` | Giải thích đúng miscegenation và luật cấm đến vụ Loving v. Virginia 1967 `[p0311-paragraph-0000]` | `grade=2`, cite `[true]` |
| `q20` | Có đáp án | 1 lượt search, lấy `p0171-paragraph-0000` | Tổng hợp đúng kết quả nghiên cứu Jerry Newman tại 7 cửa hàng fast food `[p0171-paragraph-0000]` | `grade=2`, cite `[true]` |
| `q21` | Có đáp án | 1 lượt search, lấy `p0050-paragraph-0002`, `p0050-paragraph-0004` | Giải thích đúng vai trò biến độc lập (IV) là nguyên nhân gây thay đổi và ví dụ `[p0050-paragraph-0002][p0050-paragraph-0004]` | `grade=2`, cite `[true, true]` (baseline `dense` từ chối nhầm) |
| `q24` | Thiếu bằng chứng | 4 lượt search (`EU AI Act 2024`), không có trong sách | Trả lời `Không đủ bằng chứng để trả lời.` | `abstention_correct=true` |
| `q25` | Thiếu bằng chứng | 4 lượt search (`Singapore 2025 burnout`), không có trong sách | Trả lời `Không đủ bằng chứng để trả lời.` | `abstention_correct=true` |

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:**
  - Test: [agent-test-dense/summary.json](../../starter/results/agent-test-dense/summary.json), [agent-test-hybrid/summary.json](../../starter/results/agent-test-hybrid/summary.json) (kèm `traces.json`, `reviews.json`, `review_sheet.txt`).
  - Dev: `starter/results/agent-dev-dense-final/summary.json`, `starter/results/agent-dev-hybrid-final/summary.json`.
  - Lệnh chạy tái lập: xem mục Hướng dẫn chạy trong [README.md](../../README.md).
- **Kết quả trước/sau, đánh đổi và giới hạn:**
  - **Trước/sau:** Hybrid RRF (`k=5, budget=3200`) nâng Grounded Answer trung bình trên `test` từ **1.50/2** lên **1.83/2**, xóa bỏ lỗi từ chối nhầm (`1/6 -> 0/6`), giữ `100%` citation validity và `100%` abstention đúng.
  - **Đánh đổi & tính bất định:** Tăng `top_k` và `budget` làm prompt tokens tăng ~10% trên `test` (17.783 vs 16.133 input tokens), nhưng tổng thời gian giảm nhờ ít vòng lặp hơn (34.06s vs 39.44s). Dù đặt `temperature=0`, API LLM vẫn có dao động nhỏ giữa các lần chạy (ví dụ `q14` trên `dev` trả lời được ở lần chạy trước nhưng từ chối ở `agent-dev-hybrid-final`).
  - **Giới hạn:** Với `q17`, chunk bảng `p0381-paragraph-0000` đứng hạng 4 nhưng bị `budget=3200` cắt sau 3 chunk đầu (~800 ký tự/chunk); cần tăng `budget >= 4200` hoặc nén/rerank chunk trước khi đưa vào context.
