# Báo cáo Day 08 — 2A202603228 - Đoàn Quang Minh

> Một file duy nhất cho cả bài (đã gộp recipe, QC và ghi chú cá nhân). Mọi con số kèm **mẫu số** và nguồn (file nào). Viết bằng lời của bạn.

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| person / bicycle / motorbike / car / bus / truck | YOLO26m | person = 0.15, motorbike = 0.15; car / bicycle / bus / truck = 0.3 (mặc định) | YOLO26 ổn với 6 lớp COCO (r01, r02). Ở 0.5 mất vật thật nhiều hơn số rác lọc được (r02: img_007, img_011 mất người / xe máy ở xa) nên dùng ngưỡng thấp. Hạ person và motorbike xuống 0.15 vì xoá một box thừa chỉ mất một click, còn vẽ thêm một box bị sót tốn công hơn nhiều, nên đổi một ít precision lấy recall là đáng (số cụ thể bên dưới). Car giữ 0.3 vì precision của car gần như không đổi ở mọi ngưỡng (0.47–0.58): lỗi của car là tuk-tuk bị gọi thành car, đổi ngưỡng không sửa được. Bicycle (5 vật), bus (1), truck (0) quá ít vật trong gold để chọn ngưỡng riêng, giữ mặc định |
| three_wheeler | YOLOE-26s-seg, prompt: "tuk-tuk", "cyclo", "cargo tricycle" | 0.15 | YOLO26 không có lớp xe ba bánh: gọi tuk-tuk / xích lô là truck, car, motorbike, bicycle hoặc bỏ sót (r01, r04). YOLOE tìm đúng tuk-tuk img_002 (0.93) và xích lô img_032 (0.92, 0.48, 0.27) (r03). Ở 0.15 recall trên gold 100% (7/7), precision 78% (7 đúng / 9 box) |

Số đổi được khi hạ ngưỡng từ 0.3 xuống 0.15 (từ `qc/threshold_sweep.csv`, so với gold):

| Lớp (số vật gold) | Ở 0.30 | Ở 0.15 | Đổi được |
|---|---|---|---|
| person (42) | 26 đúng, 8 thừa | 32 đúng, 12 thừa | +6 vật thật, +4 box thừa |
| motorbike (23) | 10 đúng, 1 thừa | 14 đúng, 4 thừa | +4 vật thật, +3 box thừa |

0.15 là điểm hợp lý nhất để dừng: hạ person xuống 0.10 chỉ bắt thêm 1 người (33 đúng) nhưng thêm 5 box thừa (17 thừa).

- Lệnh `merge_prelabels.py` đã dùng (dán nguyên):
  ```bash
  python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --thr "person=0.15,motorbike=0.15,three_wheeler=0.15" --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json
  ```
- **Phiên bản SAM bạn chọn** (SAM 1 / 2.1 / 3) và lý do: SAM 2.1 base (`sam2.1_b.pt`), vì tải về dùng được ngay, không cần xin quyền như SAM 3. Tốc độ 2.12 s/ảnh trên RTX 3050 (r04) đủ cho pre-label. Khi box đúng, mask bám sát vật: ở img_004, mask người 0.93 và mask tuk-tuk xanh lá không lem ra nền hay sang xe bên cạnh. Giới hạn: mask phụ thuộc hoàn toàn vào box. Ở img_013, box car bị cắt tại x=325 nên mask mất cả phần đuôi tuk-tuk.
- Lỗi nào **không sửa được bằng đổi ngưỡng** (ví dụ tuk-tuk bị gọi `car`)? Sửa thế nào? YOLO26 học trên COCO, không có lớp xe ba bánh, nên luôn gán tuk-tuk/xích lô vào lớp gần nhất: tuk-tuk ở img_004 và img_013 bị gọi car, xích lô ở img_032 bị gọi bicycle 0.69. Xe xanh ở mép trái img_011 (bị cắt nửa, có thể là xe ba bánh hoặc songthaew; ảnh thuộc pilot nên không có gold để đối chiếu) là truck 0.77 ở cả ngưỡng 0.25 lẫn 0.5. Đây là lỗi tên lớp, không phải lỗi điểm, nên đổi ngưỡng chỉ làm box biến mất chứ không đổi được nhãn. Cách sửa là dùng YOLOE (open-vocabulary, prompt three_wheeler) cho lớp này. Khi chồng nhau, box three_wheeler thắng box car/truck của YOLO26.
- Box **điểm cao mà sai** hoặc **điểm thấp mà đúng** (ảnh nào):
  - Điểm cao mà sai: bicycle 0.69 ở img_032 thực ra là một chiếc xích lô. YOLOE gán three_wheeler 0.37 cho xe bán tải trắng ở img_002, trong khi đó là truck.
  - Điểm thấp mà đúng: xích lô bên phải ở img_032 chỉ có three_wheeler 0.27 nhưng đúng là xích lô. Ở img_011, người đứng xa cuối phố chỉ có person 0.28–0.39 nhưng là người thật, và lên ngưỡng 0.5 thì bị mất.
  - Vì vậy điểm tin cậy không bảo đảm nhãn đúng, và ngưỡng không thay được bước duyệt bằng mắt.
- Giới hạn: chọn ngưỡng trên chính gold (4 ảnh) nên số đo máy trên gold là lạc quan: ngưỡng được chọn và chấm trên cùng 4 ảnh gold, và số vật rất ít (three_wheeler 7, bicycle 5, bus 1, truck 0). Vì vậy P/R ở trên là lạc quan, và recall 100% của three_wheeler khó giữ được trên toàn bộ dữ liệu.

## 2. AI error profile (từ `qc/review_stats.md`)

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted | <<>> | <<>> | <<box máy>> |
| edited | <<>> | <<>> | <<>> |
| deleted | <<>> | <<>> | <<>> |
| added | <<>> | <<>> | <<đối tượng sau duyệt>> |

- Lớp nào máy tệ nhất? Vì sao (dẫn ảnh)? <<>>
- So với gold: precision / recall của **máy** <<>> (mẫu số: <<đối tượng gold>>)

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy: <<.../12>>; loại nào lọt: <<>>; vì sao (bạn đã bỏ bước nào của checklist)? <<>>
- Accepted cao mà bắt bẫy thấp (hoặc ngược lại) nói gì về cách bạn duyệt? <<>>

## 4. Máy có đáng dùng không?

- Thời gian (từ `time_log.csv`): gán tay <<... phút/ảnh>> vs duyệt <<... phút/ảnh>>. Thiên lệch (bạn gán gold trước nên đã quen cảnh): <<>>
- Kết luận: <<có / không / chỉ cho lớp nào>>, dựa trên <<số liệu>>
- **Ba quy tắc giảm thiên kiến của bạn** (mỗi quy tắc kiểm tra được bằng số hoặc việc làm):
  1. <<>>
  2. <<>>
  3. <<>>

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| <<img_0xx>> | <<>> | <<>> | <<>> | <<>> | <<>> |

- Đối chiếu guideline mục 4.2: quy tắc nào giúp, quy tắc nào thiếu? <<>>
- *(Tuỳ chọn)* Interactor SAM trong CVAT: làm được / không (lỗi gì, đã thử gì); khác gì offline: <<>>

## 6. Calibration (Part 6)

- Số quyết định: <<>> (`calibration_decisions.csv`); quan trọng nhất (dẫn `CAL-xx`): <<>>
- Quy tắc mới đã vào `guideline_v1.1_calibrated.md`: <<>>
- Nguyên nhân gốc: <<guideline mơ hồ / chưa nắm quy tắc / máy>>

## 7. QC và quyết định (Part 6)

- Mẫu: <<n>> ảnh (<<m>> ảnh `can_review` + <<k>> ngẫu nhiên, seed 42); người kiểm: bạn (**hạn chế**: bạn vừa là người duyệt).
- Lỗi theo **đối tượng**: <<.../... = ...%>> · theo **ảnh**: <<.../... = ...%>> (ngưỡng Release ≤ 2%)
- Top lỗi (trong: bỏ sót, thừa, sai lớp, sai vị trí/kích thước, sai thuộc tính, sai mask) + một ảnh ví dụ mỗi loại: <<>>
- **Giới hạn mẫu nhỏ** (cận trên 95% trong `qc_report_auto.md`): <<>>
- Script kiểm được / không kiểm được (ví dụ ôm sát, cùng bỏ sót với máy, theo checklist thật): <<>>

**Quyết định: <<Release / Hold / Rework>>**, ngày <<YYYY-MM-DD>>. Lý do (đi theo Hold → Rework → Release): <<>>
Nếu Rework / Hold, việc cần làm tiếp: <<>>

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài** (công cụ nào, việc gì, kiểm tra thế nào; "không dùng" nếu không): <<>>
- Gold có bị nhiễm không (bạn có nhìn pre-label hay hỏi AI trước khi gán xong)? <<không / có, như sau: ...>>
- Máy / GPU / CVAT dùng; vướng mắc cài đặt: <<>>
- Dùng cache `data/cache/` thay vì tự chạy? <<không / có, vì ...>>
- Lỗi / bug script đã gặp (nếu có): <<>>
- Một điều bạn làm khác lần sau: <<>>
