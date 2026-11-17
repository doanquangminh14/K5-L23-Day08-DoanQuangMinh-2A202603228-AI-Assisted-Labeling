# Báo cáo Day 08 — 2A202603228 - Đoàn Quang Minh

> Một file duy nhất cho cả bài (đã gộp recipe, QC và ghi chú cá nhân). Mọi con số kèm **mẫu số** và nguồn (file nào). Viết bằng lời của bạn.

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| person / bicycle / motorbike / car / bus / truck | YOLO26m | person = 0.15, motorbike = 0.15; car / bicycle / bus / truck = 0.3 (mặc định) | YOLO26 ổn với 6 lớp COCO (r01, r02). Ở 0.5 mất vật thật nhiều hơn số rác lọc được (r02: img_007, img_011 mất người / xe máy ở xa) nên dùng ngưỡng thấp. Hạ person và motorbike xuống 0.15 vì xoá một box thừa chỉ mất một click, còn vẽ thêm một box bị sót tốn công hơn nhiều, nên đổi một ít precision lấy recall là đáng (số cụ thể bên dưới). Car giữ 0.3 vì precision của car gần như không đổi ở mọi ngưỡng (0.47–0.58): lỗi của car là tuk-tuk bị gọi thành car, đổi ngưỡng không sửa được. Bicycle (5 vật), bus (1), truck (0) quá ít vật trong gold để chọn ngưỡng riêng, giữ mặc định |
| three_wheeler | YOLOE-26s-seg, prompt `--classes three_wheeler` (r03, r06; khớp `work/runs.csv`) | 0.15 | YOLO26 không có lớp xe ba bánh: gọi tuk-tuk / xích lô là truck, car, motorbike, bicycle hoặc bỏ sót (r01, r04). YOLOE tìm đúng tuk-tuk img_002 (0.93) và xích lô img_032 (0.92, 0.48, 0.27) (r03). Ở 0.15 recall trên gold 100% (7/7), precision 78% (7 đúng / 9 box) |

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

Số của bản nhãn cuối (sau QC vòng 2), loại 22 đối tượng quanh bẫy:

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted | 570 | 570/690 = 82.6% | 690 box máy |
| edited | 25 | 25/690 = 3.6% | 690 box máy |
| deleted | 92 | 92/690 = 13.3% | 690 box máy |
| added | 52 | 52/648 = 8.0% | 648 đối tượng sau duyệt |

Trong deleted + added có 12 trường hợp là **đổi lớp** (box đúng chỗ, máy gán sai lớp); số vật máy **sót thật** ≈ 52 − 12 = 40.

- Lớp nào máy tệ nhất? Vì sao (dẫn ảnh)?
  - **`three_wheeler`**: 21/41 box máy bị xoá (51%), chỉ 15/41 giữ nguyên. YOLOE gọi các vật hình hộp không phải xe ba bánh là `three_wheeler`, và đây là **lỗi lặp**: xe tải thùng kín ở img_003 (YOLO26 nói đúng `truck` 0.87 nhưng luật gộp cho `three_wheeler` 0.61 thắng), quang gánh ở img_014 (0.63), minibus ở img_018 (0.53, phát hiện ở QC vòng 2).
  - **`truck`**: 6/6 box máy bị xoá, 0 giữ nguyên. Chủ yếu là xe ba bánh / xe van bị YOLO26 gọi `truck` (ví dụ xe van chở khách ở img_013 → sửa thành `car`).
  - `person` và `motorbike` nhiều box nhất nhưng ổn: 340/378 và 159/198 box giữ nguyên; lỗi chủ yếu là box thừa ở vật nhỏ, xa, và người/xe nhoè ở ảnh đêm img_022.
- So với gold: precision **74.7%** (65/87 box máy; KTC 95%: 65–83%), recall **75.6%** (65/86; KTC 95%: 66–83%). Mẫu số: **86 đối tượng gold** trên 4 ảnh. Số này lạc quan vì ngưỡng được chọn trên chính 4 ảnh gold (mục 1).

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy:
  - **Lần duyệt đầu: 5/12 = 41.7%** (KTC 95%: 19–68%), dưới ngưỡng 80% (lưu ở `qc/review_stats_lan1.md`). Lọt 7 bẫy: **loose 0/3** (img_018, img_021, img_025), missing 2/3 lọt (img_003, img_014), extra 1/3 lọt (img_026), wrong_class 1/3 lọt (img_031).
  - **Sau duyệt lại: 10/12 = 83.3%** (KTC 95%: 55–95%). Số này **lạc quan và không phải số đo độc lập**: tôi chỉ mở `traps_key.json` **sau** lần duyệt đầu (sau khi có 5/12) để tìm vị trí bẫy lọt và truy nguồn 2 bẫy *missing*; lúc duyệt lại tôi đã biết ảnh nào và vật nào có bẫy.
  - 2 bẫy còn "lọt" (img_003, img_014, loại *missing*) **thật ra là bẫy có đáp án sai**: `inject_traps.py` xoá một box máy điểm ≥ 0.5 và chờ người vẽ lại đúng lớp đó, nhưng box bị xoá là `three_wheeler` **sai** của YOLOE (thực tế là xe tải và quang gánh). Nhãn của tôi đúng (`truck` ở img_003, không gán gì ở img_014). Tính lại: 12/12 (CAL-04, CAL-05).
  - Vì sao lần đầu lọt nhiều: tôi tập trung kiểm **lớp** và **có / không có box**, nhưng **không phóng to để soi độ ôm của box**, nên cả 3 bẫy loose (box lệch ~7 px) đều lọt; với vật máy không vẽ box, tôi không quét kỹ ảnh trước khi nhìn nhãn máy.
- Accepted cao mà bắt bẫy thấp nói gì? Lần đầu accepted 621/690 = 90.0% nhưng bắt bẫy chỉ 5/12: tôi đã **tin box máy quá nhiều** (thiên kiến tự động hoá), box nào trông "gần đúng" là giữ. Lỗi lệch vài pixel và lỗi bỏ sót **không hiện trên điểm số** (`conf` vẫn cao), nên chỉ bắt được khi chủ động phóng to và quét ảnh.

## 4. Máy có đáng dùng không?

- Thời gian (từ `time_log.csv`): gán tay **29 phút / 4 ảnh = 7.3 phút/ảnh** vs duyệt có pre-label **65 phút / 24 ảnh = 2.7 phút/ảnh** (nhanh hơn ~2.7 lần). Thiên lệch: tôi gán gold trước nên đã quen cảnh và quen guideline khi duyệt; ảnh batch cũng khác ảnh gold về độ đông. Thời gian duyệt lại sau khi biết bẫy lọt và sửa lỗi QC **không** nằm trong 65 phút, nên con số duyệt thực tế cao hơn.
- Kết luận: **có, cho `person`, `motorbike`, `car`, `bicycle`** (accepted 340/378, 159/198, 50/54; recall trên gold 75.6%); **chỉ dùng như gợi ý cho `three_wheeler` và `truck`** (51% và 100% box máy bị xoá) — mọi box của 2 lớp này phải được kiểm riêng.
- **Ba quy tắc giảm thiên kiến của tôi:**
  1. **Quét ảnh trước khi bật nhãn máy**: đếm nhanh số xe ba bánh / người / xe máy rõ trong ảnh, rồi so với số box máy; lệch thì tìm box sót. Kiểm được: số bẫy *missing* bắt được.
  2. **Phóng to từng box có cạnh mờ / vật tối / vật nhỏ** và kéo cạnh cho ôm sát (≤ 2 px). Kiểm được: bẫy *loose* phải đạt 3/3 (lần đầu 0/3).
  3. **Kiểm riêng mọi box `three_wheeler` và `truck`** (lọc theo lớp trong CVAT), không dựa vào `conf`: box 0.63 có thể sai (img_014), box 0.27 có thể đúng (img_032). Kiểm được: số box `three_wheeler` / `truck` bị xoá hoặc đổi lớp sau duyệt.

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| img_032 (xích lô) | SAM 2.1 base (`sam2.1_b.pt`) | 3 (2 dương, 1 âm) | Không gồm người đàn ông đứng sát xe (click âm loại sạch); xe không chở hàng | Gần như không lỗi: đủ mui, nệm, khung, cả nan hoa; không lem sang xích lô bên cạnh. Thiếu phần yên / bánh sau bị mui che | `xichlo_img032.jpg` |
| img_013 (tuk-tuk) | SAM 2.1 base | 3 (2 dương, 1 âm) | Tài xế phần lớn bị loại nhưng còn lốm đốm; **lem sang người ngồi phía sau** nhìn qua khung | Đủ cả đuôi xe (box YOLO26 ở r04 cắt mất phần này); mask gồm cả khoảng trống bên trong cabin | `tuktuk_img013.jpg` |
| img_013 (tuk-tuk) | SAM 2.1 base | 5 (2 dương, 3 âm) | Loại sạch người | **Tệ hơn**: mask co còn mui + viền (12% ảnh so với 45%, 2 polygon) vì click âm lên người nằm lọt trong khung xe làm SAM bỏ cả thân xe | `tuktuk_img013_5click.jpg` |
| img_003 (xe ba gác) | SAM 2.1 base | 4 (3 dương, 1 âm) | **Gồm cả hàng** (dưa, rổ, túi) và **gồm em bé ngồi trên xe** dù đã click âm lên bé; người lớn đeo khẩu trang không bị tô | 2 polygon (mui tách khỏi thùng qua thanh chống mảnh) | `bagac_img003.jpg` |

- Đối chiếu guideline mục 4.2: quy tắc 4.2.4 (hai vật chạm nhau, mỗi vật một polygon, không chồng) giúp quyết định bỏ người khỏi mask xe; quy tắc 4.2.2 (lỗ thủng) khớp với khoảng trống trong khung xích lô / tuk-tuk. **Thiếu**: v1.0 không nói hàng chở trên xe và người ngồi trên xe có thuộc mask không → đã chốt ở v1.1 (CAL-01: hàng trong thùng thuộc xe; CAL-02: người không thuộc mask xe, phải sửa tay bằng eraser). Kết luận: SAM tách tốt người **đứng ngoài** xe, không tách được người **ngồi trong / trên** xe; thêm click âm có thể làm mask hỏng.
- *(Tuỳ chọn)* Interactor SAM trong CVAT: không thử; chỉ dùng `sam2_points.py` offline.

## 6. Calibration (Part 6)

- Số quyết định: **7** (`calibration_decisions.csv`: 3 `cau_hoi_mo`, 3 `loi_may`, 1 `can_review` - CAL-07: box person bị che trong ảnh đêm img_022). Quan trọng nhất:
  - **CAL-05** (img_003): luật gộp "three_wheeler thắng car/truck khi IoU ≥ 0.6" đã thay `truck` 0.87 đúng bằng `three_wheeler` 0.61 sai → đề xuất chỉ cho thắng `car`, hoặc thắng `truck` khi điểm YOLOE cao hơn.
  - **CAL-03** (img_014, img_032): không đặt mức `conf` tự chấp nhận, vì có box 0.63 sai và box 0.27 đúng.
- Quy tắc mới đã vào `guideline_v1.1_calibrated.md`: chốt câu mở 3 (ngưỡng hàng nhô 20% áp cho cả mask), 4 (người trong/trên xe không thuộc mask xe), 7 (không đặt conf tự chấp nhận); thêm quang gánh / sạp hàng vào *Không gồm* của `three_wheeler`; kiểm `truck` khi máy gợi ý `three_wheeler` (đếm bánh, nhìn cabin); tuk-tuk / xích lô luôn là `three_wheeler`. Câu mở 1, 2, 5, 6 chưa chốt (chưa gặp ca đủ rõ).
- Nguyên nhân gốc: chủ yếu **máy** (YOLO26 không có lớp xe ba bánh; YOLOE nhận nhầm vật hình hộp; luật gộp ưu tiên YOLOE) và **guideline mơ hồ** về hàng hoá / người trên xe ở mask. Lỗi của tôi (bẫy loose lọt) là do **không làm đủ checklist**, không phải do guideline.

## 7. QC và quyết định (Part 6)

- **Vòng 1** (`qc/vong1_*`): mẫu **7 ảnh** (2 ca khó + 5 ngẫu nhiên, seed 42), 174 đối tượng; người kiểm: tôi (**hạn chế**: tôi vừa là người duyệt).
  - Lỗi theo **đối tượng**: 4/174 = 2.3% · theo **ảnh**: **3/7 = 42.9%** (ngưỡng Release ≤ 2%); cận trên 95%: 75%.
  - Lỗi: `thua` 2 (img_022 #311 box person lớn bao rổ hàng ở ảnh đêm; img_035 #660 box thứ hai trên cùng một xích lô), `sai_lop` 1 (img_013 #151 xe van chở khách gán `truck`), `sai_vi_tri_kich_thuoc` 1 (img_035 #650 box xích lô thiếu phần mui). Đã sửa hết trong CVAT, build lại release.
- **Vòng 2** (`qc/qc_*`): mẫu **6 ảnh** (1 ca khó + 5 ngẫu nhiên, **seed 2026** để lấy mẫu mới), 175 đối tượng.
  - Lỗi theo **đối tượng**: 1/175 = 0.6% · theo **ảnh**: **1/6 = 16.7%**; cận trên 95%: 56%.
  - Lỗi: `sai_lop` 1 (img_018 #229 minibus gán `three_wheeler`). Đã sửa thành `bus` sau QC, build lại release.
- **Giới hạn mẫu nhỏ**: với 6–7 ảnh, kể cả 0 lỗi thì cận trên vẫn khoảng 3/n ≈ 40–50%; ngưỡng 2% **không thể chứng minh**. Vòng 2 còn lạc quan vì 4/6 ảnh **trùng với vòng 1** (batch chỉ 24 ảnh) và đã được sửa; chỉ 2 ảnh mới (img_018, img_027), trong đó 1 ảnh có lỗi.
- Script kiểm được / không kiểm được: script chỉ **đếm** theo phiếu tôi điền và so box với bẫy / gold bằng IoU. Script **không** biết box có ôm sát hay không nếu không có đáp án, **không** phát hiện lỗi mà cả máy lẫn tôi cùng bỏ sót, và chấm sai khi đáp án bẫy kế thừa lỗi máy (img_003, img_014). Việc tôi có thật sự theo checklist hay không cũng không kiểm được bằng script.

**Quyết định: Rework**, ngày 2026-11-17. Lý do (đi theo Hold → Rework → Release): không **Hold** vì guideline đã được chốt ở v1.1 cho các ca gặp phải và dữ liệu không hỏng; **Rework** vì tỉ lệ ảnh lỗi ở cả 2 vòng (3/7 và 1/6) vượt xa ngưỡng 2%, và có **lỗi lặp** của máy ở lớp `three_wheeler` (img_003, img_014, img_018); chưa thể **Release** vì mẫu nhỏ không chứng minh được ≤ 2%.
Nếu Rework / Hold, việc cần làm tiếp: (1) duyệt lại **toàn bộ box `three_wheeler`, `truck`, `bus`** trên 24 ảnh theo quy tắc 3 ở mục 4; (2) phóng to kiểm độ ôm các box ở ảnh đông / ảnh đêm; (3) sửa luật gộp theo CAL-05 rồi chạy lại pre-label cho lô sau; (4) QC vòng 3 với mẫu lớn hơn (ví dụ `--n 10`, seed mới) và ghi kết quả.

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài**: Claude Code (trong VS Code) để: đọc ảnh viz và gợi ý nhận xét cho `model_log.csv`; phân tích bảng `threshold_sweep` để chọn ngưỡng; kiểm tra file export CVAT, chạy `validate_labels` / `review_stats` / `build_release` / `qc_report`; truy nguồn 2 bẫy *missing* (img_003, img_014); chạy và nhận xét mask SAM; soạn nháp `calibration_decisions.csv`, `guideline_v1.1`, `qc_review.csv` và báo cáo này. Kiểm tra: tôi tự mở lại ảnh trong CVAT / `work/qc/review/` để xác nhận từng lỗi trước khi sửa, và tự sửa nhãn trong CVAT.
- Gold có bị nhiễm không? **Không**: 4 ảnh gold gán tay trong CVAT trước khi chạy model, không xem pre-label, không hỏi AI (xem `time_log.csv`).
- Máy / GPU / CVAT dùng: Windows 11, NVIDIA RTX 3050 Laptop 6 GB (CUDA), CVAT v2.74.1 chạy Docker local. Vướng mắc: chạy script bằng conda `base` bị thiếu `numpy` → chuyển sang env `sam3`; PowerShell không chạy được `validate_labels.py` nếu thiếu `python` và đường dẫn `scripts/`.
- Dùng cache `data/cache/`? **Không**, tự chạy YOLO26 và YOLOE (r01–r06 trong `model_log.csv`).
- Lỗi / bug script đã gặp: bẫy *missing* của `inject_traps.py` lấy box máy điểm ≥ 0.5 làm đáp án mà không kiểm box đó đúng hay sai → 2 bẫy có đáp án sai (img_003, img_014), `review_stats` chấm là lọt dù nhãn đúng.
- Một điều tôi làm khác lần sau: quét ảnh và phóng to kiểm cạnh box **ngay lần duyệt đầu** thay vì duyệt nhanh rồi sửa sau khi xem bẫy lọt; và ghi thời gian các vòng sửa vào `time_log.csv`.
