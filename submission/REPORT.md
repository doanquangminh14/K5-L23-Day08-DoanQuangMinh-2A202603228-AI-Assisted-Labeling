# Báo cáo Day 08 — 2A202603228 - Đoàn Quang Minh

> Một file duy nhất cho cả bài (đã gộp recipe, QC và ghi chú cá nhân). Mọi con số kèm **mẫu số** và nguồn (file nào).

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| person / bicycle / motorbike / car / bus / truck | YOLO26m | person 0.15, motorbike 0.15; car, bicycle, bus, truck để mặc định 0.3 | Với 6 lớp COCO thì YOLO26 làm ổn (r01, r02). Mình thử lên 0.5 thì mất người và xe máy ở xa ở img_007, img_011 nhiều hơn số box rác bỏ được, nên dùng ngưỡng thấp. Person và motorbike mình hạ xuống 0.15 vì lúc duyệt, xoá box thừa chỉ mất một click, còn vẽ thêm box bị sót thì lâu hơn nhiều. Car để 0.3 vì precision của car gần như đứng yên ở mọi ngưỡng (0.47 đến 0.58), lỗi chính của car là gọi tuk-tuk thành car, đổi ngưỡng không giải quyết được. Bicycle chỉ có 5 vật, bus 1, truck 0 trong gold nên không đủ cơ sở để chọn ngưỡng riêng |
| three_wheeler | YOLOE-26s-seg, `--classes three_wheeler` (r03, r06, khớp `work/runs.csv`) | 0.15 | YOLO26 không có lớp xe ba bánh nên tuk-tuk, xích lô bị gọi nhầm thành truck, car, motorbike, bicycle hoặc bị bỏ qua (r01, r04). YOLOE bắt được tuk-tuk ở img_002 (0.93) và xích lô ở img_032 (0.92, 0.48, 0.27) (r03). Ở ngưỡng 0.15, trên tập gold đạt recall 7/7, precision 7/9 = 78% |

Hạ ngưỡng từ 0.3 xuống 0.15 thì được và mất như sau (lấy từ `qc/threshold_sweep.csv`, so với gold):

| Lớp (số vật gold) | Ở 0.30 | Ở 0.15 | Chênh lệch |
|---|---|---|---|
| person (42) | 26 đúng, 8 thừa | 32 đúng, 12 thừa | thêm 6 vật thật, thêm 4 box thừa |
| motorbike (23) | 10 đúng, 1 thừa | 14 đúng, 4 thừa | thêm 4 vật thật, thêm 3 box thừa |

Mình dừng ở 0.15 vì nếu hạ person tiếp xuống 0.10 thì chỉ lấy thêm được 1 người (33 đúng) nhưng box thừa tăng vọt lên 17.

- Lệnh `merge_prelabels.py` đã dùng:
  ```bash
  python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --thr "person=0.15,motorbike=0.15,three_wheeler=0.15" --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json
  ```
- **Phiên bản SAM đã chọn** (SAM 1 / 2.1 / 3) và lý do: Mình dùng SAM 2.1 base (`sam2.1_b.pt`). Lý do chính là tải về chạy được ngay, còn SAM 3 phải xin quyền truy cập. Trên RTX 3050 chạy mất 2.12 s/ảnh (r04), hoàn toàn đáp ứng được số lượng ảnh của bài. Khi box chuẩn thì mask khá sát, ví dụ img_004 mask người 0.93 và mask chiếc tuk-tuk xanh lá không bị lem ra nền hay sang xe bên cạnh. Điểm yếu là mask phụ thuộc chặt vào box: ở img_013 box car bị cắt ở x=325 nên mask cũng mất luôn phần đuôi tuk-tuk.
- Lỗi nào **không sửa được bằng đổi ngưỡng** (ví dụ tuk-tuk bị gọi `car`)? Sửa thế nào? Đó là lỗi nhận diện sai tên lớp xe ba bánh. YOLO26 huấn luyện trên bộ COCO vốn không có nhãn xe ba bánh, nên model tự gán vào lớp gần nhất: tuk-tuk ở img_004, img_013 thành car; xích lô ở img_032 thành bicycle 0.69. Chiếc xe xanh ở mép trái img_011 bị gán là truck 0.77 ở cả ngưỡng 0.25 lẫn 0.5; xe này bị cắt mất một nửa nên khó phân biệt giữa xe ba bánh và songthaew, đồng thời img_011 thuộc tập pilot nên không có nhãn gold để đối chiếu. Vì điểm tin cậy của các box này khá cao, việc chỉnh ngưỡng chỉ làm biến mất box chứ không sửa được tên lớp. Cách giải quyết là dùng YOLOE với prompt `three_wheeler` cho riêng lớp này; khi hai box chồng lấn nhau thì ưu tiên lấy `three_wheeler` thay cho `car`/`truck` của YOLO26.
- Box **điểm cao mà sai** hoặc **điểm thấp mà đúng** (ảnh nào):
  - Điểm cao mà sai: `bicycle` 0.69 ở img_032 thực chất là xích lô. Ngoài ra YOLOE gán `three_wheeler` 0.37 cho chiếc bán tải trắng ở img_002, trong khi đó là `truck`.
  - Điểm thấp mà đúng: chiếc xích lô bên phải ở img_032 chỉ đạt `three_wheeler` 0.27 nhưng hoàn toàn chính xác. Ở img_011, nhóm người đứng xa cuối phố chỉ có điểm `person` từ 0.28 đến 0.39 nhưng đều là người thật; nếu nâng ngưỡng lên 0.5 sẽ bị sót hết.
  - Vì vậy, điểm tin cậy không đồng nghĩa với độ chính xác của nhãn, việc chỉnh ngưỡng không thể thay thế khâu kiểm tra trực tiếp trên ảnh.
- Giới hạn: Ngưỡng được tinh chỉnh và đánh giá trên cùng một tập 4 ảnh gold với số lượng mẫu ít (`three_wheeler` 7, `bicycle` 5, `bus` 1, `truck` 0). Do đó các chỉ số P/R hiện tại đang mang tính lạc quan, mức recall 7/7 của `three_wheeler` khó giữ nguyên khi áp dụng ra toàn bộ dữ liệu.

## 2. Phân tích lỗi pre-label (từ `qc/review_stats.md`)

Số liệu dưới đây thống kê trên bản nhãn cuối (sau QC vòng 2). Script đánh giá đã loại trừ 22 đối tượng nằm quanh vùng bẫy vì đó là lỗi chủ động cài vào, không phải lỗi sinh ra từ quá trình chạy model.

| Trạng thái | Số lượng | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted | 570 | 570/690 = 82.6% | 690 box pre-label |
| edited | 25 | 25/690 = 3.6% | 690 box pre-label |
| deleted | 92 | 92/690 = 13.3% | 690 box pre-label |
| added | 52 | 52/648 = 8.0% | 648 đối tượng sau duyệt |

Trong nhóm `deleted` và `added` có 12 trường hợp thực chất là đổi nhãn lớp (tọa độ box đúng nhưng sai tên lớp). Trừ đi phần này thì số đối tượng bị sót hoàn toàn rơi vào khoảng 52 − 12 = 40.

- Lớp nào pre-label tệ nhất? Vì sao (dẫn ảnh)?
  - Tệ nhất là `three_wheeler`: 21/41 box sinh ra bị mình xoá (51%), chỉ giữ nguyên 15/41. YOLOE thường xuyên nhận nhầm các vật thể có hình khối hộp thành xe ba bánh, lỗi này lặp lại nhiều lần: xe tải thùng kín ở img_003 (YOLO26 bắt đúng là `truck` 0.87 nhưng luật gộp lại ưu tiên `three_wheeler` 0.61), quang gánh ở img_014 (0.63), xe minibus ở img_018 (0.53, lỗi này đến vòng QC 2 mình mới phát hiện ra).
  - `truck` cũng có tỉ lệ sai 100%: toàn bộ 6/6 box đều bị xoá. Đa số là xe ba bánh hoặc xe van bị YOLO26 gán nhầm thành `truck`, chẳng hạn chiếc xe van chở khách ở img_013 được mình sửa lại thành `car`.
  - `person` và `motorbike` chiếm số lượng áp đảo nhưng chất lượng ổn định: giữ nguyên 340/378 và 159/198. Lỗi chủ yếu là sinh box thừa ở các vật thể quá nhỏ, ở xa, hoặc người và xe bị nhoè trong bối cảnh ban đêm ở img_022.
- So với gold: precision đạt 74.7% (65/87 box pre-label, KTC 95%: 65–83%), recall đạt 75.6% (65/86, KTC 95%: 66–83%). Mẫu số là 86 đối tượng gold trên 4 ảnh. Các con số này cũng mang tính lạc quan vì ngưỡng tự tin được chọn ngay trên chính 4 ảnh này.

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Kết quả bắt bẫy:
  - Ở lần duyệt đầu tiên, mình chỉ bắt được 5/12 = 41.7% (KTC 95%: 19–68%), thấp hơn mức chuẩn 80% (file `qc/review_stats_lan1.md`). Bị lọt 7 bẫy: lọt toàn bộ 3 bẫy `loose` (img_018, img_021, img_025), 2/3 bẫy `missing` (img_003, img_014), 1/3 bẫy `extra` (img_026), và 1/3 bẫy `wrong_class` (img_031).
  - Sau khi duyệt lại, tỉ lệ tăng lên 10/12 = 83.3% (KTC 95%: 55–95%). Tuy nhiên con số này không nên xem là thước đo độc lập: sau khi nhận kết quả 5/12, mình đã mở `traps_key.json` để kiểm tra vị trí bẫy và tìm hiểu nguyên nhân lọt 2 bẫy `missing`, nên ở lần duyệt lại mình đã biết trước ảnh và đối tượng nào có cài bẫy.
  - 2 bẫy vẫn bị tính là lọt (img_003 và img_014, đều thuộc loại `missing`) xuất phát từ việc đáp án gốc của bẫy bị sai. Script `inject_traps.py` tự động xoá một box pre-label có điểm ≥ 0.5 và kỳ vọng người duyệt sẽ vẽ lại đúng lớp đó. Tuy nhiên, box bị xoá lại chính là box `three_wheeler` nhận diện sai của YOLOE: ở img_003 vật thể đó là xe tải, còn ở img_014 là quang gánh. Việc mình gán `truck` ở img_003 và bỏ trống không gán ở img_014 là hoàn toàn hợp lý. Nếu hiệu chỉnh lại đáp án chuẩn thì tỉ lệ bắt bẫy đạt 12/12 (chi tiết tại CAL-04, CAL-05).
  - Nguyên nhân lần đầu lọt nhiều bẫy là do mình chỉ tập trung kiểm tra tên lớp và tìm vật thể bị sót, mà không phóng to để soi độ ôm sát của khung bao. Các box `loose` chỉ lệch khoảng 7 px nên mình đã bỏ qua cả 3. Ngoài ra, với những đối tượng chưa có khung bao sẵn, mình cũng chưa rà soát kỹ toàn bộ khung hình trước khi bật lớp nhãn pre-label.
- Hệ quả của việc giữ nguyên (accepted) cao nhưng bắt bẫy thấp: Ở lần đầu, mình giữ nguyên tới 621/690 = 90.0% box có sẵn nhưng chỉ phát hiện được 5/12 bẫy. Điều này phản ánh tâm lý chủ quan, tin tưởng vào nhãn có sẵn nên những box trông tương đối ổn đều được cho qua. Các sai lệch nhỏ vài pixel hoặc các vật thể bị sót không thể hiện qua điểm tin cậy (conf vẫn cao), bắt buộc người duyệt phải chủ động phóng to và quét kỹ từng vùng ảnh.

## 4. Đánh giá hiệu quả của pre-label

- Thời gian thực hiện (từ `time_log.csv`): Gán nhãn thủ công hoàn toàn mất 29 phút cho 4 ảnh, trung bình khoảng 7.3 phút/ảnh. Duyệt trên nền pre-label mất 65 phút cho 24 ảnh, trung bình khoảng 2.7 phút/ảnh, tức nhanh hơn gần 3 lần. Con số này có độ chênh lệch nhất định do mình làm tập gold trước nên đã quen thuộc với bối cảnh và quy chuẩn (guideline), đồng thời mật độ đối tượng trong tập batch cũng phân bố không đồng đều. Ngoài ra, thời gian duyệt lại sau khi kiểm tra bẫy và thời gian chỉnh sửa hậu QC chưa được cộng vào mốc 65 phút, nên thời gian thực tế sẽ cao hơn.
- Kết luận: Rất hiệu quả đối với các lớp `person`, `motorbike`, `car`, `bicycle` (tỉ lệ giữ nguyên đạt 340/378, 159/198, 50/54; recall trên gold đạt 75.6%). Riêng với `three_wheeler` và `truck`, pre-label chỉ nên đóng vai trò tham khảo vì có tới 51% và 100% số box của hai lớp này bị xoá bỏ, đòi hỏi phải kiểm tra thủ công từng đối tượng.
- **Ba quy tắc giảm thiên kiến tự động hóa**:
  1. Quan sát tổng thể ảnh trước khi bật lớp nhãn pre-label: Đếm nhẩm số lượng người, xe máy, xe ba bánh rõ nét rồi đối chiếu với tổng số box hiện có; nếu thấy chênh lệch lớn thì chủ động tìm các vùng bị sót. Đo lường hiệu quả qua số bẫy `missing` bắt được.
  2. Phóng to các box ở vùng tối, đối tượng kích thước nhỏ hoặc biên mờ và kéo lại cạnh cho ôm sát vật thể (sai số không quá 2 px). Đo lường qua bẫy `loose`: mục tiêu nâng từ 0/3 lên 3/3.
  3. Lọc riêng toàn bộ các box `three_wheeler` và `truck` trong CVAT để duyệt độc lập từng cái, tuyệt đối không dựa vào điểm `conf` (vì box 0.63 vẫn sai ở img_014 trong khi box 0.27 lại đúng ở img_032). Đo lường qua số lượng box thuộc hai lớp này bị xoá hoặc đổi nhãn sau khi duyệt.

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| img_032 (xích lô) | SAM 2.1 base (`sam2.1_b.pt`) | 3 (2 dương, 1 âm) | Loại bỏ hoàn toàn người đàn ông đứng sát xe nhờ click âm; xe không chở hàng | Gần như hoàn hảo, lấy trọn vẹn mui, nệm, khung và cả nan hoa bánh xe, không lem sang chiếc xích lô kế bên. Khuyết phần yên và bánh sau vì bị mui che khuất | `xichlo_img032.jpg` |
| img_013 (tuk-tuk) | SAM 2.1 base | 3 (2 dương, 1 âm) | Phần lớn tài xế được tách ra nhưng vẫn sót vài đốm nhỏ; bị lem sang người ngồi phía sau nhìn xuyên qua khung xe | Lấy đầy đủ phần đuôi xe (khắc phục được lỗi mất đuôi ở lần chạy r04 dùng box YOLO26); tuy nhiên mask bị đặc cả phần khoảng trống bên trong cabin | `tuktuk_img013.jpg` |
| img_013 (tuk-tuk) | SAM 2.1 base | 5 (2 dương, 3 âm) | Loại bỏ sạch bóng người | Kết quả tệ hơn: mask chỉ còn lại phần mui và viền khung, diện tích giảm từ 45% (ở lần 3 click) xuống còn 12% ảnh và bị đứt thành 2 polygon. Nguyên nhân do đặt click âm lên người ngồi lọt trong khung xe khiến SAM loại bỏ luôn phần thân xe | `tuktuk_img013_5click.jpg` |
| img_003 (xe ba gác) | SAM 2.1 base | 4 (3 dương, 1 âm) | Bao gồm cả hàng hoá (dưa, rổ, túi) và em bé ngồi trên thùng xe dù đã đặt click âm lên bé; người lớn đeo khẩu trang ngồi phía sau trên xe thì được tách ra | Bị tách thành 2 polygon, phần mui bị đứt rời khỏi thùng xe tại vị trí thanh chống mảnh | `bagac_img003.jpg` |

- Đối chiếu guideline mục 4.2: Quy tắc 4.2.4 (hai đối tượng tiếp xúc nhau thì tách thành từng polygon riêng biệt, không chồng lấn) là căn cứ để mình loại bỏ phần người ra khỏi mask phương tiện. Quy tắc 4.2.2 về xử lý lỗ thủng áp dụng khớp với các khoảng trống xuyên qua khung xe xích lô và tuk-tuk. Điểm khuyết của bản v1.0 là chưa quy định rõ hàng hoá và người ngồi trên xe có gộp chung vào mask phương tiện hay không, nên mình đã bổ sung vào bản v1.1: hàng hoá nằm gọn trong thùng được tính vào mask xe (CAL-01), còn người ngồi trên xe thì không tính và phải dùng công cụ tẩy thủ công (CAL-02). Kinh nghiệm rút ra là SAM tách tốt người đứng tách biệt bên ngoài nhưng dễ nhầm lẫn với người ngồi lọt bên trong khung xe, đồng thời việc tăng thêm click âm không phải lúc nào cũng cải thiện chất lượng mask.
- *(Tuỳ chọn)* Interactor SAM trong CVAT: Mình không sử dụng trực tiếp trên CVAT mà chạy độc lập bằng script `sam2_points.py` ở local.

## 6. Calibration (Part 6)

- Tổng số quyết định: 7 dòng trong `calibration_decisions.csv`, phân loại thành 3 `cau_hoi_mo`, 3 `loi_may` và 1 `can_review` (CAL-07, trường hợp người bị che khuất trong ảnh đêm img_022). Hai quyết định mang tính bước ngoặt nhất:
  - CAL-05 (img_003): Quy tắc gộp "ưu tiên `three_wheeler` đè lên `car`/`truck` khi IoU ≥ 0.6" đã vô tình thay thế box `truck` 0.87 chính xác bằng box `three_wheeler` 0.61 sai lệch. Mình đề xuất điều chỉnh luật gộp: chỉ cho phép `three_wheeler` ghi đè lên `car`, hoặc chỉ ghi đè lên `truck` khi điểm tin cậy của YOLOE cao hơn YOLO26.
  - CAL-03 (img_014, img_032): Không thiết lập ngưỡng `conf` để tự động chấp nhận nhãn (auto-accept), vì thực tế tồn tại box điểm 0.63 vẫn sai trong khi box 0.27 lại đúng.
- Các quy tắc mới cập nhật vào `guideline_v1.1_calibrated.md`: Chốt câu hỏi mở số 3 (quy định mức hàng nhô 20% áp dụng đồng nhất cho cả segmentation mask), số 4 (người ngồi trong hoặc trên xe không thuộc mask phương tiện), số 7 (không áp dụng ngưỡng auto-accept); bổ sung quang gánh, sạp hàng vào danh mục loại trừ của `three_wheeler`; quy định rõ khi nhãn gợi ý `three_wheeler` chồng lấn với `truck` thì bắt buộc phải kiểm tra số lượng bánh xe và kết cấu cabin; khẳng định tuk-tuk và xích lô luôn thuộc nhóm `three_wheeler`; với người bị che khuất trong điều kiện thiếu sáng thì chỉ gán nhãn khi nhận diện rõ phần đầu và thân trên. Các câu hỏi mở 1, 2, 5, 6 tạm thời chưa chốt do chưa gặp ca dữ liệu thực tế đủ rõ ràng.
- Phân tích nguyên nhân gốc: Phần lớn sai sót xuất phát từ hạn chế của bộ pre-label (YOLO26 thiếu lớp xe ba bánh, YOLOE dễ nhầm lẫn các vật thể dạng hộp, và luật gộp đang ưu tiên tuyệt đối cho YOLOE), một phần nhỏ do guideline v1.0 chưa bao quát hết tình huống hàng hoá và người ngồi trên xe khi tạo mask. Riêng việc để lọt các bẫy `loose` hoàn toàn là lỗi thao tác cá nhân do chưa tuân thủ chặt chẽ các bước trong checklist kiểm duyệt.

## 7. QC và quyết định (Part 6)

- Vòng 1 (`qc/vong1_*`): Lấy mẫu 7 ảnh (2 ca khó và 5 ảnh ngẫu nhiên, `seed 42`) với tổng cộng 174 đối tượng. Một hạn chế khách quan là người thực hiện QC cũng chính là người trực tiếp duyệt nhãn.
  - Lỗi theo đối tượng 4/174 = 2.3%, tính theo ảnh là 3/7 = 42.9% (vượt ngưỡng Release quy định ≤ 2%), cận trên KTC 95% lên tới 75%.
  - Chi tiết 4 lỗi: 2 box thừa (img_022 #311 là box `person` quá khổ trùm cả rổ hàng trong ảnh đêm; img_035 #660 là box thứ hai trên cùng một chiếc xích lô), 1 lỗi sai lớp (img_013 #151 xe van chở khách bị gán nhầm thành `truck`), 1 lỗi sai kích thước (img_035 #650 box xích lô bị cắt cụt phần mui). Toàn bộ các lỗi này đã được mình khắc phục trên CVAT và đóng gói lại bản release.
- Vòng 2 (`qc/qc_*`): Lấy mẫu 6 ảnh (1 ca khó và 5 ảnh ngẫu nhiên), đổi sang `seed 2026` để mở rộng độ phủ, thu được 175 đối tượng.
  - Lỗi theo đối tượng giảm còn 1/175 = 0.6%, tính theo ảnh là 1/6 = 16.7%, cận trên KTC 95% ở mức 56%.
  - Chi tiết lỗi: 1 lỗi sai lớp (img_018 #229 xe minibus bị gán nhầm thành `three_wheeler`). Mình đã đổi thành `bus` ngay sau khi QC và xuất lại bản release.
- Hạn chế của kích thước mẫu nhỏ: Với quy mô mẫu chỉ 6 đến 7 ảnh, ngay cả khi không phát hiện lỗi nào thì cận trên KTC 95% vẫn neo ở mức 40–50%, do đó chưa đủ sức mạnh thống kê để chứng minh tỉ lệ lỗi dưới ngưỡng 2%. Kết quả vòng 2 còn mang tính lạc quan do có tới 4/6 ảnh bị trùng lặp với tập mẫu vòng 1 (vốn đã được sửa sạch lỗi từ trước, do toàn bộ batch chỉ có 24 ảnh). Trong số 2 ảnh mới hoàn toàn là img_018 và img_027 thì đã có 1 ảnh dính lỗi.
- Phạm vi kiểm soát của script tự động: Script chỉ thống kê dựa trên các phiếu ghi nhận thủ công và đối chiếu tọa độ box với tập bẫy/gold thông qua chỉ số IoU. Công cụ này không thể tự đánh giá độ ôm sát của khung bao nếu thiếu ground truth, không phát hiện được các đối tượng bị sót đồng thời ở cả bước pre-label lẫn bước duyệt tay, và sẽ chấm sai nếu dữ liệu bẫy gốc lấy từ nhãn sai của pre-label như trường hợp img_003 và img_014. Việc người duyệt có tuân thủ đúng quy trình checklist hay không cũng nằm ngoài khả năng kiểm tra của script.

**Quyết định: Rework**, ngày 2026-11-17. Lý do (xét theo thang Hold → Rework → Release): Không chọn Hold vì các tình huống gặp trong bài đã được chốt ở guideline v1.1 (câu mở 1, 2, 5, 6 chưa gặp nên chưa chốt) và nguồn dữ liệu gốc không gặp lỗi hỏng hóc. Chọn Rework vì cả hai vòng QC đều ghi nhận tỉ lệ ảnh lỗi cao hơn đáng kể so với ngưỡng 2% (3/7 và 1/6), đồng thời lỗi nhận diện sai ở nhóm `three_wheeler` vẫn lặp lại có hệ thống (img_003, img_014, img_018). Chưa thể Release được vì kích thước mẫu hiện tại quá nhỏ để đảm bảo độ tin cậy thống kê cho ngưỡng ≤ 2%.
Kế hoạch hành động tiếp theo cho giai đoạn Rework:
1. Rà soát lại toàn bộ các box thuộc nhóm `three_wheeler`, `truck`, và `bus` trên cả 24 ảnh theo đúng quy tắc số 3 tại mục 4.
2. Phóng to kiểm tra lại độ ôm sát của toàn bộ khung bao tại các khung hình có mật độ đông đúc và ảnh chụp ban đêm.
3. Cập nhật lại logic gộp nhãn theo đề xuất CAL-05 để áp dụng cho các lô dữ liệu tiếp theo.
4. Tiến hành QC vòng 3 với kích thước mẫu lớn hơn (ví dụ `--n 10` kèm seed mới) và ghi nhận lại toàn bộ chỉ số.