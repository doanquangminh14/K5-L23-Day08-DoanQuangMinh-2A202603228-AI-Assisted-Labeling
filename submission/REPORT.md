# Báo cáo Day 08 — 2A202603228 - Đoàn Quang Minh

> Một file duy nhất cho cả bài (đã gộp recipe, QC và ghi chú cá nhân). Mọi con số kèm **mẫu số** và nguồn (file nào). Viết bằng lời của bạn.

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| person / bicycle / motorbike / car / bus / truck | YOLO26m | person 0.15, motorbike 0.15; car, bicycle, bus, truck để mặc định 0.3 | Với 6 lớp COCO thì YOLO26 làm ổn (r01, r02). Mình thử lên 0.5 thì mất người và xe máy ở xa ở img_007, img_011 nhiều hơn số box rác bỏ được, nên dùng ngưỡng thấp. Person và motorbike mình hạ xuống 0.15 vì lúc duyệt, xoá box thừa chỉ một click, còn vẽ thêm box bị sót thì lâu hơn nhiều. Car để 0.3 vì precision của car gần như đứng yên ở mọi ngưỡng (0.47 đến 0.58), lỗi chính của car là gọi tuk-tuk thành car, đổi ngưỡng không giải quyết được. Bicycle chỉ có 5 vật, bus 1, truck 0 trong gold nên không đủ để chọn ngưỡng riêng |
| three_wheeler | YOLOE-26s-seg, `--classes three_wheeler` (r03, r06, khớp `work/runs.csv`) | 0.15 | YOLO26 không có lớp xe ba bánh nên tuk-tuk, xích lô bị gọi thành truck, car, motorbike, bicycle hoặc bị bỏ qua (r01, r04). YOLOE thì bắt được tuk-tuk ở img_002 (0.93) và xích lô ở img_032 (0.92, 0.48, 0.27) (r03). Ở 0.15, trên gold được recall 7/7, precision 7/9 = 78% |

Hạ ngưỡng từ 0.3 xuống 0.15 thì được và mất như sau (lấy từ `qc/threshold_sweep.csv`, so với gold):

| Lớp (số vật gold) | Ở 0.30 | Ở 0.15 | Chênh lệch |
|---|---|---|---|
| person (42) | 26 đúng, 8 thừa | 32 đúng, 12 thừa | thêm 6 vật thật, thêm 4 box thừa |
| motorbike (23) | 10 đúng, 1 thừa | 14 đúng, 4 thừa | thêm 4 vật thật, thêm 3 box thừa |

Mình dừng ở 0.15 vì nếu hạ person tiếp xuống 0.10 thì chỉ được thêm 1 người (33 đúng) mà box thừa tăng lên 17.

- Lệnh `merge_prelabels.py` đã dùng (dán nguyên):
  ```bash
  python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --thr "person=0.15,motorbike=0.15,three_wheeler=0.15" --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json
  ```
- **Phiên bản SAM bạn chọn** (SAM 1 / 2.1 / 3) và lý do: mình dùng SAM 2.1 base (`sam2.1_b.pt`). Lý do chính là tải về chạy được luôn, SAM 3 thì phải xin quyền. Trên RTX 3050 chạy 2.12 s/ảnh (r04), với số ảnh của bài là đủ. Khi box đúng thì mask khá sát, ví dụ img_004 mask người 0.93 và mask chiếc tuk-tuk xanh lá không bị lem ra nền hay sang xe bên cạnh. Điểm yếu là mask đi theo box: ở img_013 box car bị cắt ở x=325 nên mask cũng mất luôn phần đuôi tuk-tuk.
- Lỗi nào **không sửa được bằng đổi ngưỡng** (ví dụ tuk-tuk bị gọi `car`)? Sửa thế nào? Là lỗi gọi sai tên xe ba bánh. YOLO26 train trên COCO, COCO không có xe ba bánh, nên nó gán vào lớp nào gần nhất: tuk-tuk ở img_004, img_013 thành car, xích lô ở img_032 thành bicycle 0.69. Xe xanh ở mép trái img_011 thì là truck 0.77 ở cả 0.25 lẫn 0.5; xe này bị cắt mất một nửa, mình không chắc là xe ba bánh hay songthaew, và img_011 thuộc pilot nên không có gold để so. Vì điểm của các box này không thấp, đổi ngưỡng chỉ làm box biến mất chứ tên lớp vẫn sai. Cách mình sửa là dùng YOLOE với prompt three_wheeler cho lớp này, khi hai box chồng nhau thì lấy three_wheeler thay cho car/truck của YOLO26.
- Box **điểm cao mà sai** hoặc **điểm thấp mà đúng** (ảnh nào):
  - Điểm cao mà sai: bicycle 0.69 ở img_032 thật ra là xích lô. Ngoài ra YOLOE cho three_wheeler 0.37 vào chiếc bán tải trắng ở img_002, đó là truck.
  - Điểm thấp mà đúng: xích lô bên phải ở img_032 chỉ được three_wheeler 0.27 nhưng đúng là xích lô. Ở img_011 mấy người đứng xa cuối phố chỉ có person 0.28 đến 0.39 nhưng đều là người thật, lên 0.5 là mất.
  - Nên mình không coi điểm tin cậy là nhãn đúng, ngưỡng không thay được việc nhìn ảnh.
- Giới hạn: ngưỡng được chọn và chấm trên cùng 4 ảnh gold, số vật lại ít (three_wheeler 7, bicycle 5, bus 1, truck 0). Vì vậy P/R ở trên đang lạc quan, recall 7/7 của three_wheeler chắc chắn khó giữ được khi ra cả bộ dữ liệu.

## 2. AI error profile (từ `qc/review_stats.md`)

Số dưới đây là của bản nhãn cuối (sau QC vòng 2). Script đã loại 22 đối tượng nằm quanh bẫy vì đó là lỗi cài vào, không phải lỗi của máy.

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted | 570 | 570/690 = 82.6% | 690 box máy |
| edited | 25 | 25/690 = 3.6% | 690 box máy |
| deleted | 92 | 92/690 = 13.3% | 690 box máy |
| added | 52 | 52/648 = 8.0% | 648 đối tượng sau duyệt |

Trong số deleted và added có 12 trường hợp là đổi lớp (box đúng chỗ nhưng máy gán sai lớp). Trừ ra thì số vật máy thật sự bỏ sót khoảng 52 − 12 = 40.

- Lớp nào máy tệ nhất? Vì sao (dẫn ảnh)?
  - Tệ nhất là three_wheeler: 21/41 box máy bị mình xoá (51%), chỉ 15/41 giữ nguyên. YOLOE hay gọi những thứ có dạng hộp là xe ba bánh, và lỗi này lặp lại nhiều lần: xe tải thùng kín ở img_003 (YOLO26 nói đúng là truck 0.87 nhưng luật gộp lại để three_wheeler 0.61 thắng), quang gánh ở img_014 (0.63), minibus ở img_018 (0.53, mình chỉ phát hiện ra lúc QC vòng 2).
  - Truck cũng tệ: cả 6/6 box máy đều bị xoá. Phần lớn là xe ba bánh hoặc xe van bị YOLO26 gọi là truck, ví dụ xe van chở khách ở img_013 mình sửa lại thành car.
  - Person và motorbike nhiều box nhất nhưng ổn: giữ nguyên 340/378 và 159/198. Lỗi chủ yếu là box thừa ở vật nhỏ, ở xa, và người, xe bị nhoè trong ảnh đêm img_022.
- So với gold: precision 74.7% (65/87 box máy, KTC 95%: 65–83%), recall 75.6% (65/86, KTC 95%: 66–83%). Mẫu số là 86 đối tượng gold trên 4 ảnh. Số này cũng lạc quan vì ngưỡng chọn trên chính 4 ảnh này.

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy:
  - Lần duyệt đầu mình chỉ bắt được 5/12 = 41.7% (KTC 95%: 19–68%), dưới mức 80% (file `qc/review_stats_lan1.md`). Lọt 7 bẫy: loose lọt cả 3 (img_018, img_021, img_025), missing lọt 2/3 (img_003, img_014), extra lọt 1/3 (img_026), wrong_class lọt 1/3 (img_031).
  - Sau khi duyệt lại thì lên 10/12 = 83.3% (KTC 95%: 55–95%). Con số này không nên coi là số đo độc lập: sau khi có kết quả 5/12, mình mới mở `traps_key.json` để xem bẫy nằm ở đâu và tìm hiểu 2 bẫy missing, nên lúc duyệt lại mình đã biết trước ảnh nào, vật nào có bẫy.
  - 2 bẫy vẫn bị tính là lọt (img_003 và img_014, đều là missing) thì mình cho là đáp án của bẫy sai. `inject_traps.py` xoá một box máy có điểm ≥ 0.5 rồi chờ người vẽ lại đúng lớp đó. Nhưng box nó xoá lại là box three_wheeler sai của YOLOE: ở img_003 đó là xe tải, ở img_014 là quang gánh. Mình gán truck ở img_003 và không gán gì ở img_014 là đúng. Nếu tính lại thì được 12/12 (xem CAL-04, CAL-05).
  - Lần đầu lọt nhiều vì mình chủ yếu kiểm xem box có đúng lớp không và có thiếu box nào không, nhưng không phóng to để xem box có ôm sát không. Box loose chỉ lệch khoảng 7 px nên mình bỏ qua cả 3. Với những vật máy không vẽ box, mình cũng không quét kỹ ảnh trước khi nhìn nhãn máy.
- Accepted cao mà bắt bẫy thấp nói gì? Lần đầu mình giữ nguyên 621/690 = 90.0% box máy mà chỉ bắt được 5/12 bẫy. Tức là mình đã tin máy quá, box nào nhìn gần đúng là cho qua. Box lệch vài pixel hay vật bị sót thì không thấy trên điểm số (conf vẫn cao), muốn bắt được phải chủ động phóng to và quét lại cả ảnh.

## 4. Máy có đáng dùng không?

- Thời gian (từ `time_log.csv`): gán tay 29 phút cho 4 ảnh, khoảng 7.3 phút/ảnh. Duyệt có pre-label 65 phút cho 24 ảnh, khoảng 2.7 phút/ảnh, tức nhanh hơn gần 3 lần. Có thiên lệch: mình gán gold trước nên lúc duyệt đã quen cảnh và quen guideline, ảnh batch cũng đông hơn hoặc thưa hơn tuỳ ảnh. Thời gian duyệt lại sau khi biết bẫy lọt và thời gian sửa lỗi QC mình không ghi vào 65 phút, nên thực tế duyệt tốn hơn con số này.
- Kết luận: đáng dùng cho person, motorbike, car, bicycle (giữ nguyên 340/378, 159/198, 50/54; recall trên gold 75.6%). Với three_wheeler và truck thì chỉ nên coi là gợi ý, vì 51% và 100% box máy của hai lớp này bị xoá, box nào cũng phải tự kiểm.
- **Ba quy tắc giảm thiên kiến của bạn**:
  1. Nhìn ảnh trước khi bật nhãn máy: đếm sơ số người, xe máy, xe ba bánh rõ ràng rồi so với số box máy, lệch nhiều thì đi tìm box bị sót. Kiểm bằng số bẫy missing bắt được.
  2. Phóng to những box ở vật tối, vật nhỏ, cạnh mờ và kéo cạnh cho sát (lệch không quá khoảng 2 px). Kiểm bằng bẫy loose: lần đầu 0/3, phải được 3/3.
  3. Lọc riêng các box three_wheeler và truck trong CVAT để kiểm từng cái, không dựa vào conf, vì box 0.63 vẫn sai (img_014) còn box 0.27 vẫn đúng (img_032). Kiểm bằng số box hai lớp này bị xoá hoặc đổi lớp sau khi duyệt.

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| img_032 (xích lô) | SAM 2.1 base (`sam2.1_b.pt`) | 3 (2 dương, 1 âm) | Không lấy người đàn ông đứng sát xe, click âm bỏ được hết; xe không chở hàng | Gần như không lỗi, lấy đủ mui, nệm, khung và cả nan hoa, không lem sang xích lô bên cạnh. Thiếu phần yên và bánh sau vì bị mui che | `xichlo_img032.jpg` |
| img_013 (tuk-tuk) | SAM 2.1 base | 3 (2 dương, 1 âm) | Phần lớn tài xế bị loại nhưng vẫn còn lốm đốm; bị lem sang người ngồi phía sau, nhìn thấy qua khung xe | Lấy đủ cả đuôi xe (lần r04 dùng box YOLO26 thì mất phần này); mask lấy luôn khoảng trống trong cabin | `tuktuk_img013.jpg` |
| img_013 (tuk-tuk) | SAM 2.1 base | 5 (2 dương, 3 âm) | Không còn người | Ra tệ hơn: mask chỉ còn mui và viền, 12% ảnh so với 45% ở lần 3 click, bị tách 2 polygon. Mình nghĩ do đặt click âm lên người ngồi lọt trong khung xe nên SAM bỏ luôn thân xe | `tuktuk_img013_5click.jpg` |
| img_003 (xe ba gác) | SAM 2.1 base | 4 (3 dương, 1 âm) | Lấy cả hàng (dưa, rổ, túi) và cả em bé ngồi trên xe, dù mình đã click âm lên bé; người lớn đeo khẩu trang thì không bị lấy | Ra 2 polygon, phần mui bị tách khỏi thùng ở chỗ thanh chống nhỏ | `bagac_img003.jpg` |

- Đối chiếu guideline mục 4.2: quy tắc 4.2.4 (hai vật chạm nhau thì mỗi vật một polygon, không chồng) giúp mình quyết là phải bỏ người ra khỏi mask xe. Quy tắc 4.2.2 về lỗ thủng thì khớp với khoảng trống trong khung xích lô và tuk-tuk. Cái thiếu là v1.0 không nói hàng chở trên xe và người ngồi trên xe có tính vào mask hay không, nên mình chốt ở v1.1: hàng trong thùng tính vào xe (CAL-01), người không tính vào mask xe và phải tẩy tay bằng eraser (CAL-02). Rút ra: SAM tách được người đứng bên ngoài xe nhưng không tách được người ngồi trong hay trên xe, và thêm click âm không phải lúc nào cũng tốt hơn.
- *(Tuỳ chọn)* Interactor SAM trong CVAT: mình không thử, chỉ chạy `sam2_points.py` trên máy.

## 6. Calibration (Part 6)

- Số quyết định: 7 dòng trong `calibration_decisions.csv`, gồm 3 `cau_hoi_mo`, 3 `loi_may` và 1 `can_review` (CAL-07, người bị che trong ảnh đêm img_022). Hai quyết định mình thấy quan trọng nhất:
  - CAL-05 (img_003): luật gộp "three_wheeler thắng car/truck khi IoU ≥ 0.6" đã thay box truck 0.87 đúng bằng box three_wheeler 0.61 sai. Mình đề xuất chỉ cho three_wheeler thắng car, hoặc chỉ thắng truck khi điểm YOLOE cao hơn YOLO26.
  - CAL-03 (img_014, img_032): không đặt mức conf nào để tự động nhận box, vì có box 0.63 sai và box 0.27 đúng.
- Quy tắc mới đã vào `guideline_v1.1_calibrated.md`: chốt câu mở 3 (mức hàng nhô 20% áp dụng cho cả mask), câu 4 (người trong hay trên xe không thuộc mask xe), câu 7 (không đặt conf tự nhận); thêm quang gánh, sạp hàng vào phần "không gồm" của three_wheeler; khi máy gợi ý three_wheeler đè lên truck thì phải đếm bánh, nhìn cabin; tuk-tuk và xích lô luôn là three_wheeler; với người bị che trong ảnh tối thì nhận ra được đầu và thân trên mới gán. Câu mở 1, 2, 5, 6 mình chưa chốt vì chưa gặp trường hợp đủ rõ.
- Nguyên nhân gốc: phần lớn do máy (YOLO26 không có lớp xe ba bánh, YOLOE nhầm vật dạng hộp, luật gộp ưu tiên YOLOE), một phần do guideline chưa nói rõ chuyện hàng hoá và người trên xe khi làm mask. Còn chuyện bẫy loose bị lọt là lỗi của mình, do không làm đủ các bước checklist, không phải do guideline.

## 7. QC và quyết định (Part 6)

- Vòng 1 (`qc/vong1_*`): mẫu 7 ảnh (2 ca khó và 5 ngẫu nhiên, seed 42), 174 đối tượng. Người kiểm là mình, mà mình cũng là người duyệt, đây là một hạn chế.
  - Lỗi theo đối tượng 4/174 = 2.3%, theo ảnh 3/7 = 42.9% (ngưỡng Release là ≤ 2%), cận trên 95% là 75%.
  - Các lỗi: 2 box thừa (img_022 #311 là box person rất to bao cả rổ hàng trong ảnh đêm; img_035 #660 là box thứ hai trên cùng một chiếc xích lô), 1 sai lớp (img_013 #151 xe van chở khách bị gán truck), 1 sai vị trí kích thước (img_035 #650 box xích lô không lấy phần mui). Mình đã sửa hết trong CVAT và build lại release.
- Vòng 2 (`qc/qc_*`): mẫu 6 ảnh (1 ca khó và 5 ngẫu nhiên). Mình đổi seed sang 2026 để lấy mẫu khác, được 175 đối tượng.
  - Lỗi theo đối tượng 1/175 = 0.6%, theo ảnh 1/6 = 16.7%, cận trên 95% là 56%.
  - Lỗi: 1 sai lớp (img_018 #229 là minibus nhưng bị gán three_wheeler). Mình đã đổi thành bus sau khi QC và build lại release.
- Giới hạn mẫu nhỏ: với 6 hay 7 ảnh thì kể cả không có lỗi nào, cận trên vẫn ở mức 40–50%, nên không thể chứng minh được mức 2%. Vòng 2 còn lạc quan hơn nữa vì 4/6 ảnh trùng với vòng 1 (batch chỉ có 24 ảnh) và đã được sửa rồi. Chỉ có 2 ảnh mới là img_018 và img_027, và một trong hai có lỗi.
- Script kiểm được / không kiểm được: script chỉ đếm theo phiếu mình tự điền và so box với bẫy, với gold bằng IoU. Nó không biết box có ôm sát hay không nếu không có đáp án, không bắt được lỗi mà cả máy lẫn mình cùng bỏ sót, và chấm sai khi đáp án bẫy lấy từ box sai của máy như ở img_003, img_014. Mình có làm đúng checklist hay không thì script cũng không kiểm được.

**Quyết định: Rework**, ngày 2026-11-17. Lý do (đi theo Hold → Rework → Release): không chọn Hold vì những ca mình gặp đã được chốt trong guideline v1.1 và dữ liệu không bị hỏng. Chọn Rework vì cả hai vòng đều có tỉ lệ ảnh lỗi cao hơn nhiều so với 2% (3/7 và 1/6), và máy lặp lại cùng một kiểu lỗi ở three_wheeler (img_003, img_014, img_018). Chưa Release được vì mẫu nhỏ thế này không chứng minh được ≤ 2%.
Nếu Rework / Hold, việc cần làm tiếp: (1) duyệt lại toàn bộ box three_wheeler, truck, bus trên 24 ảnh theo quy tắc 3 ở mục 4; (2) phóng to kiểm lại độ sát của box ở các ảnh đông và ảnh đêm; (3) sửa luật gộp như CAL-05 rồi chạy lại pre-label cho lô sau; (4) QC thêm vòng 3 với mẫu lớn hơn, ví dụ `--n 10` với seed mới, rồi ghi kết quả.

