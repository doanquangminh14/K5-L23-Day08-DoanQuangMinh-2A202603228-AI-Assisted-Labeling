# Label contract — Phố châu Á — BỔ SUNG v1.1 — 2A202603228 - Đoàn Quang Minh

> **Chỉ viết phần thay đổi so với v1.0** (không copy cả file). Chốt các câu hỏi mở ở mục 6 của v1.0 (hoặc nói rõ câu nào chưa chốt, vì sao)
> và thêm quy tắc mới rút ra từ lỗi của máy / lỗi của bạn. Mỗi ca khó **có ảnh ví dụ** (`img_0xx`) và khớp một dòng trong `qc/calibration_decisions.csv`.

## 1. Câu hỏi mở đã chốt (từ mục 6 của v1.0)

| # | Câu hỏi | Quyết định | Ảnh ví dụ | `case_id` |
|---|---|---|---|---|
| 3 | Hàng nhô ra / hàng chở trên xe: box và mask xe có gồm hàng không? | Hàng **nằm trong thùng** hoặc nhô ra **≤ ~20%** chiều dài xe → thuộc xe, **cả box lẫn mask**. Nhô ra > 20% → box và mask chỉ lấy thân xe. (v1.0 mới nói box; v1.1 áp cùng ngưỡng cho mask.) | img_003 (ba gác chở dưa, rổ, túi trong thùng) | CAL-01 |
| 4 | Người ngồi trong / trên xe ba bánh | **Box:** giữ ngưỡng v1.0, gán `person` nếu nhìn thấy ≥ 20% người. **Mask:** người ngồi trong/trên xe **không thuộc** mask xe, kể cả khi nằm lọt trong khung xe; vùng người che xe thuộc người. SAM không tự tách được → phải sửa mask bằng eraser. | img_003 (em bé ngồi trên ba gác), img_013 (tài xế tuk-tuk) | CAL-02 |
| 7 | Mức `conf` tối thiểu để nhận box máy không cần sửa | **Không đặt.** `conf` chỉ dùng xếp thứ tự duyệt; mọi box phải được nhìn. Lý do: box sai có `conf` cao, box đúng có `conf` thấp. | img_014 (quang gánh, three_wheeler 0.63 **sai**), img_032 (xích lô, three_wheeler 0.27 **đúng**) | CAL-03 |

**Câu chưa chốt** (1, 2, 5, 6): khi duyệt 4 ảnh gold + 24 ảnh batch mình chưa gặp ca đủ rõ để chốt, và chưa làm mask ô tô có kính (Part 5 chỉ làm mask xe ba bánh). Giữ nguyên v1.0, gặp ca thực tế thì tick `can_review` và chốt ở v1.2.

## 2. Ca khó mới (từ lỗi của máy, bẫy lọt, box `can_review`)

| Ảnh | Tình huống | Máy làm gì (model nào) | Quyết định của bạn | Quy tắc mới (vào mục nào của v1.0) |
|---|---|---|---|---|
| img_014 | Người đi bộ gánh quang gánh hàng (2 thúng treo đòn gánh) | YOLOE gán `three_wheeler` **0.63**; YOLO26 không có box. Bẫy *missing* lấy chính box sai này làm đáp án | **Không gán** (không phải xe); giữ nhãn dù `review_stats` tính là bẫy lọt | **Mục 2**, lớp `three_wheeler`, cột *Không gồm*: thêm quang gánh, sạp hàng, xe đẩy tay không phải xe ba bánh (CAL-04) |
| img_003 | Xe tải thùng kín nhỏ ở cuối đường | YOLO26: `truck` **0.87** (đúng); YOLOE: `three_wheeler` 0.61 (sai); luật gộp Part 4 (three_wheeler thắng car/truck khi IoU ≥ 0.6) giữ box **sai**. Bẫy *missing* lấy box sai này làm đáp án | Gán `truck` (có cabin, ≥ 4 bánh); giữ nhãn dù `review_stats` tính là bẫy lọt | **Mục 4.3**: máy gợi ý `three_wheeler` chồng lên `truck` → đếm bánh / nhìn cabin, không tin nhãn gộp. Đề xuất sửa luật gộp: chỉ thắng `car`, hoặc thắng `truck` khi điểm YOLOE > YOLO26 (CAL-05) |
| img_004, img_013, img_032 | Tuk-tuk, xích lô | YOLO26 (COCO không có lớp xe ba bánh) gọi tuk-tuk là `car`, xích lô là `bicycle` 0.69; đổi ngưỡng 0.25 → 0.5 không đổi được tên lớp | Luôn `three_wheeler` | **Mục 2**: xe ba bánh có mui / có ghế khách vẫn là `three_wheeler`; `car` / `bicycle` trên xe ba bánh là lỗi máy, phải đổi lớp (CAL-06) |
| img_022 | Ảnh đêm: người tối màu đứng sau người đi xe máy, bị che phần lớn (box #304, 22×42 px); tôi tick `can_review` vì khó ước lượng "còn thấy ≥ 20%" | YOLO26 gán `person` 0.29 | Giữ `person`: nhận ra được đầu và vai, cao > 20 px | **Mục 4.1.2**: ở ảnh tối / mờ, dùng tiêu chí "nhận ra được đầu + thân trên là người" thay cho ước lượng 20% diện tích; chỉ thấy mảng tối không rõ hình người thì không gán (CAL-07) |

## 3. Lịch sử thay đổi

| Phiên bản | Thay đổi | Lý do (ca nào, `case_id`) |
|---|---|---|
| v1.0 | Bản giảng viên | — |
| v1.1 | Chốt câu mở 3 (ngưỡng hàng 20% áp cho cả mask), 4 (người trong/trên xe không thuộc mask xe), 7 (không đặt conf tự chấp nhận). Thêm vào *Không gồm* của `three_wheeler`: quang gánh, sạp hàng. Thêm quy tắc kiểm `truck` khi máy gợi ý `three_wheeler`. Nhấn mạnh tuk-tuk / xích lô luôn là `three_wheeler`. Thêm tiêu chí gán người bị che trong ảnh tối (mục 4.1.2). | CAL-01, CAL-02, CAL-03 (câu mở); CAL-04, CAL-05, CAL-06 (lỗi máy); CAL-07 (`can_review`) |
