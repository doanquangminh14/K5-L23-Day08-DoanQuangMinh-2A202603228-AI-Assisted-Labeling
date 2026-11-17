# Số đo review (det)

- Ảnh tính: 24 (batch,gold); box máy: 690; đối tượng sau duyệt: 670; ngưỡng accepted: IoU >= 0.9
- Đã **loại 22 đối tượng nằm quanh các bẫy** khỏi bảng này: lỗi nhân tạo không phản ánh chất lượng máy.

## 1. Bốn trạng thái (AI error profile)
| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted (giữ nguyên) | 621 | 621/690 = 90.0% | 690 box máy |
| edited (sửa) | 8 | 8/690 = 1.2% | 690 box máy |
| deleted (xoá) | 61 | 61/690 = 8.8% | 690 box máy |
| added (vẽ thêm) | 41 | 41/670 = 6.1% | 670 đối tượng sau duyệt |

Trong số deleted + added, **12 trường hợp là đổi lớp** (máy gán sai lớp, box đúng chỗ): đó không phải máy bỏ sót. Số vật máy **sót thật** ≈ added − đổi lớp = 29.

### Theo lớp
| Lớp | accepted | edited | deleted | added |
|---|---|---|---|---|
| person | 358 | 3 | 17 | 11 |
| bicycle | 5 | 0 | 2 | 5 |
| motorbike | 178 | 4 | 19 | 15 |
| car | 51 | 0 | 3 | 3 |
| bus | 3 | 0 | 0 | 2 |
| truck | 2 | 0 | 4 | 2 |
| three_wheeler | 24 | 1 | 16 | 3 |

## 2. So với gold (nhãn gán tay từ đầu)
Ảnh gold: 4; đối tượng gold: 86 (mẫu số của recall)
| | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| Máy (pre-label) | 65 | 22 | 21 | 74.7% (KTC 95%: 65–83%) | 75.6% (KTC 95%: 66–83%) |

*Lạc quan: ngưỡng được chọn trên chính các ảnh gold này (in-sample), và chỉ có 4 ảnh. Đọc khoảng tin cậy, đừng đọc con số đơn.*


(Ảnh gold không nằm trong nhãn sau duyệt: không tính residual / new error. Đó là chủ ý: duyệt lại ảnh bạn vừa tự gán tay sẽ đo trí nhớ, không đo thiên kiến.)

## 3. Ảnh bẫy
- Bắt được 5/12 = 41.7% (KTC 95%: 19–68%) (ngưỡng đạt: 80%). Chỉ 12 bẫy nên khoảng tin cậy rất rộng: đọc cùng danh sách bẫy lọt, đừng kết luận chỉ từ %.
  - extra: 2/3
  - loose: 0/3
  - missing: 1/3
  - wrong_class: 2/3
  - lọt: img_018.jpg (loose), img_026.jpg (extra), img_003.jpg (missing), img_031.jpg (wrong_class), img_021.jpg (loose), img_025.jpg (loose), img_014.jpg (missing)

## Gợi ý đọc số
- `deleted` cao → ngưỡng thấp hoặc ánh xạ nhãn sai; `added` cao → máy bỏ sót (xem lớp nào);
- `accepted` thấp (< 50%) → sửa mỗi box lâu hơn vẽ mới: đổi model/ngưỡng hoặc vẽ tay;
- bắt bẫy < 80% → duyệt lại lô, siết checklist (quét ảnh TRƯỚC khi nhìn nhãn máy).
