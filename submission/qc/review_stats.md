# Số đo review (det)

- Ảnh tính: 24 (batch,gold); box máy: 690; đối tượng sau duyệt: 648; ngưỡng accepted: IoU >= 0.9
- Đã **loại 21 đối tượng nằm quanh các bẫy** khỏi bảng này: lỗi nhân tạo không phản ánh chất lượng máy.

## 1. Bốn trạng thái (AI error profile)
| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted (giữ nguyên) | 570 | 570/690 = 82.6% | 690 box máy |
| edited (sửa) | 25 | 25/690 = 3.6% | 690 box máy |
| deleted (xoá) | 92 | 92/690 = 13.3% | 690 box máy |
| added (vẽ thêm) | 52 | 52/648 = 8.0% | 648 đối tượng sau duyệt |

Trong số deleted + added, **12 trường hợp là đổi lớp** (máy gán sai lớp, box đúng chỗ): đó không phải máy bỏ sót. Số vật máy **sót thật** ≈ added − đổi lớp = 40.

### Theo lớp
| Lớp | accepted | edited | deleted | added |
|---|---|---|---|---|
| person | 340 | 8 | 30 | 12 |
| bicycle | 4 | 0 | 3 | 5 |
| motorbike | 159 | 11 | 28 | 18 |
| car | 50 | 0 | 4 | 9 |
| bus | 2 | 1 | 0 | 3 |
| truck | 0 | 0 | 6 | 2 |
| three_wheeler | 15 | 5 | 21 | 3 |

## 2. So với gold (nhãn gán tay từ đầu)
Ảnh gold: 4; đối tượng gold: 86 (mẫu số của recall)
| | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| Máy (pre-label) | 65 | 22 | 21 | 74.7% (KTC 95%: 65–83%) | 75.6% (KTC 95%: 66–83%) |

*Lạc quan: ngưỡng được chọn trên chính các ảnh gold này (in-sample), và chỉ có 4 ảnh. Đọc khoảng tin cậy, đừng đọc con số đơn.*


(Ảnh gold không nằm trong nhãn sau duyệt: không tính residual / new error. Đó là chủ ý: duyệt lại ảnh bạn vừa tự gán tay sẽ đo trí nhớ, không đo thiên kiến.)

## 3. Ảnh bẫy
- Bắt được 10/12 = 83.3% (KTC 95%: 55–95%) (ngưỡng đạt: 80%). Chỉ 12 bẫy nên khoảng tin cậy rất rộng: đọc cùng danh sách bẫy lọt, đừng kết luận chỉ từ %.
  - extra: 3/3
  - loose: 3/3
  - missing: 1/3
  - wrong_class: 3/3
  - lọt: img_003.jpg (missing), img_014.jpg (missing)

## Gợi ý đọc số
- `deleted` cao → ngưỡng thấp hoặc ánh xạ nhãn sai; `added` cao → máy bỏ sót (xem lớp nào);
- `accepted` thấp (< 50%) → sửa mỗi box lâu hơn vẽ mới: đổi model/ngưỡng hoặc vẽ tay;
- bắt bẫy < 80% → duyệt lại lô, siết checklist (quét ảnh TRƯỚC khi nhìn nhãn máy).
