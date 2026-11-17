# Báo cáo QC tự động (nháp)

> **Gợi ý** do script tính; bạn xác nhận. Script chỉ đếm theo phiếu kiểm bằng mắt người.

## Mẫu
- 6 ảnh (1 ca khó + 5 ngẫu nhiên), 175 đối tượng.
- taxonomy v1.0

## Số đo
| Chỉ số | Giá trị | Mẫu số |
|---|---|---|
| Lỗi **theo đối tượng** | 1/175 = 0.6% | 175 đối tượng trong mẫu |
| **Ảnh có lỗi** (dùng cho ngưỡng) | 1/6 = 16.7% | 6 ảnh trong mẫu |
| Máy: accepted / edited / deleted | 571/690 = 82.8% / 25/690 = 3.6% / 91/690 = 13.2% | 690 box máy |
| Máy: added | 51/648 = 7.9% | 648 đối tượng sau duyệt |
| Bắt lỗi bẫy | 10/12 = 83.3% | 12 bẫy |

## Top 3 loại lỗi
| Loại lỗi | Số lỗi | Ảnh ví dụ |
|---|---|---|
| sai_lop | 1 | img_018.jpg |

## Giới hạn thống kê
- 1 ảnh lỗi trong 6 ảnh mẫu: cận trên 95% (Wilson) của tỉ lệ ảnh lỗi thật là **56%**. Ngưỡng 2% không thể được **chứng minh** bằng mẫu nhỏ như vậy: hãy nói rõ giới hạn này trong báo cáo.

## Gợi ý quyết định
**REWORK**

- ảnh có lỗi 16.7% > 2%

Cây quyết định: Hold (guideline chưa rõ / dữ liệu hỏng) -> Rework (> ngưỡng, lỗi lặp, bắt bẫy / nhất quán thấp) -> Release.
