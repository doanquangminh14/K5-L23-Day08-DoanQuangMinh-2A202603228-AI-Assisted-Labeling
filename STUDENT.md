# Thông tin học viên

- Họ và tên: Đoàn Quang Minh
- MSHV: 2A202603228
- Lớp (L2 / L3 / khác): L2
- Tài khoản GitHub: doanquangminh14
- Repo bài nộp (URL): https://github.com/doanquangminh14/K5-L23-Day08-DoanQuangMinh-2A202603228-AI-Assisted-Labeling
- Commit chốt bài (hash): 6b9828f
- Máy: Windows 11 Home, GPU NVIDIA RTX 3050 Laptop 6GB (CUDA)
- CVAT (Docker): v2.74.1 (server + UI), chạy local tại localhost:8080

## Kết quả tóm tắt (điền cuối bài; số lấy từ `submission/qc/*`, kèm mẫu số)

- Máy: accepted / edited / deleted: 570/690 = 82.6% / 25/690 = 3.6% / 92/690 = 13.3% (mẫu số: 690 box máy); added: 52/648 = 8.0% (mẫu số: 648 đối tượng sau duyệt)
- So với gold: precision / recall của **máy** 74.7% (65/87) / 75.6% (65/86 đối tượng gold)
- Bắt bẫy: lần đầu 5/12 = 41.7%; sau duyệt lại 10/12 = 83.3% (2 bẫy còn lọt có đáp án sai, xem REPORT mục 3)
- Thời gian: gán tay 7.3 phút/ảnh (29/4); duyệt có pre-label 2.7 phút/ảnh (65/24)
- Lỗi QC: vòng 1: 42.9% theo ảnh (3/7), 2.3% theo đối tượng (4/174); vòng 2: 16.7% theo ảnh (1/6), 0.6% theo đối tượng (1/175)
- **Quyết định:** Rework — lý do một dòng: ảnh lỗi 3/7 và 1/6 vượt xa ngưỡng 2%, có lỗi lặp three_wheeler (img_003, img_014, img_018), mẫu nhỏ không chứng minh được ≤ 2%
