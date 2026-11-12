# TÀI LIỆU ÔN TẬP – GIẢI THÍCH CHI TIẾT PROJECT COMPUTER VISION (INS3155)

> Tài liệu này giải thích **cách làm** và **nguyên lý cơ bản** của từng phần trong project, viết bằng lời thường, không dùng công thức toán. Mục tiêu là đọc hiểu và học thuộc để trình bày / vấn đáp.
>
> Cách đọc gợi ý: đọc phần **Tổng quan** trước, sau đó học từng Part theo thứ tự: *Mục tiêu → Khái niệm → Các bước làm → Chỉ số đánh giá → Kết quả → Nhận xét*. Cuối tài liệu có **Bộ câu hỏi vấn đáp** và **Tóm tắt 1 trang**.

---

## MỤC LỤC

0. [Tổng quan project](#0-tổng-quan-project)
1. [Part A – Lọc nhiễu và làm nét ảnh](#1-part-a--lọc-nhiễu-và-làm-nét-ảnh-image-filtering)
2. [Part B – Tái tạo 3D từ cặp ảnh stereo (Teddy)](#2-part-b--tái-tạo-3d-từ-cặp-ảnh-stereo-teddy)
3. [Part C – Ghép ảnh toàn cảnh (Panorama Stitching)](#3-part-c--ghép-ảnh-toàn-cảnh-panorama-stitching)
4. [Bảng so sánh tổng hợp](#4-bảng-so-sánh-tổng-hợp)
5. [Bộ câu hỏi vấn đáp thường gặp](#5-bộ-câu-hỏi-vấn-đáp-thường-gặp)
6. [Tóm tắt 1 trang để học thuộc](#6-tóm-tắt-1-trang-để-học-thuộc)

---

## 0. TỔNG QUAN PROJECT

### 0.1. Đề bài yêu cầu gì?

Đề bài yêu cầu dùng **xử lý ảnh truyền thống** (không được dùng deep learning, không dùng mạng nơ-ron huấn luyện sẵn) để làm 3 việc:

| Phần | Tên | Câu hỏi cần trả lời | Điểm |
|---|---|---|---|
| Part A | Lọc ảnh | Làm sao xoá nhiễu khỏi ảnh mà vẫn giữ được chi tiết? Bộ lọc nào tốt nhất cho loại nhiễu nào? | 20 |
| Part B | Tái tạo 3D | Từ 2 ảnh chụp lệch nhau một chút, làm sao biết vật nào gần, vật nào xa và dựng lại hình 3D? | 25 |
| Part C | Ghép ảnh | Từ 4 ảnh chụp chồng lấn nhau, làm sao ghép thành 1 ảnh toàn cảnh liền mạch? | 25 |

Ngoài ra còn 15 điểm **phân tích so sánh** (bắt buộc so sánh ít nhất 2 phương pháp mỗi phần, có cả **số liệu** và **nhận xét bằng mắt**), 10 điểm chất lượng báo cáo, 5 điểm code.

### 0.2. Thư viện sử dụng

- **OpenCV (cv2)**: thư viện xử lý ảnh chính. Gần như mọi thuật toán (lọc, SIFT, ORB, BM, SGBM, RANSAC, homography) đều lấy từ đây.
- **NumPy**: xử lý ảnh dưới dạng mảng số (mỗi ảnh là một bảng số, mỗi ô là một pixel).
- **scikit-image**: dùng để tính các chỉ số chất lượng ảnh PSNR, SSIM, MSE ở Part A.
- **Pandas**: lưu bảng kết quả ra file CSV.
- **Matplotlib**: vẽ hình, ghép lưới ảnh, vẽ đám mây điểm 3D.

### 0.3. Cấu trúc thư mục

| Thư mục / file | Nội dung |
|---|---|
| `part_a_filtering.py` | Code Part A |
| `part_b_stereo_3d.py` | Code Part B |
| `part_c_stitching.py` | Code Part C |
| `data/` | Ảnh đầu vào: ảnh kiến trúc `reference_image.jpg`, 4 ảnh `view_1..4.png`, thư mục `teddy/` chứa cặp ảnh stereo |
| `results/part_a, part_b, part_c` | Ảnh kết quả và bảng số liệu CSV của từng phần |
| `report_latex/` | Báo cáo LaTeX (`main.tex`) và hình trong báo cáo |

### 0.4. Cách chạy

```bash
pip install -r requirements.txt
python part_a_filtering.py
python part_b_stereo_3d.py
python part_c_stitching.py
```

Mỗi script tự tạo thư mục kết quả, tự lưu hình vào cả `results/` và `report_latex/figures/`, và in bảng số liệu ra màn hình.

### 0.5. Ý tưởng chung của cả project

Cả 3 phần đều theo cùng một khuôn:

1. **Chuẩn bị dữ liệu** (tạo ảnh nhiễu / đọc cặp ảnh stereo / cắt 4 ảnh chồng lấn).
2. **Chạy nhiều phương pháp hoặc nhiều tham số khác nhau** trên cùng dữ liệu.
3. **Đo bằng số** (độ chính xác, sai số, thời gian chạy).
4. **Vẽ hình** để so sánh bằng mắt.
5. **Rút ra nhận xét**: phương pháp nào tốt ở đâu, kém ở đâu, vì sao.

---

## 1. PART A – LỌC NHIỄU VÀ LÀM NÉT ẢNH (IMAGE FILTERING)

### 1.1. Mục tiêu

- Tự tạo ra ảnh bị nhiễu từ một ảnh sạch.
- Dùng 4 bộ lọc để xử lý: **Mean**, **Gaussian**, **Median**, **Laplacian Sharpening**.
- Thử mỗi bộ lọc với 3 kích thước cửa sổ: **3×3, 5×5, 7×7**.
- So sánh xem bộ lọc nào xoá nhiễu tốt nhất mà vẫn giữ được đường nét.

### 1.2. Các khái niệm cơ bản cần nhớ

**Ảnh số là gì?**
Ảnh là một bảng gồm rất nhiều ô vuông nhỏ gọi là **pixel**. Với ảnh xám, mỗi pixel có một giá trị độ sáng từ 0 (đen tuyền) đến 255 (trắng tinh). Ảnh trong Part A có kích thước 640×480 pixel.

**Nhiễu là gì?**
Nhiễu là những thay đổi ngẫu nhiên, không mong muốn làm sai lệch giá trị pixel so với ảnh thật. Nhìn bằng mắt sẽ thấy ảnh bị lấm tấm, sần sùi hoặc có chấm đen trắng.

**Hai loại nhiễu được dùng trong project:**

| Loại nhiễu | Hình dung | Nguồn gốc ngoài đời | Cách tạo trong code |
|---|---|---|---|
| **Gaussian noise** (nhiễu Gauss) | Ảnh bị "sần" đều khắp nơi, như hạt cát mịn phủ lên toàn ảnh | Cảm biến máy ảnh nóng lên, chụp thiếu sáng, ISO cao | Mỗi pixel được cộng thêm một lượng ngẫu nhiên nhỏ, lúc sáng hơn lúc tối hơn. Độ mạnh của nhiễu (sigma) = 25 |
| **Salt-and-Pepper noise** (nhiễu muối tiêu) | Những chấm trắng tinh và đen tuyền rải rác, như rắc muối và tiêu | Lỗi truyền dữ liệu, điểm ảnh chết trên cảm biến | 4% số pixel bị đổi thành trắng (muối), 4% bị đổi thành đen (tiêu). Tổng 8% pixel bị hỏng |

> Điểm khác biệt quan trọng: nhiễu Gauss **làm lệch nhẹ tất cả** các pixel, còn nhiễu muối tiêu **làm hỏng hoàn toàn một số ít** pixel (các pixel còn lại vẫn đúng 100%).

**Bộ lọc (filter) và cửa sổ trượt (kernel):**
Bộ lọc hoạt động bằng cách cho một **cửa sổ nhỏ** (ví dụ 3×3 = 9 pixel) trượt qua từng vị trí trên ảnh. Tại mỗi vị trí, bộ lọc nhìn vào các pixel nằm trong cửa sổ, tính ra một giá trị mới và gán cho pixel ở giữa. Cửa sổ càng lớn thì pixel mới càng bị ảnh hưởng bởi nhiều hàng xóm hơn, ảnh càng mượt nhưng càng mờ.

### 1.3. Bốn bộ lọc – nguyên lý hoạt động

#### (1) Mean Filter – Lọc trung bình (Box filter)

- **Nguyên lý:** thay mỗi pixel bằng **giá trị trung bình cộng** của tất cả pixel trong cửa sổ. Mọi pixel trong cửa sổ có vai trò ngang nhau.
- **Hình dung:** giống như hỏi ý kiến cả nhóm hàng xóm rồi lấy trung bình.
- **Ưu điểm:** cực kỳ đơn giản, chạy rất nhanh.
- **Nhược điểm:**
  - Làm **mờ cạnh** (đường biên giữa vùng sáng và vùng tối bị nhoè ra) vì nó trộn cả pixel hai bên cạnh vào nhau.
  - Với nhiễu muối tiêu thì **thất bại**: một chấm trắng 255 khi lấy trung bình sẽ không biến mất mà bị loang ra thành một vệt xám.

#### (2) Gaussian Filter – Lọc Gauss

- **Nguyên lý:** cũng lấy trung bình, nhưng là **trung bình có trọng số**. Pixel ở giữa cửa sổ được coi trọng nhất, càng ra xa trọng số càng giảm dần (theo hình quả chuông).
- **Tham số sigma** quyết định "quả chuông" rộng hay hẹp. Sigma lớn thì làm mờ mạnh hơn. Trong project: cửa sổ 3×3 dùng sigma 0.8, 5×5 dùng 1.2, 7×7 dùng 1.6.
- **Hình dung:** hỏi ý kiến hàng xóm, nhưng hàng xóm sát vách được tin nhiều hơn hàng xóm ở xa.
- **Ưu điểm:** làm mượt tự nhiên hơn Mean, ít làm nhoè cạnh hơn. Rất hợp với **nhiễu Gauss**.
- **Nhược điểm:** vẫn là phép trung bình nên cũng **không xử lý được nhiễu muối tiêu**.

#### (3) Median Filter – Lọc trung vị

- **Nguyên lý:** xếp tất cả pixel trong cửa sổ theo thứ tự từ tối đến sáng, rồi lấy **giá trị nằm ở giữa** dãy.
- **Hình dung:** trong 9 pixel, nếu có 1 chấm trắng 255 lạc loài, khi xếp hàng nó sẽ đứng ở cuối dãy và **không bao giờ được chọn**. Giá trị ở giữa là giá trị "bình thường" của vùng đó.
- **Ưu điểm:**
  - **Vô địch với nhiễu muối tiêu**: loại bỏ hoàn toàn các chấm đen trắng.
  - **Giữ cạnh tốt**: vì giá trị đầu ra luôn là một giá trị có thật trong cửa sổ, không tạo ra giá trị trung gian mờ nhạt.
- **Nhược điểm:** chậm hơn vì phải sắp xếp, cửa sổ càng to càng chậm (7×7 tốn khoảng 18 ms so với dưới 1 ms của Mean/Gaussian). Cửa sổ to sẽ làm mất các chi tiết mảnh (góc nhọn, đường kẻ nhỏ).
- Median là bộ lọc **phi tuyến** (không phải phép cộng có trọng số), còn Mean và Gaussian là bộ lọc **tuyến tính**.

#### (4) Laplacian Sharpening – Làm nét bằng Laplacian

- **Đây không phải bộ lọc xoá nhiễu**, mà là bộ lọc **làm nét** (tăng độ sắc của cạnh).
- **Nguyên lý:** toán tử Laplacian phát hiện những chỗ **độ sáng thay đổi đột ngột** (tức là cạnh, đường viền). Sau đó lấy ảnh gốc **trừ đi một phần** kết quả Laplacian này. Kết quả là phía sáng của cạnh sáng hơn, phía tối tối hơn, nên cạnh trông sắc hơn.
- **Hệ số alpha** điều chỉnh mức độ làm nét: 3×3 dùng 0.25, 5×5 dùng 0.12, 7×7 dùng 0.03 (cửa sổ lớn cho phản ứng mạnh hơn nên phải giảm alpha).
- **Cách áp dụng trong project:** làm nét **trên ảnh đã được khử nhiễu trước** (không làm nét trực tiếp ảnh nhiễu):
  - Với nhiễu Gauss: khử bằng Gaussian rồi mới làm nét.
  - Với nhiễu muối tiêu: khử bằng Median rồi mới làm nét.
- **Nhược điểm:**
  - Làm nét cũng **khuếch đại phần nhiễu còn sót lại**.
  - Tạo ra **hiệu ứng quầng sáng (halo) / gợn sóng (ringing)**: một viền sáng hoặc viền tối mỏng chạy dọc theo cạnh.
  - Vì vậy PSNR của Laplacian luôn thấp nhất, mặc dù mắt người có thể thấy ảnh "sắc" hơn.

### 1.4. Các bước code làm (theo đúng thứ tự trong `part_a_filtering.py`)

1. Đọc ảnh `data/reference_image.jpg` (ảnh tường kiến trúc có hoa văn hình học).
2. Chuyển sang **ảnh xám** và thu nhỏ về 640×480. Lưu thành `data/clean_gray.png`. Đây là **ảnh sạch chuẩn** để so sánh.
3. Cố định bộ sinh số ngẫu nhiên (seed = 42) để mỗi lần chạy đều ra cùng một kết quả (tính **tái lập**).
4. Tạo 2 ảnh nhiễu: một ảnh nhiễu Gauss, một ảnh nhiễu muối tiêu.
5. Với **mỗi loại nhiễu** và **mỗi kích thước cửa sổ** (3, 5, 7):
   - Chạy Mean, Gaussian, Median, Laplacian Sharpening.
   - Bấm giờ thời gian chạy của từng bộ lọc.
   - So ảnh kết quả với ảnh sạch để tính MSE, PSNR, SSIM.
6. Vẽ lưới ảnh 4×4 cho mỗi loại nhiễu: hàng đầu là ảnh gốc và ảnh nhiễu, 3 hàng sau là kết quả với cửa sổ 3×3, 5×5, 7×7. Lưu thành `grid_Gaussian.png` và `grid_SaltPepper.png`.
7. Lưu toàn bộ số liệu vào `quantitative_part_a.csv`.

### 1.5. Các chỉ số đánh giá (hiểu ý nghĩa, không cần công thức)

| Chỉ số | Ý nghĩa đơn giản | Tốt khi |
|---|---|---|
| **MSE** (Mean Squared Error) | Trung bình độ chênh lệch giữa ảnh kết quả và ảnh sạch (chênh lệch được bình phương lên để phạt nặng các lỗi lớn) | **Càng thấp càng tốt** |
| **PSNR** (Peak Signal-to-Noise Ratio), đơn vị dB | Đổi MSE sang thang đo dB cho dễ đọc. MSE càng nhỏ thì PSNR càng lớn. Thường trên 30 dB là rất tốt, 20–30 dB là chấp nhận được | **Càng cao càng tốt** |
| **SSIM** (Structural Similarity) | Đo mức độ giống nhau về **cấu trúc** (độ sáng, độ tương phản, hoa văn) theo cách gần với mắt người cảm nhận. Giá trị từ 0 đến 1 | **Càng gần 1 càng tốt** |
| **Time (ms)** | Thời gian chạy bộ lọc, tính bằng mili giây | **Càng thấp càng tốt** |

> Mẹo nhớ: **PSNR đo "sai bao nhiêu" từng pixel, SSIM đo "giống về hình dáng" như mắt người nhìn.** Có trường hợp PSNR cao nhưng SSIM thấp hơn và ngược lại.

### 1.6. Kết quả chính

**Với nhiễu Gauss:**

| Bộ lọc | Cửa sổ | PSNR (dB) | SSIM | Ghi chú |
|---|---|---|---|---|
| Gaussian (sigma 0.8) | 3×3 | **24.77** | 0.677 | PSNR cao nhất |
| Median | 3×3 | 24.45 | 0.644 | |
| Mean | 3×3 | 23.73 | 0.687 | |
| Gaussian (sigma 1.6) | 7×7 | 21.86 | **0.741** | SSIM cao nhất |
| Laplacian | 7×7 | 13.03 | 0.342 | Tệ nhất |

**Với nhiễu muối tiêu:**

| Bộ lọc | Cửa sổ | PSNR (dB) | SSIM | Ghi chú |
|---|---|---|---|---|
| **Median** | 3×3 | **26.80** | **0.929** | Thắng tuyệt đối |
| Median | 5×5 | 23.77 | 0.874 | |
| Gaussian | 3×3 | 21.82 | 0.553 | Thất bại |
| Mean | 3×3 | 21.66 | 0.561 | Thất bại |

### 1.7. Nhận xét (phần quan trọng nhất để trình bày)

1. **Nhiễu Gauss thì dùng Gaussian, nhiễu muối tiêu thì dùng Median.** Đây là kết luận cốt lõi.
2. **Cửa sổ nhỏ (3×3) cho PSNR tốt nhất** vì ảnh kiến trúc có rất nhiều hoa văn và đường kẻ mảnh. Cửa sổ to xoá nhiễu mạnh hơn nhưng cũng xoá luôn chi tiết, gây hiện tượng **làm mờ quá mức (over-smoothing)**.
3. **SSIM cao nhất lại ở Gaussian 7×7** với nhiễu Gauss: vì cửa sổ to làm các vùng tường phẳng trở nên rất mịn, về mặt "cấu trúc" thì giống ảnh gốc hơn, dù từng pixel lệch nhiều hơn.
4. **Mean và Gaussian thất bại với nhiễu muối tiêu** vì chúng lấy trung bình, nên chấm trắng/đen bị loang ra thành vệt xám thay vì biến mất.
5. **Median giữ cạnh tốt** vì nó chọn một giá trị có thật chứ không trộn giá trị.
6. **Laplacian làm cạnh sắc hơn nhưng PSNR giảm mạnh**, đặc biệt với cửa sổ 5×5 và 7×7, vì nó khuếch đại nhiễu còn sót và tạo quầng sáng (halo) dọc theo cạnh.
7. **Về tốc độ**: Mean và Gaussian luôn dưới 1 ms. Median chậm dần theo kích thước cửa sổ (7×7 mất khoảng 18 ms) vì phải sắp xếp nhiều giá trị.

---

## 2. PART B – TÁI TẠO 3D TỪ CẶP ẢNH STEREO (TEDDY)

### 2.1. Mục tiêu

- Từ **2 ảnh chụp cùng một cảnh từ 2 vị trí lệch ngang nhau** (ảnh trái và ảnh phải), tìm ra **độ sâu** (vật nào gần, vật nào xa) của từng pixel.
- Các việc cụ thể theo đề:
  1. Tìm các cặp điểm tương ứng giữa 2 ảnh, ước lượng **ma trận cơ bản F** bằng **RANSAC**, vẽ **đường epipolar** lên 2 ảnh.
  2. Tính **bản đồ disparity dày đặc** bằng 2 phương pháp **BM** và **SGBM**.
  3. Thử nhiều tham số (`blockSize`, `numDisparities`).
  4. Dựng **đám mây điểm 3D (point cloud)**.

### 2.2. Nguyên lý stereo – hiểu bằng đôi mắt

- Hãy giơ một ngón tay trước mặt, lần lượt nhắm mắt trái rồi mắt phải. Ngón tay sẽ **"nhảy" qua lại** so với cảnh nền phía sau.
- Ngón tay **càng gần mắt thì nhảy càng nhiều**, vật ở xa (bức tường) gần như không nhảy.
- Độ "nhảy" đó, tức **khoảng lệch theo chiều ngang** của cùng một điểm giữa ảnh trái và ảnh phải, gọi là **disparity** (độ chênh lệch / thị sai), đo bằng pixel.
- **Quy tắc vàng cần thuộc:**
  > **Disparity lớn → vật ở gần. Disparity nhỏ → vật ở xa.**
  > Độ sâu tỉ lệ nghịch với disparity.
- Để đổi disparity ra khoảng cách thật (mét) cần biết thêm 2 thông số của hệ camera:
  - **Tiêu cự (focal length)**: độ "zoom" của camera, tính bằng pixel.
  - **Baseline**: khoảng cách giữa 2 camera (giống khoảng cách giữa 2 mắt).

### 2.3. Dữ liệu: bộ Teddy của Middlebury

- **Middlebury** là bộ dữ liệu stereo chuẩn nổi tiếng nhất thế giới, dùng để so sánh các thuật toán stereo. Bộ **Teddy** (năm 2003) có cảnh gồm: chú gấu bông hồng, con ếch nhồi bông xanh, ngôi nhà gỗ đỏ, chậu cây, tranh vẽ và tấm áp phích bảng tuần hoàn phía sau.
- 4 file đã dùng (nằm trong `data/teddy/`):

| File | Vai trò |
|---|---|
| `im2.png` | **Ảnh trái** (450×375 pixel) |
| `im6.png` | **Ảnh phải** |
| `disp2.png` | **Disparity chuẩn (ground truth)** của ảnh trái |
| `disp6.png` | **Disparity chuẩn** của ảnh phải |

- **Ground truth** là đáp án đúng, được đo bằng kỹ thuật chiếu ánh sáng có cấu trúc (structured light) nên rất chính xác. Nhờ có đáp án, ta tính được thuật toán **sai bao nhiêu** (RMSE, tỉ lệ pixel sai).
- **Lưu ý khi đọc ground truth:**
  - Giá trị trong file đã được **nhân 4** để lưu cho chính xác, nên phải **chia 4** mới ra disparity thật. Sau khi chia, disparity thật nằm trong khoảng **12.5 đến 52.75 pixel**.
  - Pixel có giá trị **0 nghĩa là "không biết"** (không có đáp án), không dùng để chấm điểm.
- **Ảnh Teddy đã được "rectify" (chỉnh song song)**: hai ảnh đã được căn sao cho **một điểm ở hàng nào trong ảnh trái thì cũng nằm đúng hàng đó trong ảnh phải**, chỉ lệch theo chiều ngang. Điều này rất quan trọng vì BM và SGBM chỉ tìm kiếm theo hàng ngang.

### 2.4. Vùng bị che khuất (occlusion) và mặt nạ non-occluded

- **Occlusion** là những vùng **chỉ một mắt nhìn thấy**, mắt còn lại bị vật khác che mất. Ví dụ: phần tường ngay sát bên trái ngôi nhà gỗ, ảnh trái thấy nhưng ảnh phải bị ngôi nhà che.
- Ở vùng này **không thể tìm được điểm tương ứng**, nên không có thuật toán nào tính đúng được. Nếu chấm điểm ở đây là không công bằng.
- **Cách project xác định vùng không bị che (non-occluded):** dùng cả 2 ground truth để **kiểm tra chéo trái – phải**:
  1. Với mỗi pixel ở ảnh trái, dùng disparity chuẩn của nó để nhảy sang vị trí tương ứng ở ảnh phải.
  2. Đọc disparity chuẩn của ảnh phải tại vị trí đó.
  3. Nếu 2 giá trị **khớp nhau (lệch không quá 1 pixel)** thì pixel đó được cả 2 mắt nhìn thấy, gọi là **non-occluded**.
- Kết quả: **87.26%** số pixel là non-occluded. Mọi chỉ số độ chính xác chỉ tính trên vùng này.

### 2.5. Bước 1 – Tìm điểm tương ứng bằng SIFT

- **Keypoint (điểm đặc trưng)** là những điểm "nổi bật", dễ nhận ra ở cả 2 ảnh, ví dụ góc cạnh, đốm, chỗ hoa văn đặc biệt. Vùng tường trơn thì không có keypoint vì chỗ nào trông cũng giống nhau.
- **SIFT** (Scale-Invariant Feature Transform) là thuật toán tìm keypoint nổi tiếng:
  - Tìm điểm ở **nhiều mức phóng to/thu nhỏ** khác nhau, nên vật dù to hay nhỏ đều tìm được (bất biến với tỉ lệ).
  - Với mỗi điểm, tạo ra một **descriptor (bộ mô tả)** gồm 128 con số, mô tả hướng của các đường nét xung quanh điểm đó. Có thể coi descriptor là **"dấu vân tay"** của điểm.
  - Bền vững khi ảnh bị xoay, phóng to thu nhỏ, hoặc thay đổi độ sáng.
- **Ghép cặp (matching):** với mỗi điểm ở ảnh trái, tìm **2 điểm có dấu vân tay giống nhất** ở ảnh phải (gọi là KNN với K = 2).
- **Lowe's ratio test (kiểm tra tỉ lệ của Lowe), ngưỡng 0.75:**
  - Nếu điểm giống nhất **giống hơn hẳn** điểm giống thứ nhì (khoảng cách nhỏ hơn 75% của điểm thứ nhì) thì **chấp nhận**.
  - Nếu 2 ứng viên giống nhau xấp xỉ thì **loại bỏ**, vì không chắc chắn (thường xảy ra ở hoa văn lặp lại, ví dụ các ô trong bảng tuần hoàn).
- Kết quả trên Teddy: ảnh trái có **706** keypoint, ảnh phải có **729**, sau ratio test còn **337 cặp** khớp.

### 2.6. Bước 2 – Ma trận cơ bản F và RANSAC

**Ma trận cơ bản F (Fundamental Matrix) là gì?**
- Là một bảng số 3×3 **mô tả mối quan hệ hình học giữa 2 camera**.
- Ý nghĩa thực tế: **cho một điểm ở ảnh trái, F cho biết điểm tương ứng ở ảnh phải chắc chắn nằm trên một đường thẳng nào đó**. Đường thẳng này gọi là **đường epipolar**.
- Nhờ đó, thay vì phải tìm điểm tương ứng trên **cả bức ảnh**, ta chỉ cần tìm **dọc theo một đường thẳng**. Đây là **ràng buộc epipolar**, nền tảng của mọi thuật toán stereo.
- Cần tối thiểu khoảng 8 cặp điểm đúng để tính được F.

**RANSAC là gì?** (Random Sample Consensus, "đồng thuận từ mẫu ngẫu nhiên")
- Vấn đề: trong 337 cặp khớp vẫn có những cặp **ghép sai** (gọi là **outlier**). Nếu đưa cả cặp sai vào tính thì F sẽ sai.
- Cách RANSAC làm, lặp đi lặp lại nhiều lần:
  1. **Chọn ngẫu nhiên** một nhóm nhỏ các cặp điểm (vừa đủ để tính F).
  2. Tính ra một F thử nghiệm từ nhóm đó.
  3. **Đếm xem bao nhiêu cặp còn lại "đồng ý"** với F này (tức là nằm gần đường epipolar, cách không quá ngưỡng 1 pixel). Những cặp đồng ý gọi là **inlier**.
  4. Ghi nhớ F nào có **nhiều inlier nhất**.
- Cuối cùng giữ lại F tốt nhất. Các cặp không đồng ý bị coi là ghép sai và bị loại.
- **Hình dung:** giống bỏ phiếu. Mỗi lần thử đưa ra một "ứng cử viên", ứng cử viên được nhiều phiếu nhất thắng. Những cặp sai không thể bầu cho cùng một đáp án nên tự bị loại.
- Kết quả: **315/337 cặp là inlier (93.47%)**.

**Đường epipolar trên hình (`epipolar_lines.png`):**
- Code chọn 20 cặp inlier rải đều từ trên xuống dưới, vẽ chấm tròn tại điểm và vẽ đường epipolar cùng màu ở cả 2 ảnh.
- Vì Teddy đã được rectify nên **các đường epipolar đều nằm ngang**, và điểm ở ảnh trái và ảnh phải nằm trên **cùng một hàng**. Đây là dấu hiệu kiểm chứng F ước lượng đúng.

**Sai số epipolar (đo chất lượng của F):**

| Chỉ số | Ý nghĩa | Kết quả |
|---|---|---|
| **Mean Epipolar Error** | Trung bình khoảng cách từ mỗi điểm đến đường epipolar mà nó lẽ ra phải nằm trên (tính cả 2 chiều trái sang phải và phải sang trái) | **0.2389 px** |
| **Mean Sampson Error** | Một cách đo sai số tương tự nhưng "công bằng" hơn, ước lượng khoảng cách nhỏ nhất cần dịch chuyển cả 2 điểm để chúng thoả mãn F | **0.1689 px** |
| Độ lệch dọc trung bình | Trung bình chênh lệch theo chiều dọc giữa điểm trái và điểm phải | **0.17 px** |

> Cả 3 đều **dưới 1 pixel** (sai số dưới mức điểm ảnh, gọi là **sub-pixel**), chứng tỏ F rất chính xác.

### 2.7. Bước 3 – Tính bản đồ disparity dày đặc: BM và SGBM

**"Dày đặc" (dense)** nghĩa là tính disparity cho **mọi pixel**, khác với SIFT chỉ tính cho vài trăm keypoint ("thưa", sparse).

**Ý tưởng chung:** với mỗi pixel ở ảnh trái, lấy một **ô vuông nhỏ** xung quanh nó, rồi **trượt ô vuông đó theo hàng ngang** trên ảnh phải để tìm vị trí trông giống nhất. Khoảng trượt tại vị trí giống nhất chính là disparity.

#### (1) BM – Block Matching (so khớp khối)

- **Nguyên lý:** với mỗi pixel, so ô vuông ở ảnh trái với từng ô vuông ứng viên dọc theo hàng ở ảnh phải. Cách so: cộng tổng độ chênh lệch độ sáng của từng cặp pixel trong 2 ô (gọi là **SAD – Sum of Absolute Differences**). Ô nào có tổng chênh lệch nhỏ nhất là ô khớp.
- Mỗi pixel được quyết định **độc lập**, không quan tâm pixel bên cạnh chọn gì. Đây là phương pháp **cục bộ (local)**.
- Chạy trên **ảnh xám**.
- **Ưu điểm:** rất nhanh.
- **Nhược điểm:**
  - Ở **vùng trơn không có hoa văn** (tường trắng, mái nhà trắng), mọi ô vuông trông như nhau nên không chọn được, kết quả bị **thủng lỗ** (invalid).
  - Ở **mép vật thể**, ô vuông chứa cả vật phía trước lẫn nền phía sau, nên vật phía trước bị **"phình to"** ra (hiện tượng **foreground fattening**).

#### (2) SGBM – Semi-Global Block Matching (so khớp khối bán toàn cục)

- **Nguyên lý:** vẫn so khớp ô vuông như BM, nhưng thêm một **luật "mượt mà"**:
  - Pixel cạnh nhau thường thuộc cùng một bề mặt nên disparity của chúng **thường bằng nhau hoặc gần bằng nhau**.
  - Nếu 2 pixel cạnh nhau có disparity **chênh 1 đơn vị** thì bị **phạt nhẹ** (tham số **P1**), cho phép bề mặt nghiêng thay đổi từ từ.
  - Nếu **chênh nhiều** thì bị **phạt nặng** (tham số **P2**), chỉ chấp nhận khi thật sự có mép vật thể.
- Luật mượt này được lan truyền **dọc theo nhiều hướng** trên ảnh (ngang, dọc, chéo), rồi gộp lại. Gọi là "bán toàn cục" vì nó không xét toàn bộ ảnh một lúc (quá chậm), mà xét theo nhiều đường thẳng 1 chiều, cho kết quả gần bằng tối ưu toàn cục.
- Chạy trên **ảnh màu** (dùng nhiều thông tin hơn).
- **Ưu điểm:** bản đồ disparity **dày đặc, mượt, ít lỗ**. Ở vùng trơn, thông tin từ các vùng có hoa văn xung quanh được "lan" vào nên vẫn đoán được.
- **Nhược điểm:** chậm hơn BM khoảng 2–2.5 lần, tốn bộ nhớ hơn.

> Mẹo nhớ: **BM là "mỗi người tự quyết", SGBM là "hỏi thêm ý kiến hàng xóm theo nhiều hướng".**

#### (3) Các tham số quan trọng

| Tham số | Ý nghĩa | Ảnh hưởng |
|---|---|---|
| **blockSize** (b) | Kích thước ô vuông dùng để so khớp | Nhỏ thì giữ chi tiết, mép sắc nhưng dễ nhiễu. Lớn thì mượt, ít nhiễu nhưng mép bị phình, mất chi tiết mảnh |
| **numDisparities** (d) | Khoảng trượt tối đa khi tìm kiếm (phải chia hết cho 16) | Phải **lớn hơn disparity lớn nhất** trong cảnh, nếu không vật ở gần sẽ bị tính sai. Nhưng quá lớn thì chậm hơn và **dải đen bên trái ảnh rộng ra** |
| **uniquenessRatio** = 10 | Ô khớp tốt nhất phải tốt hơn hẳn ô tốt thứ nhì (giống ratio test) | Loại các pixel mơ hồ |
| **speckleWindowSize** = 100, **speckleRange** = 2 | Lọc các "đốm" nhỏ có disparity lạc loài | Xoá các mảng lỗi nhỏ |
| **disp12MaxDiff** = 1 (chỉ SGBM) | Kiểm tra chéo trái – phải: tính disparity theo 2 chiều, nếu không khớp thì loại | Loại vùng bị che khuất |
| **P1, P2** (chỉ SGBM) | Mức phạt khi disparity thay đổi ít / nhiều giữa 2 pixel cạnh nhau | P2 lớn hơn P1 nhiều để ưu tiên bề mặt mượt |

**Vì sao có dải đen bên trái mọi bản đồ disparity?**
Các cột ngoài cùng bên trái của ảnh trái, khi trượt sang ảnh phải sẽ bị **ra ngoài mép ảnh**, nên không có chỗ để so. Độ rộng dải đen bằng đúng `numDisparities`. Vì vậy d = 96 có dải đen rộng hơn d = 64.

**Lưu ý kỹ thuật:** OpenCV trả về disparity **nhân 16** (để lưu phần lẻ), nên code chia 16. Pixel không tính được sẽ có giá trị âm, nên coi là **invalid**.

#### (4) 7 cấu hình đã thử

- BM: b = 9, 15, 21 (cùng d = 64)
- SGBM: b = 3, 5, 9 (cùng d = 64), và thêm b = 5 với d = 96

Mỗi cấu hình chạy **5 lần** rồi lấy thời gian **trung vị** để tránh sai lệch do máy tính lúc nhanh lúc chậm.

### 2.8. Các chỉ số đánh giá Part B

| Chỉ số | Ý nghĩa | Tốt khi |
|---|---|---|
| **Valid (%)** | Tỉ lệ pixel trên **toàn ảnh** mà thuật toán đưa ra được disparity | Cao |
| **Valid Non-occ. (%)** | Tỉ lệ pixel có disparity **trong vùng không bị che** | Cao |
| **RMSE (px)** | Sai số trung bình (kiểu bình phương, phạt nặng lỗi lớn) so với ground truth | Thấp |
| **Bad > 1px (%)** | Tỉ lệ pixel bị sai **hơn 1 pixel** so với đáp án | Thấp |
| **Bad > 2px (%)** | Tỉ lệ pixel bị sai **hơn 2 pixel** so với đáp án | Thấp |
| **Runtime (ms)** | Thời gian chạy | Thấp |

### 2.9. Kết quả Part B

| Phương pháp | b | d | Valid (%) | Valid Non-occ. (%) | RMSE (px) | Bad>1px (%) | Bad>2px (%) | Time (ms) |
|---|---|---|---|---|---|---|---|---|
| BM | 9 | 64 | 68.59 | 75.34 | **1.745** | **5.16** | 4.29 | 4.34 |
| BM | 15 | 64 | 70.54 | 76.80 | 1.874 | 7.39 | 6.02 | 4.43 |
| BM | 21 | 64 | 69.66 | 75.72 | 1.974 | 8.71 | 7.07 | **4.05** |
| SGBM | 3 | 64 | **81.12** | **88.21** | 2.092 | 6.35 | **3.43** | 10.42 |
| SGBM | 5 | 64 | 80.70 | 87.73 | 2.833 | 7.04 | 4.08 | 9.86 |
| SGBM | 9 | 64 | 80.88 | 87.79 | 3.071 | 9.16 | 5.81 | 8.59 |
| SGBM | 5 | 96 | 73.77 | 79.86 | 3.009 | 7.40 | 4.28 | 9.39 |

**Cấu hình tốt nhất: SGBM với b = 3, d = 64.**

### 2.10. Nhận xét Part B (học thuộc)

1. **Độ phủ (coverage):** SGBM phủ khoảng **81%** ảnh, BM chỉ khoảng **70%**. BM bị thủng lỗ ở các vùng trơn: mái nhà gỗ màu trắng, bức tranh ở góc trên bên trái, mảng tường sau gấu bông. SGBM lấp được nhờ luật mượt lan truyền từ các vùng xung quanh.
2. **Độ chính xác:** SGBM b=3 có **tỉ lệ pixel sai >2px thấp nhất (3.43%)**, tức là vừa dày đặc nhất vừa chính xác nhất.
3. **Vì sao BM có RMSE thấp hơn SGBM?** Vì BM **chỉ trả lời những pixel nó chắc chắn**, bỏ trống các pixel khó. SGBM cố gắng trả lời cả những pixel khó (ví dụ tờ báo nằm nghiêng dưới sàn), nên có một số lỗi rất lớn, mà RMSE phạt nặng lỗi lớn. **Đây là đánh đổi giữa độ phủ và độ chính xác.**
4. **Ảnh hưởng blockSize:** ô càng to thì **càng tệ** trên Teddy:
   - BM: Bad>2px tăng từ 4.29% (b=9) lên 7.07% (b=21) vì mép vật thể bị **phình to** (gấu bông, mép nhà, lá cây lan ra nền).
   - SGBM: tăng từ 3.43% (b=3) lên 5.81% (b=9) vì lá cây mảnh và cây cột nhỏ dưới ngôi nhà bị làm mượt mất.
   - Tuy nhiên ô to giúp BM giảm nhiễu lấm tấm trên tấm áp phích có hoa văn lặp.
5. **Ảnh hưởng numDisparities:** disparity thật lớn nhất chỉ khoảng **53 px**, nên d = 64 là đủ. Tăng lên 96 **không giúp chính xác hơn** mà chỉ làm dải đen bên trái rộng thêm, khiến độ phủ giảm từ 80.70% xuống 73.77%.
6. **Vùng che khuất:** cả 2 phương pháp đều đánh dấu đúng phần lớn vùng bị che là invalid (bên trái ngôi nhà, khe sau con ếch, giữa các lá cây). Lỗi còn lại tập trung ở **mép vật thể** và **cây lá có cấu trúc mảnh**.
7. **Tốc độ:** BM khoảng 4 ms, SGBM khoảng 9–10 ms, tức BM nhanh hơn khoảng 2–2.5 lần.

Hình `disparity_error_maps.png` minh hoạ điều này: màu **đỏ** là pixel sai >2px, màu **xanh dương** là pixel không tính được, màu **đen** là vùng bị che / không có đáp án.

### 2.11. Bước 4 – Dựng đám mây điểm 3D (Point Cloud)

- **Point cloud** là một tập hợp rất nhiều điểm trong không gian 3 chiều, mỗi điểm có vị trí (X, Y, Z) và màu.
- **Cách làm:**
  1. Lấy bản đồ disparity tốt nhất (SGBM b=3).
  2. Với mỗi pixel có disparity, đổi disparity thành **độ sâu Z** (disparity lớn thì Z nhỏ, tức gần).
  3. Từ vị trí pixel trên ảnh và độ sâu Z, tính ra vị trí ngang X và dọc Y trong không gian. Việc này dùng một **ma trận chiếu ngược Q** chứa tiêu cự, tâm ảnh và baseline, OpenCV có sẵn hàm `reprojectImageTo3D`.
  4. Gán cho mỗi điểm 3D **màu** của pixel tương ứng ở ảnh trái.
  5. Bỏ các điểm có disparity quá nhỏ (dưới 10 pixel) vì quá xa và không đáng tin.
- **Thông số camera:** bộ Teddy **không cung cấp thông số camera**, nên project **giả định** tiêu cự = 450 pixel và baseline = 0.16 m. Vì vậy **hình dạng 3D đúng, nhưng con số mét chỉ mang tính tương đối** (đúng tỉ lệ, không đúng kích thước tuyệt đối).
- **Lưu kết quả:** file `reconstructed_cloud.ply` (khoảng 136 nghìn điểm). Định dạng **PLY** là định dạng chuẩn cho point cloud, mở được bằng **MeshLab** hoặc **CloudCompare** để xoay xem 3D.
- **Hình `point_cloud_views.png`** gồm 3 góc nhìn:
  - SGBM nhìn chéo (oblique).
  - SGBM nhìn từ trên xuống (top-down), thấy rõ các lớp độ sâu.
  - Point cloud dựng từ **ground truth** để so sánh.
- **Nhận xét:** point cloud của SGBM tái hiện đúng các lớp độ sâu giống ground truth: **con ếch và chậu cây ở gần nhất** (khoảng 1.5–2.5 m), **ngôi nhà gỗ và gấu bông ở giữa**, **tường áp phích ở xa nhất** (khoảng 5 m). Nhìn từ trên xuống thấy rõ bức tranh nằm nghiêng bên trái và bức tường phẳng phía sau. Có vài vệt điểm lạc ở mép vật thể, tương ứng với các pixel đỏ trong hình lỗi.

### 2.12. Quy trình Part B tóm gọn

1. Đọc ảnh trái, ảnh phải, 2 ground truth (chia 4).
2. Tính mặt nạ non-occluded bằng kiểm tra chéo 2 ground truth.
3. SIFT tìm keypoint, KNN + ratio test ghép cặp.
4. RANSAC ước lượng F, đo sai số epipolar và Sampson, vẽ đường epipolar.
5. Chạy 7 cấu hình BM/SGBM, đo độ phủ, RMSE, Bad pixel, thời gian.
6. Vẽ bảng so sánh disparity và bản đồ lỗi.
7. Chọn SGBM tốt nhất, chiếu ra 3D, lưu PLY, vẽ point cloud.

---

## 3. PART C – GHÉP ẢNH TOÀN CẢNH (PANORAMA STITCHING)

### 3.1. Mục tiêu

- Ghép **4 ảnh chụp chồng lấn nhau** thành **1 ảnh toàn cảnh (panorama)** liền mạch.
- So sánh **3 thuật toán tìm đặc trưng**: **SIFT, ORB, AKAZE**.
- Thử RANSAC với **2 ngưỡng khác nhau** (2 px và 4 px).

### 3.2. Chuẩn bị 4 ảnh đầu vào

- Lấy ảnh kiến trúc `reference_image.jpg`, phóng về 1200×650.
- Cắt ra **4 ảnh con** kích thước 480×560, mỗi ảnh lệch sang phải khoảng 230 pixel so với ảnh trước, nên **2 ảnh liền kề chồng lấn nhau khoảng một nửa**. Vị trí dọc cũng lệch nhẹ mỗi ảnh một chút.
- Mỗi ảnh con bị **bóp méo phối cảnh nhẹ** (kéo lệch các góc khoảng 6 pixel) để giả lập việc cầm máy chụp tay bị nghiêng.
- Lưu thành `view_1.png` đến `view_4.png`, và hình tổng hợp `input_4_views.png`.

### 3.3. Pipeline ghép ảnh – 4 bước

#### Bước 1: Tìm đặc trưng (Feature Extraction) – so sánh 3 thuật toán

| Thuật toán | Nguyên lý cơ bản | Loại descriptor | Cách so sánh descriptor | Đặc điểm |
|---|---|---|---|---|
| **SIFT** | Tìm các "đốm" nổi bật ở nhiều mức phóng to/thu nhỏ, mô tả bằng hướng đường nét xung quanh | 128 số thực | **Khoảng cách Euclid (L2)**: đo độ chênh lệch giữa 2 dãy số | Chính xác, bền vững nhất, nhưng **chậm** |
| **ORB** | Tìm **góc** bằng FAST (kiểm tra vòng tròn pixel quanh điểm xem có sáng/tối hơn hẳn không), mô tả bằng BRIEF (so sánh độ sáng từng cặp pixel ngẫu nhiên, ghi lại 0/1), có thêm hướng xoay | Chuỗi bit 0/1 (nhị phân) | **Khoảng cách Hamming**: đếm số bit khác nhau | **Rất nhanh**, miễn phí bản quyền, nhưng kém chính xác hơn. Giới hạn tối đa 3000 điểm mỗi ảnh |
| **AKAZE** | Tạo các mức phóng to/thu nhỏ bằng cách làm mờ "thông minh" (giữ lại cạnh, chỉ làm mờ vùng phẳng), nên vị trí điểm chính xác hơn | Chuỗi bit nhị phân | Khoảng cách Hamming | **Cân bằng** giữa tốc độ và độ chính xác |

> Mẹo nhớ: **SIFT = chính xác, ORB = nhanh, AKAZE = ở giữa nhưng giữ cạnh tốt.**
> **Descriptor số thực thì so bằng L2, descriptor nhị phân thì so bằng Hamming.**

#### Bước 2: Ghép cặp đặc trưng (Feature Matching)

- Dùng **Brute-Force Matcher**: so từng descriptor ảnh này với **tất cả** descriptor ảnh kia.
- **KNN với K = 2**: lấy 2 ứng viên giống nhất.
- **Lowe's ratio test, ngưỡng 0.75** (giống Part B): chỉ giữ cặp khi ứng viên tốt nhất **tốt hơn hẳn** ứng viên thứ nhì. Rất quan trọng với ảnh kiến trúc có **hoa văn lặp lại**, nơi một điểm có thể giống nhiều điểm khác.

#### Bước 3: Ước lượng Homography bằng RANSAC

**Homography là gì?**
- Là một phép biến đổi được mô tả bằng bảng số 3×3, cho phép **"bẻ" (warp) một ảnh sao cho khớp với góc nhìn của ảnh khác**. Nó xử lý được dịch chuyển, xoay, phóng to thu nhỏ và cả **méo phối cảnh** (hình chữ nhật biến thành hình thang).
- Homography chỉ đúng khi **cảnh là một mặt phẳng** (như bức tường) hoặc khi camera **chỉ xoay tại chỗ**. Ảnh tường kiến trúc thoả điều kiện này.
- Cần tối thiểu **4 cặp điểm** để tính.
- **Khác với ma trận F ở Part B:** F chỉ cho biết điểm tương ứng nằm **trên một đường thẳng**, còn homography cho biết **chính xác vị trí** điểm tương ứng.

**RANSAC** hoạt động giống Part B: chọn ngẫu nhiên 4 cặp, tính homography, đếm số cặp đồng ý (sai lệch dưới ngưỡng), lặp nhiều lần, giữ kết quả tốt nhất.

**Ngưỡng RANSAC (2 px và 4 px):**
- Ngưỡng **nhỏ (2 px)**: khắt khe, chỉ chấp nhận cặp khớp rất sát, nên **ít inlier hơn nhưng sai số nhỏ hơn**.
- Ngưỡng **lớn (4 px)**: dễ dãi, chấp nhận cả cặp hơi lệch, nên **nhiều inlier hơn nhưng sai số lớn hơn**.

**Chọn ảnh mốc (anchor) và nối chuỗi homography:**
- Chọn **ảnh số 2 làm mốc** (ảnh ở giữa) để méo phối cảnh chia đều 2 bên, không bị dồn về một phía.
- Tính 3 homography cho 3 cặp ảnh liền kề:
  - Ảnh 1 về ảnh 2.
  - Ảnh 3 về ảnh 2.
  - Ảnh 4 về ảnh 3.
- Ảnh 4 không chồng lấn trực tiếp với ảnh 2, nên phải **nối chuỗi**: đưa ảnh 4 về ảnh 3, rồi từ ảnh 3 về ảnh 2 (nhân 2 homography với nhau).

#### Bước 4: Bẻ ảnh và trộn ảnh (Warping & Blending)

**Tính kích thước khung panorama:**
- Đưa 4 góc của mỗi ảnh qua homography của nó để biết ảnh sẽ nằm ở đâu.
- Lấy vùng bao trùm tất cả các góc, nên biết được khung panorama to bao nhiêu.
- Dịch toàn bộ sao cho không có phần nào bị ra ngoài bên trái hoặc bên trên (tọa độ âm). Khung được giới hạn tối đa 2400×1400.

**Bẻ ảnh (warping):** dùng homography đưa từng ảnh vào đúng vị trí trên khung panorama.

**Trộn ảnh bằng Feathering (làm mềm theo khoảng cách):**
- **Vấn đề:** nếu chỉ dán ảnh chồng lên nhau thì ở chỗ giao nhau sẽ thấy **đường nối (seam)** rõ ràng, do độ sáng 2 ảnh hơi khác nhau hoặc khớp chưa hoàn hảo.
- **Nguyên lý feathering:**
  1. Với mỗi ảnh, tính **khoảng cách từ mỗi pixel đến mép ảnh** gần nhất (dùng **Distance Transform**). Pixel ở giữa ảnh có khoảng cách lớn, pixel sát mép có khoảng cách nhỏ.
  2. Ở vùng chồng lấn, pixel kết quả là **trộn của 2 ảnh**, ảnh nào có pixel nằm **sâu vào trong hơn** (xa mép hơn) thì được **trọng số lớn hơn**.
  3. Kết quả là **chuyển tiếp mượt dần** từ ảnh này sang ảnh kia, không có đường nối đột ngột.
- **Hình dung:** như tô màu nước, ở chỗ giao nhau hai màu hoà dần vào nhau chứ không có ranh giới cứng.
- Các ảnh được trộn **lần lượt** vào panorama.

### 3.4. Các chỉ số đánh giá Part C

| Chỉ số | Ý nghĩa | Tốt khi |
|---|---|---|
| **Total Keypoints** | Tổng số điểm đặc trưng phát hiện được (cộng qua 3 cặp ảnh) | Nhiều (đủ để khớp) |
| **Initial Matches** | Số cặp khớp sau ratio test | Nhiều |
| **Inliers** | Số cặp được RANSAC giữ lại là đúng | Nhiều |
| **Inlier Ratio (%)** | Tỉ lệ cặp đúng trên tổng số cặp khớp | Cao |
| **Mean RMSE (px)** | **Sai số chiếu lại (reprojection error)**: dùng homography đưa điểm từ ảnh này sang ảnh kia, đo xem lệch bao nhiêu so với vị trí khớp thật | Thấp |
| **Stitching Time (ms)** | Tổng thời gian toàn bộ quy trình ghép | Thấp |

### 3.5. Kết quả Part C

| Detector | Ngưỡng RANSAC | Keypoints | Matches | Inliers | Inlier Ratio (%) | RMSE (px) | Time (ms) |
|---|---|---|---|---|---|---|---|
| SIFT | 2 px | 10,043 | 2,449 | 1,488 | 60.58 | 0.207 | 1606.60 |
| SIFT | 4 px | 10,043 | 2,449 | 1,497 | 60.95 | 0.256 | 1450.76 |
| ORB | 2 px | **17,582** | **3,113** | 1,977 | 62.66 | 0.817 | **856.84** |
| ORB | 4 px | **17,582** | **3,113** | **2,296** | **73.29** | 0.967 | 880.76 |
| AKAZE | 4 px | 7,285 | 1,951 | 1,252 | 63.84 | **0.192** | 1310.94 |

### 3.6. Nhận xét Part C (học thuộc)

1. **Độ chính xác căn chỉnh:** **AKAZE (0.192 px)** và **SIFT (0.207 px)** có sai số nhỏ hơn **ORB (0.82–0.97 px)** khoảng 4 lần. Lý do: SIFT và AKAZE xác định vị trí điểm **chính xác tới mức dưới pixel**, còn ORB định vị thô hơn.
2. **Tốc độ:** **ORB nhanh nhất** (khoảng 860 ms so với 1300–1600 ms) vì descriptor nhị phân so sánh rất nhanh bằng cách đếm bit khác nhau. ORB cũng tìm được **nhiều keypoint và inlier nhất**, tập trung ở các góc có độ tương phản cao.
3. **Ảnh hưởng ngưỡng RANSAC:** tăng ngưỡng từ 2 lên 4 px thì **số inlier tăng** (ORB từ 1977 lên 2296) nhưng **RMSE cũng tăng** (ORB từ 0.817 lên 0.967), vì chấp nhận thêm cả những cặp hơi lệch.
4. **Hoa văn lặp lại:** với SIFT và AKAZE, khoảng **36–39% cặp khớp ban đầu bị RANSAC loại**, vì ảnh kiến trúc có nhiều hoa văn giống nhau dễ gây ghép nhầm.
5. **Chất lượng hình ảnh:** feathering **xoá được đường nối và hiện tượng bóng ma (ghosting)**, các đường hoa văn chéo chạy liên tục qua toàn bộ panorama. Ảnh mốc ở giữa giúp méo phối cảnh được chia đều 2 bên.

---

## 4. BẢNG SO SÁNH TỔNG HỢP

### 4.1. Phương pháp tốt nhất mỗi phần

| Phần | Tình huống | Phương pháp tốt nhất | Lý do ngắn gọn |
|---|---|---|---|
| A | Nhiễu Gauss | Gaussian 3×3 | Trung bình có trọng số khử nhiễu đều, cửa sổ nhỏ giữ chi tiết |
| A | Nhiễu muối tiêu | Median 3×3 | Lấy giá trị giữa nên bỏ qua hoàn toàn chấm đen trắng |
| B | Disparity | SGBM b=3, d=64 | Luật mượt lấp vùng trơn, ô nhỏ giữ mép sắc |
| C | Chính xác nhất | AKAZE / SIFT | Định vị điểm dưới pixel |
| C | Nhanh nhất | ORB | Descriptor nhị phân, so sánh bằng đếm bit |

### 4.2. Các "đánh đổi" (trade-off) xuất hiện xuyên suốt project

| Đánh đổi | Ví dụ trong project |
|---|---|
| **Khử nhiễu mạnh và giữ chi tiết** | Cửa sổ lọc to thì sạch nhiễu nhưng mờ hoa văn (Part A); blockSize to thì mượt nhưng mép phình (Part B) |
| **Tốc độ và độ chính xác** | ORB nhanh nhưng kém chính xác hơn SIFT (Part C); BM nhanh nhưng thủng lỗ nhiều hơn SGBM (Part B) |
| **Độ phủ và độ tin cậy** | SGBM trả lời nhiều pixel hơn nhưng có vài lỗi lớn; BM trả lời ít hơn nhưng chắc chắn hơn (Part B) |
| **Số lượng inlier và sai số** | Ngưỡng RANSAC lớn thì nhiều inlier nhưng sai số tăng (Part C) |

### 4.3. Kỹ thuật dùng lại ở nhiều phần

| Kỹ thuật | Dùng ở | Vai trò |
|---|---|---|
| SIFT | Part B, Part C | Tìm và mô tả điểm đặc trưng |
| KNN + Lowe's ratio test (0.75) | Part B, Part C | Ghép cặp và loại cặp mơ hồ |
| RANSAC | Part B (ước lượng F), Part C (ước lượng homography) | Loại cặp ghép sai, tìm mô hình đúng |
| Cửa sổ trượt | Part A (bộ lọc), Part B (BM/SGBM so khớp khối) | Xử lý dựa trên vùng lân cận |

---

## 5. BỘ CÂU HỎI VẤN ĐÁP THƯỜNG GẶP

### Part A

**Hỏi: Tại sao Median lọc nhiễu muối tiêu tốt hơn Mean?**
Đáp: Vì Median lấy giá trị nằm giữa sau khi sắp xếp, nên các chấm trắng hoặc đen cực đoan luôn bị đẩy về 2 đầu dãy và không được chọn. Mean lấy trung bình nên chấm trắng/đen vẫn đóng góp vào kết quả, bị loang thành vệt xám.

**Hỏi: Gaussian khác Mean ở điểm nào?**
Đáp: Mean coi mọi pixel trong cửa sổ quan trọng như nhau. Gaussian ưu tiên pixel ở giữa, càng xa trọng số càng nhỏ, nên làm mượt tự nhiên hơn và ít làm nhoè cạnh hơn.

**Hỏi: Tại sao cửa sổ to lại cho PSNR thấp hơn?**
Đáp: Vì cửa sổ to làm mờ quá mức (over-smoothing), xoá luôn các chi tiết mảnh và hoa văn của ảnh kiến trúc, nên ảnh kết quả khác xa ảnh sạch.

**Hỏi: Laplacian sharpening có tác dụng gì, tại sao PSNR lại thấp?**
Đáp: Nó làm cạnh sắc nét hơn bằng cách tăng độ tương phản tại các chỗ độ sáng thay đổi đột ngột. PSNR thấp vì nó cũng khuếch đại phần nhiễu còn sót và tạo quầng sáng (halo) dọc theo cạnh.

**Hỏi: Tại sao làm nét trên ảnh đã khử nhiễu chứ không làm trực tiếp trên ảnh nhiễu?**
Đáp: Vì nhiễu cũng là những thay đổi đột ngột về độ sáng, Laplacian sẽ coi nhiễu là cạnh và khuếch đại nó lên. Khử nhiễu trước giúp chỉ làm nét các cạnh thật.

**Hỏi: PSNR và SSIM khác nhau thế nào?**
Đáp: PSNR đo sai lệch từng pixel. SSIM đo sự giống nhau về cấu trúc (độ sáng, độ tương phản, hoa văn), gần với cách mắt người cảm nhận hơn.

**Hỏi: Bộ lọc tuyến tính và phi tuyến khác nhau thế nào?**
Đáp: Bộ lọc tuyến tính (Mean, Gaussian) tính kết quả bằng cách cộng các pixel nhân trọng số. Bộ lọc phi tuyến (Median) dùng phép sắp xếp và chọn, không phải phép cộng có trọng số.

### Part B

**Hỏi: Disparity là gì, liên hệ với độ sâu ra sao?**
Đáp: Disparity là độ lệch ngang của cùng một điểm giữa ảnh trái và ảnh phải. Disparity càng lớn thì vật càng gần, càng nhỏ thì vật càng xa.

**Hỏi: Ma trận F dùng để làm gì?**
Đáp: F mô tả quan hệ hình học giữa 2 camera. Cho một điểm ở ảnh trái, F cho biết đường thẳng (đường epipolar) ở ảnh phải mà điểm tương ứng chắc chắn nằm trên. Nhờ vậy việc tìm kiếm thu gọn từ cả bức ảnh xuống một đường thẳng.

**Hỏi: Tại sao đường epipolar trên Teddy nằm ngang?**
Đáp: Vì cặp ảnh Teddy đã được chỉnh song song (rectified): 2 camera đặt ngang hàng, nên điểm tương ứng luôn nằm cùng một hàng, đường epipolar trùng với các hàng ngang.

**Hỏi: RANSAC hoạt động thế nào?**
Đáp: Lặp nhiều lần: chọn ngẫu nhiên một nhóm nhỏ cặp điểm, tính mô hình, đếm số cặp đồng ý với mô hình. Giữ mô hình được nhiều cặp đồng ý nhất. Các cặp không đồng ý bị coi là ghép sai và loại bỏ.

**Hỏi: BM và SGBM khác nhau thế nào?**
Đáp: BM so khớp từng ô vuông độc lập, mỗi pixel tự quyết, nên nhanh nhưng thủng lỗ ở vùng trơn và phình mép. SGBM thêm luật mượt: pixel cạnh nhau nên có disparity giống nhau, luật này lan truyền theo nhiều hướng, nên kết quả dày đặc và mượt hơn nhưng chậm hơn.

**Hỏi: Tại sao bản đồ disparity có dải đen bên trái?**
Đáp: Các cột ngoài cùng bên trái của ảnh trái không có vùng tương ứng trong ảnh phải để so sánh (bị ra ngoài mép). Độ rộng dải đen bằng numDisparities.

**Hỏi: Vùng occlusion là gì, xử lý thế nào?**
Đáp: Là vùng chỉ một camera nhìn thấy, camera kia bị vật khác che. Không thể tìm điểm tương ứng ở vùng này. Project dùng kiểm tra chéo 2 ground truth để loại vùng này khỏi việc chấm điểm, còn SGBM dùng kiểm tra chéo trái – phải (disp12MaxDiff) để đánh dấu invalid.

**Hỏi: Tại sao BM có RMSE thấp hơn SGBM mà SGBM vẫn được coi là tốt hơn?**
Đáp: BM chỉ trả lời các pixel nó chắc chắn, bỏ trống pixel khó. SGBM trả lời nhiều hơn (81% so với 70%), trong đó có vài pixel khó bị sai rất nhiều, mà RMSE phạt nặng lỗi lớn. Xét tỉ lệ pixel sai >2px thì SGBM b=3 thấp nhất (3.43%).

**Hỏi: blockSize to hay nhỏ thì tốt?**
Đáp: Trên Teddy, nhỏ tốt hơn. Ô to làm mượt và giảm nhiễu nhưng làm mép vật thể phình ra và mất chi tiết mảnh như lá cây.

**Hỏi: Chọn numDisparities thế nào?**
Đáp: Phải lớn hơn disparity lớn nhất trong cảnh (Teddy khoảng 53 px, nên chọn 64). Chọn quá lớn chỉ làm chậm hơn và mất thêm cột bên trái.

**Hỏi: Làm sao từ disparity ra được point cloud?**
Đáp: Đổi disparity thành độ sâu bằng tiêu cự và baseline, rồi từ vị trí pixel trên ảnh tính ra vị trí ngang và dọc trong không gian, gán màu từ ảnh trái. OpenCV làm việc này bằng ma trận Q và hàm reprojectImageTo3D.

**Hỏi: Tại sao số mét trong point cloud chỉ mang tính tương đối?**
Đáp: Vì bộ Teddy không cung cấp thông số camera, project phải giả định tiêu cự 450 px và baseline 0.16 m. Hình dạng và thứ tự gần xa vẫn đúng, chỉ có kích thước tuyệt đối là ước lượng.

### Part C

**Hỏi: Homography là gì, khác gì ma trận F?**
Đáp: Homography là phép biến đổi bẻ ảnh này cho khớp với góc nhìn ảnh kia, cho biết chính xác vị trí điểm tương ứng. Nó đúng khi cảnh phẳng hoặc camera chỉ xoay. F thì chỉ cho biết điểm tương ứng nằm trên một đường thẳng, dùng cho cảnh 3D bất kỳ.

**Hỏi: SIFT, ORB, AKAZE khác nhau thế nào?**
Đáp: SIFT dùng descriptor số thực, chính xác nhất nhưng chậm. ORB dùng descriptor nhị phân, nhanh nhất nhưng kém chính xác hơn. AKAZE dùng không gian tỉ lệ giữ cạnh, descriptor nhị phân, cân bằng giữa hai bên và trong project có sai số nhỏ nhất.

**Hỏi: Tại sao ORB dùng khoảng cách Hamming?**
Đáp: Vì descriptor của ORB là chuỗi bit 0/1. Khoảng cách Hamming là đếm số bit khác nhau, máy tính làm việc này cực nhanh.

**Hỏi: Lowe's ratio test để làm gì?**
Đáp: Loại các cặp khớp mơ hồ. Chỉ giữ cặp khi ứng viên giống nhất tốt hơn hẳn ứng viên thứ nhì. Đặc biệt quan trọng với hoa văn lặp lại.

**Hỏi: Ngưỡng RANSAC ảnh hưởng thế nào?**
Đáp: Ngưỡng nhỏ thì khắt khe: ít inlier nhưng chính xác hơn. Ngưỡng lớn thì dễ dãi: nhiều inlier nhưng sai số tăng.

**Hỏi: Tại sao chọn ảnh 2 làm mốc?**
Đáp: Chọn ảnh ở giữa giúp méo phối cảnh được chia đều sang 2 bên. Nếu chọn ảnh ở rìa thì ảnh ở đầu kia sẽ bị kéo giãn méo rất nhiều.

**Hỏi: Feather blending hoạt động thế nào?**
Đáp: Ở vùng chồng lấn, mỗi pixel được trộn từ các ảnh, ảnh nào có pixel nằm xa mép ảnh hơn thì được trọng số lớn hơn. Khoảng cách đến mép được tính bằng distance transform. Kết quả là chuyển tiếp mượt, không thấy đường nối.

**Hỏi: Reprojection error là gì?**
Đáp: Dùng homography đưa một điểm từ ảnh này sang ảnh kia, rồi đo độ lệch so với vị trí khớp thật. Càng nhỏ thì homography căn chỉnh càng chính xác.

### Câu hỏi chung

**Hỏi: Tại sao không dùng deep learning?**
Đáp: Đề bài yêu cầu chỉ dùng phương pháp truyền thống. Ưu điểm của phương pháp truyền thống: dễ giải thích từng bước, chạy nhanh trên CPU thường, không cần dữ liệu huấn luyện.

**Hỏi: Làm sao đảm bảo kết quả tái lập được?**
Đáp: Cố định seed ngẫu nhiên khi tạo nhiễu (Part A dùng seed 42), dữ liệu đầu vào được lưu trong thư mục `data/`, và thời gian chạy ở Part B lấy trung vị của 5 lần chạy.

---

## 6. TÓM TẮT 1 TRANG ĐỂ HỌC THUỘC

**PART A – LỌC ẢNH**
- 2 loại nhiễu: **Gauss** (lệch nhẹ mọi pixel, sigma 25) và **muối tiêu** (8% pixel thành trắng/đen).
- 4 bộ lọc: **Mean** (trung bình đều), **Gaussian** (trung bình ưu tiên ở giữa), **Median** (lấy giá trị giữa), **Laplacian** (làm nét cạnh).
- 3 cửa sổ: 3×3, 5×5, 7×7. Chỉ số: **MSE thấp, PSNR cao, SSIM gần 1, thời gian thấp**.
- Kết luận: **Gauss dùng Gaussian 3×3 (24.77 dB)**, **muối tiêu dùng Median 3×3 (26.80 dB, SSIM 0.929)**. Cửa sổ to thì mờ quá mức. Laplacian làm sắc nhưng gây halo và khuếch đại nhiễu.

**PART B – STEREO 3D (TEDDY)**
- 2 ảnh lệch ngang, **disparity lớn là gần, nhỏ là xa**. Dữ liệu Middlebury Teddy, ground truth chia 4, đã rectify.
- **SIFT + ratio test 0.75**: 337 cặp. **RANSAC** ước lượng **F**: 315 inlier (93.5%), sai số epipolar 0.24 px, Sampson 0.17 px. Đường epipolar nằm ngang.
- **BM**: so khớp khối độc lập, nhanh (~4 ms), thủng lỗ vùng trơn, phình mép.
- **SGBM**: thêm luật mượt theo nhiều hướng, dày đặc (~81%), chậm hơn (~10 ms).
- Tốt nhất **SGBM b=3, d=64**: Bad>2px **3.43%**. Ô to thì tệ hơn. d=96 không cần thiết vì disparity tối đa ~53.
- Point cloud: đổi disparity ra độ sâu, giả định f = 450 px, B = 0.16 m, lưu PLY ~136 nghìn điểm. Ếch và cây gần, nhà và gấu giữa, tường xa.

**PART C – GHÉP PANORAMA**
- 4 ảnh cắt từ ảnh kiến trúc, chồng lấn khoảng một nửa, méo phối cảnh nhẹ.
- Pipeline: **tìm đặc trưng, ghép cặp (KNN + ratio test), homography + RANSAC (2 px, 4 px), bẻ ảnh về ảnh mốc 2, trộn feathering**.
- **SIFT** chính xác (0.21 px), **ORB** nhanh nhất (~860 ms) nhưng sai số ~0.8–0.97 px, **AKAZE** sai số nhỏ nhất (0.19 px).
- Ngưỡng RANSAC lớn thì nhiều inlier hơn nhưng sai số lớn hơn.
- Feathering trộn theo khoảng cách đến mép ảnh, xoá đường nối và bóng ma.

**3 CÂU CHỐT**
1. Không có phương pháp tốt nhất cho mọi trường hợp, phải chọn theo **loại nhiễu / loại cảnh / yêu cầu tốc độ**.
2. Luôn có đánh đổi giữa **mượt và chi tiết**, giữa **nhanh và chính xác**.
3. **RANSAC + ratio test** là "bộ lọc cặp ghép sai" dùng chung cho cả stereo (Part B) và ghép ảnh (Part C).
