# Giới thiệu

Bài tập nhận dạng từ số đơn khóa xử lý tiếng nói UET. Để thực hiện ta có thể sử dụng 2 thuật toán DTW

và HMM.

# DTW

Dynamic time warping là thuật toán với ý tưởng cơ bản là dóng hàng các đặc trưng mfcc của từng template

với đặc trưng mfcc tương ứng của từ đơn sử dụng để test. Template tương nào có hàm lỗi nhỏ nhất sẽ được

chọn làm nhãn dự đoán cho test case đầu vào.

DTW có thể nhận dạng tốt trong trường hợp template và test là của cùng một người nói. Trong trường hợp

chúng khác nhau thì DTW cho kết quả kém hơn, cụ thể trong thí nghiệm ở đây thì DTW cho độ chính xác 80%

trong trường hợp giống nhau và 61% trong trường hợp còn lại.

## Nhận dạng với DTW

Template được lấy ra bởi một người và test trên cùng người đó

`python3 dtw_recognition.py --train_config_file <path_to_train_config_file>`

Template được lấy ra bởi một người và test trên nhiều người

`p1=--train_config_file  <path_to_train_config_file>`

`p2=--test_config_file  <path_to_test_config_file>`

`python3 dtw_recognition.py p1 p2`

# GMM-HMM

Sử dụng mô hình âm thanh trộn Gauss mỗi phân phối tương ứng với 1 người do có thể bao được nhiều khả năng

hơn so với DTW. Bằng các khởi tạo ma trận truyển trạng thái transmat, xác suất chuyển trạng thái của trạng

thái 1 start_prob ta có thể sử dụng thuật toán EM để huấn luyện GMMHMM.

## Sinh dữ liệu huấn luyện và test cho GMMHMM

`p1=(hmm_train_data_config_file <path_to_config_data_training_hmm>)`

`p2=(hmm_test_data_config_file <path_to_config_data_testing_hmm>)`

`python3 dataset.py p1 p2`

## Nhận dạng với GMMHMM

Gồm quá trình huấn luyện và test với dữ liệu được sinh ở phần trước

`p1=(train_config_file <Path to hmm train config>)`

`p2=(test_config_file <Path to hmm test config>)`

`python3 hmm.py p1  p2`

# Video demo

[![Watch the video](https://i.imgur.com/vKb2F1B.png)](https://drive.google.com/file/d/1Y6r5yV7rstuG138w0e07o5yydkm-KrHN/view?usp=sharing)

