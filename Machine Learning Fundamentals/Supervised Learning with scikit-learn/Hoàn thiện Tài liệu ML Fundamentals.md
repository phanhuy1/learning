## **Phần 1: Giới thiệu Tổng quan về Học Máy**

Học máy (Machine Learning \- ML) là một lĩnh vực của trí tuệ nhân tạo (AI) tập trung vào việc phát triển các thuật toán và mô hình thống kê cho phép hệ thống máy tính "học" từ dữ liệu mà không cần được lập trình một cách tường minh. Thay vì tuân theo các quy tắc do con người định sẵn, các hệ thống ML xây dựng một mô hình toán học dựa trên dữ liệu mẫu, được gọi là "dữ liệu huấn luyện", để đưa ra dự đoán hoặc quyết định.

### **1.1. Định nghĩa và Phân loại chi tiết**

Cốt lõi của học máy là khả năng tự động phát hiện các mẫu, quy luật và cấu trúc trong dữ liệu. Dựa trên bản chất của dữ liệu và mục tiêu của bài toán, học máy được phân thành ba loại chính. Việc lựa chọn loại hình học máy không chỉ là một quyết định kỹ thuật mà còn là một quyết định chiến lược, ảnh hưởng trực tiếp đến chi phí, nguồn lực và cách thức triển khai. Một mô hình học có giám sát yêu cầu dữ liệu có nhãn, dẫn đến chi phí đáng kể trong việc thu thập và gán nhãn, đặc biệt trong các lĩnh vực chuyên môn như y tế. Ngược lại, học không giám sát có thể tận dụng các kho dữ liệu khổng lồ chưa được khai thác, nhưng kết quả của nó thường cần sự diễn giải của con người để tạo ra giá trị kinh doanh. Học tăng cường đòi hỏi một môi trường mô phỏng an toàn để "thử và sai", làm cho nó rất mạnh mẽ trong các lĩnh vực như game nhưng rủi ro khi áp dụng trực tiếp vào các hệ thống thực tế có chi phí sai lầm cao.

#### **Học có giám sát (Supervised Learning)**

Học có giám sát là phương pháp phổ biến nhất trong học máy. Bản chất của nó là học từ một tập dữ liệu đã được gán nhãn, trong đó mỗi điểm dữ liệu đầu vào (input) đều đi kèm với một đầu ra (output) hoặc "nhãn" (label) chính xác. Nhãn này được coi là "sự thật cơ bản" (ground truth) mà mô hình cố gắng học để tạo ra một hàm ánh xạ từ đầu vào đến đầu ra. Mục tiêu cốt lõi không chỉ là dự đoán chính xác trên dữ liệu đã thấy, mà là khả năng **tổng quát hóa (generalization)** trên dữ liệu mới, chưa từng thấy. Đây là điểm mấu chốt để phân biệt một mô hình hữu ích và một mô hình chỉ đơn thuần "học vẹt" dữ liệu huấn luyện.

Các bài toán học có giám sát được chia thành hai loại chính:

* **Hồi quy (Regression):** Mục tiêu là dự đoán một giá trị số liên tục. Ví dụ bao gồm dự đoán giá nhà dựa trên diện tích và vị trí, dự báo nhiệt độ ngày mai, hoặc ước tính doanh thu của một công ty.  
* **Phân loại (Classification):** Mục tiêu là dự đoán một nhãn rời rạc hoặc một danh mục. Ví dụ bao gồm phân loại email là "spam" hay "không spam", chẩn đoán một khối u là "lành tính" hay "ác tính", hoặc nhận dạng chữ viết tay.

#### **Học không giám sát (Unsupervised Learning)**

Trái ngược với học có giám sát, học không giám sát làm việc với dữ liệu không có nhãn. Thay vì dự đoán một đầu ra cụ thể, mục tiêu của nó là khám phá các mẫu và cấu trúc tiềm ẩn trong chính dữ liệu. Vai trò chính của học không giám sát không phải là "dự đoán", mà là **"mô tả"** và **"nén"** thông tin, giúp con người hiểu rõ hơn về dữ liệu của mình.

Các bài toán học không giám sát điển hình bao gồm:

* **Phân cụm (Clustering):** Nhóm các điểm dữ liệu tương tự nhau vào các cụm. Các điểm trong cùng một cụm có đặc tính giống nhau hơn so với các điểm ở cụm khác. Ứng dụng phổ biến là phân khúc khách hàng để cá nhân hóa chiến dịch marketing, hoặc phân nhóm các gen có biểu hiện tương tự trong sinh học.  
* **Giảm chiều dữ liệu (Dimensionality Reduction):** Giảm số lượng biến (đặc trưng) của tập dữ liệu trong khi vẫn cố gắng giữ lại phần lớn thông tin quan trọng. Kỹ thuật này hữu ích trong việc trực quan hóa dữ liệu nhiều chiều và tăng hiệu quả tính toán cho các thuật toán khác.  
* **Khai phá luật kết hợp (Association Rule Mining):** Tìm ra các mối quan hệ hoặc quy tắc thú vị trong các tập dữ liệu lớn. Ví dụ kinh điển là phân tích giỏ hàng trong siêu thị để tìm ra quy tắc "những khách hàng mua bia cũng thường mua tã".

#### **Học tăng cường (Reinforcement Learning \- RL)**

Học tăng cường là một lĩnh vực học máy lấy cảm hứng từ tâm lý học hành vi. Nó mô tả cách một **tác nhân (agent)** thông minh nên thực hiện các **hành động (actions)** trong một **môi trường (environment)** để tối đa hóa phần thưởng tích lũy.1 Thay vì được cung cấp các cặp đầu vào-đầu ra đúng, agent học thông qua một quá trình thử và sai. Sau mỗi hành động, agent nhận được một phản hồi từ môi trường dưới dạng một

**trạng thái mới (new state)** và một **phần thưởng (reward)** hoặc **hình phạt (penalty)**.2

Mục tiêu cốt lõi của RL không phải là tối đa hóa phần thưởng tức thời, mà là **phần thưởng tích lũy dài hạn (cumulative reward)**. Điều này đòi hỏi agent phải cân bằng một cách thông minh giữa **khai thác (exploitation)** các hành động đã biết là mang lại phần thưởng tốt và **khám phá (exploration)** các hành động mới có thể dẫn đến phần thưởng tốt hơn trong tương lai. Các ứng dụng nổi bật của RL bao gồm trí tuệ nhân tạo chơi game (ví dụ: AlphaGo), điều khiển robot, và các hệ thống đề xuất tự tối ưu hóa.

### **1.2. Phân biệt AI, ML, và Deep Learning**

Các thuật ngữ Trí tuệ nhân tạo (AI), Học máy (ML), và Học sâu (Deep Learning) thường được sử dụng thay thế cho nhau, nhưng chúng đại diện cho các khái niệm có phạm vi khác nhau. Mối quan hệ giữa chúng có thể được hình dung như các vòng tròn đồng tâm:

* **Trí tuệ nhân tạo (AI):** Là lĩnh vực khoa học máy tính rộng lớn nhất, bao trùm bất kỳ kỹ thuật nào cho phép máy tính bắt chước trí thông minh của con người. Nó bao gồm mọi thứ từ các hệ thống dựa trên quy tắc logic (expert systems) đến các phương pháp thống kê phức tạp.  
* **Học máy (ML):** Là một tập hợp con của AI. ML tập trung vào các phương pháp cho phép máy móc học hỏi từ dữ liệu để cải thiện hiệu suất mà không cần lập trình rõ ràng.  
* **Học sâu (Deep Learning):** Là một tập hợp con chuyên biệt của ML. Học sâu sử dụng một kiến trúc cụ thể gọi là mạng nơ-ron nhân tạo nhiều lớp (sâu). Chính độ sâu của các mạng này cho phép chúng tự động học các biểu diễn (representations) và đặc trưng phức tạp từ dữ liệu thô, chẳng hạn như hình ảnh hoặc văn bản.

### **1.3. Quy trình làm việc của một dự án ML điển hình**

Một dự án học máy thành công không chỉ dừng lại ở việc lựa chọn và huấn luyện một mô hình. Nó là một quy trình lặp đi lặp lại bao gồm nhiều giai đoạn, từ việc hiểu vấn đề kinh doanh đến triển khai và bảo trì mô hình.

1. **Hiểu bài toán kinh doanh và định hình vấn đề (Business Understanding & Problem Framing):** Giai đoạn quan trọng nhất là chuyển đổi một mục tiêu kinh doanh thành một bài toán học máy có thể giải quyết được. Ví dụ, mục tiêu "giảm tỷ lệ khách hàng rời bỏ" được chuyển thành "xây dựng một mô hình phân loại nhị phân để dự đoán khách hàng nào có khả năng rời bỏ trong tháng tới".  
2. **Thu thập và chuẩn bị dữ liệu (Data Collection & Preparation):** Giai đoạn này thường tốn nhiều thời gian và công sức nhất, có thể chiếm tới 70-80% tổng thời gian dự án. Nó bao gồm các công việc như thu thập dữ liệu từ nhiều nguồn, làm sạch dữ liệu (xử lý giá trị thiếu, loại bỏ nhiễu), tiền xử lý (chuẩn hóa, mã hóa), và kỹ thuật đặc trưng (tạo ra các biến mới có ý nghĩa hơn).  
3. **Huấn luyện mô hình (Model Training):** Lựa chọn một hoặc nhiều thuật toán phù hợp với bài toán và dữ liệu. Mô hình được huấn luyện trên một phần dữ liệu được gọi là "tập huấn luyện" (training set).  
4. **Đánh giá mô hình (Model Evaluation):** Hiệu suất của mô hình được đánh giá trên một phần dữ liệu riêng biệt mà nó chưa từng thấy, gọi là "tập kiểm tra" (test set) hoặc "tập xác thực" (validation set). Việc này đảm bảo rằng mô hình có khả năng tổng quát hóa tốt.  
5. **Tinh chỉnh siêu tham số (Hyperparameter Tuning):** Hầu hết các mô hình đều có các "siêu tham số" (hyperparameters) không được học từ dữ liệu mà phải được thiết lập trước (ví dụ: số cụm K trong K-Means). Giai đoạn này tìm ra bộ siêu tham số tối ưu để mô hình đạt hiệu suất cao nhất.  
6. **Triển khai và giám sát (Deployment & Monitoring):** Sau khi một mô hình thỏa mãn các yêu cầu về hiệu suất, nó được triển khai vào môi trường sản phẩm (production) để đưa ra dự đoán trên dữ liệu thực tế. Công việc không dừng lại ở đây; mô hình cần được giám sát liên tục để phát hiện sự suy giảm hiệu suất do thay đổi trong phân phối dữ liệu (data drift) và lên kế hoạch tái huấn luyện định kỳ.

## **Phần 2: Nền tảng Toán học cho Học Máy**

Toán học là ngôn ngữ và nền tảng của học máy. Việc nắm vững các khái niệm toán học cốt lõi không chỉ giúp hiểu sâu hơn về cách các thuật toán hoạt động mà còn cho phép tùy chỉnh, gỡ lỗi và phát triển các phương pháp mới. Các nhánh toán học không tồn tại độc lập mà phối hợp chặt chẽ với nhau. Đại số tuyến tính cung cấp cấu trúc dữ liệu và các phép biến đổi. Xác suất và thống kê cung cấp hàm mục tiêu để đánh giá mô hình. Giải tích cung cấp công cụ (gradient) để tối ưu hóa hàm mục tiêu đó, và các thuật toán tối ưu hóa là quy trình lặp lại việc sử dụng gradient để cập nhật các tham số của mô hình.

### **2.1. Đại số tuyến tính: Ngôn ngữ của dữ liệu**

Trong học máy, dữ liệu—dù là hình ảnh, văn bản, hay bảng tính—đều được chuyển đổi thành các đối tượng toán học là vector và ma trận. Đại số tuyến tính cung cấp bộ công cụ để thao tác và biến đổi các đối tượng này một cách hiệu quả.

* **Vector và Ma trận:** Một điểm dữ liệu (ví dụ: một khách hàng) có thể được biểu diễn dưới dạng một vector, trong đó mỗi phần tử là một đặc trưng (tuổi, thu nhập). Một tập hợp các điểm dữ liệu tạo thành một ma trận.  
* **Tích vô hướng (Dot Product):** Là một phép toán cơ bản, được sử dụng rộng rãi trong việc tính toán đầu ra của các nơ-ron trong mạng nơ-ron và đo lường sự tương đồng giữa các vector.  
* **Giá trị riêng (Eigenvalues) và Vector riêng (Eigenvectors):** Là những khái niệm trung tâm trong nhiều kỹ thuật giảm chiều dữ liệu.

#### **Case Study: Phân tích thành phần chính (PCA)**

PCA là một kỹ thuật giảm chiều dữ liệu phổ biến, minh họa rõ nét sức mạnh của đại số tuyến tính. Khi một tập dữ liệu có quá nhiều đặc trưng, nó có thể gặp phải "lời nguyền của chiều dữ liệu" (curse of dimensionality), làm cho mô hình khó học và dễ bị quá khớp (overfitting). PCA giải quyết vấn đề này bằng cách tìm ra các hướng mới trong không gian dữ liệu để biểu diễn thông tin một cách cô đọng hơn.

1. **Chuẩn hóa dữ liệu:** Đầu tiên, dữ liệu được chuẩn hóa để mỗi đặc trưng có giá trị trung bình bằng 0 và phương sai bằng 1\.  
2. **Tính Ma trận Hiệp phương sai (Covariance Matrix):** Ma trận hiệp phương sai được tính toán từ dữ liệu đã chuẩn hóa. Ma trận này mô tả mối quan hệ tuyến tính và sự biến thiên đồng thời giữa các cặp đặc trưng.4  
3. **Phân rã Eigen (Eigen-decomposition):** Ma trận hiệp phương sai được phân rã để tìm ra các cặp giá trị riêng và vector riêng của nó.4  
4. **Diễn giải:**  
   * **Eigenvectors (Vector riêng):** Mỗi vector riêng đại diện cho một **hướng (trục)** mới trong không gian dữ liệu. Các hướng này, được gọi là **Thành phần chính (Principal Components)**, là các tổ hợp tuyến tính của các đặc trưng ban đầu và chúng trực giao với nhau.  
   * **Eigenvalues (Giá trị riêng):** Mỗi giá trị riêng cho biết **lượng phương sai** của dữ liệu được giải thích bởi vector riêng tương ứng. Vector riêng có giá trị riêng lớn nhất chính là hướng mà dữ liệu có sự biến thiên lớn nhất.4  
5. **Giảm chiều:** Bằng cách sắp xếp các vector riêng theo thứ tự giảm dần của giá trị riêng, ta có thể chọn ra k vector riêng hàng đầu (những hướng quan trọng nhất) để tạo thành một không gian con mới. Sau đó, dữ liệu ban đầu được chiếu lên không gian con này, tạo ra một biểu diễn dữ liệu mới với k chiều, ít hơn số chiều ban đầu nhưng vẫn giữ lại phần lớn phương sai (thông tin) của dữ liệu.

### **2.2. Giải tích: Công cụ tối ưu hóa**

Quá trình "học" của một mô hình máy học về cơ bản là một bài toán tối ưu hóa: tìm bộ tham số (trọng số) sao cho một **hàm mất mát (loss function)**—đo lường sai số giữa dự đoán của mô hình và giá trị thực tế—đạt giá trị nhỏ nhất. Giải tích, đặc biệt là đạo hàm, cung cấp công cụ để giải quyết bài toán này.

* **Đạo hàm và Đạo hàm riêng (Derivatives and Partial Derivatives):** Đạo hàm đo lường tốc độ thay đổi của một hàm số. Trong không gian nhiều chiều, đạo hàm riêng cho biết hàm số thay đổi như thế nào khi chỉ một biến thay đổi.  
* **Gradient:** Là một vector chứa tất cả các đạo hàm riêng của hàm mất mát theo từng tham số của mô hình. Gradient có một tính chất quan trọng: nó luôn chỉ về **hướng tăng nhanh nhất** của hàm số.6  
* **Quy tắc chuỗi (Chain Rule):** Là nền tảng của thuật toán lan truyền ngược (backpropagation) trong mạng nơ-ron, cho phép tính toán gradient của hàm mất mát một cách hiệu quả qua nhiều lớp phức tạp.

#### **Giải thích sâu: Thuật toán Gradient Descent**

Gradient Descent là thuật toán tối ưu hóa lặp đi lặp lại được sử dụng rộng rãi nhất để huấn luyện các mô hình học máy.

1. **Khởi tạo:** Bắt đầu với một bộ tham số (trọng số) ngẫu nhiên.  
2. **Tính Gradient:** Tại vị trí hiện tại, tính toán gradient của hàm mất mát.  
3. Cập nhật tham số: Vì gradient chỉ về hướng tăng nhanh nhất, để tối thiểu hóa hàm mất mát, ta cần di chuyển theo hướng ngược lại của gradient. Tham số được cập nhật theo công thức:

   wnew​=wold​−η⋅∇f(wold​)

   trong đó w là vector tham số, η (eta) là tốc độ học (learning rate)—một siêu tham số kiểm soát độ lớn của mỗi bước cập nhật, và ∇f(wold​) là gradient.6  
4. **Lặp lại:** Lặp lại bước 2 và 3 cho đến khi hàm mất mát hội tụ (không còn giảm đáng kể).

**Hạn chế của Gradient Descent:**

* **Cực tiểu cục bộ (Local Minima):** Thuật toán có thể bị "kẹt" tại một điểm cực tiểu cục bộ, không đảm bảo tìm thấy điểm cực tiểu toàn cục của hàm mất mát.6  
* **Tốc độ học (Learning Rate):** Việc lựa chọn learning rate rất quan trọng. Nếu quá lớn, thuật toán có thể "nhảy qua" điểm tối ưu và không hội tụ. Nếu quá nhỏ, thuật toán sẽ hội tụ rất chậm, tốn nhiều thời gian và tài nguyên tính toán.6

### **2.3. Xác suất & Thống kê: Mô hình hóa sự không chắc chắn**

Hầu hết các dự đoán trong học máy đều mang tính xác suất. Xác suất và thống kê cung cấp một khuôn khổ toán học để định lượng và làm việc với sự không chắc chắn này.

* **Phân phối xác suất (Probability Distributions):** Các phân phối như Gaussian (chuẩn), Bernoulli, và Poisson được sử dụng để mô hình hóa dữ liệu và các giả định trong mô hình.  
* **Kỳ vọng (Expectation) và Phương sai (Variance):** Là các thước đo thống kê cơ bản để mô tả và tóm tắt dữ liệu.  
* **Định lý Bayes:** Cung cấp một cách để cập nhật niềm tin (xác suất) về một giả thuyết khi có bằng chứng mới.

#### **Case Study: Phân loại Naïve Bayes**

Thuật toán Naïve Bayes là một ví dụ điển hình về ứng dụng của lý thuyết xác suất trong phân loại, đặc biệt hiệu quả cho các bài toán xử lý văn bản như lọc email spam.

1. Nền tảng \- Định lý Bayes: Định lý Bayes phát biểu rằng:

   P(A∣B)=P(B)P(B∣A)⋅P(A)​

   Trong bối cảnh phân loại, công thức này được diễn giải là:

   P(Lớp∣Đặc trưng)=P(Đặc trưng)P(Đặc trưng∣Lớp)⋅P(Lớp)​

   Mục tiêu là tìm lớp có xác suất hậu nghiệm P(Lớp∣Đặc trưng) lớn nhất.8  
2. **Giả định "Ngây thơ" (The "Naive" Assumption):** Để tính toán P(Đặc trưng∣Lớp), ta cần xem xét xác suất kết hợp của tất cả các đặc trưng. Điều này rất phức tạp. Naïve Bayes đơn giản hóa vấn đề bằng một giả định mạnh mẽ nhưng "ngây thơ": tất cả các đặc trưng là **độc lập có điều kiện** với nhau khi biết lớp.8 Ví dụ, trong lọc spam, nó giả định rằng sự xuất hiện của từ "miễn phí" không ảnh hưởng đến sự xuất hiện của từ "khuyến mãi", miễn là chúng ta biết email đó thuộc lớp "spam".  
3. Hệ quả: Giả định này cho phép chúng ta tính toán P(Đặc trưng∣Lớp) bằng cách nhân các xác suất riêng lẻ:

   P(feature1​,feature2​,…∣Lớp)=P(feature1​∣Lớp)⋅P(feature2​∣Lớp)⋅…

   Mặc dù giả định độc lập hiếm khi đúng trong thực tế, Naïve Bayes vẫn hoạt động hiệu quả một cách đáng ngạc nhiên, đặc biệt là với dữ liệu văn bản có số chiều cao, nơi nó cung cấp một phương pháp phân loại nhanh và mạnh mẽ.

### **2.4. Tối ưu hóa**

Tối ưu hóa là lĩnh vực nghiên cứu các thuật toán để tìm ra các giá trị cực trị (cực đại hoặc cực tiểu) của các hàm số. Trong học máy, nó là trung tâm của quá trình huấn luyện mô hình.

* **Các biến thể của Gradient Descent:**  
  * **Batch Gradient Descent (BGD):** Tính toán gradient trên toàn bộ tập dữ liệu huấn luyện trước khi thực hiện một bước cập nhật. Nó đảm bảo hội tụ đến điểm tối ưu (đối với hàm lồi) nhưng rất chậm và tốn bộ nhớ với dữ liệu lớn.  
  * **Stochastic Gradient Descent (SGD):** Cập nhật tham số sau mỗi điểm dữ liệu. Điều này làm cho quá trình cập nhật nhanh hơn nhiều và có thể giúp thoát khỏi các điểm cực tiểu cục bộ nông, nhưng các bước cập nhật rất "nhiễu" và không ổn định.7  
  * **Mini-batch Gradient Descent:** Là một sự thỏa hiệp, cập nhật tham số sau mỗi "lô" (batch) nhỏ dữ liệu. Đây là phương pháp được sử dụng phổ biến nhất trong thực tế, cân bằng giữa tốc độ của SGD và sự ổn định của BGD.  
* **Các Optimizers nâng cao:**  
  * **Momentum:** Bổ sung một "đà" (momentum) vào quá trình cập nhật, giúp thuật toán tăng tốc theo các hướng có gradient nhất quán và giảm dao động, từ đó hội tụ nhanh hơn và có khả năng vượt qua các điểm cực tiểu cục bộ.  
  * **Adam (Adaptive Moment Estimation):** Là một trong những optimizer phổ biến và hiệu quả nhất hiện nay. Adam kết hợp ý tưởng của momentum với việc điều chỉnh tốc độ học một cách thích ứng cho từng tham số riêng biệt. Nó tính toán và lưu trữ cả moment bậc nhất (trung bình của gradient) và moment bậc hai (trung bình của bình phương gradient) để điều chỉnh tốc độ học, giúp nó hoạt động tốt trên nhiều loại bài toán khác nhau mà không cần tinh chỉnh learning rate quá nhiều.10

## **Phần 3: Xử lý và Chuẩn bị Dữ liệu**

Chất lượng của một mô hình học máy phụ thuộc rất lớn vào chất lượng của dữ liệu đầu vào. Giai đoạn chuẩn bị dữ liệu, bao gồm tiền xử lý và kỹ thuật đặc trưng, thường là giai đoạn tốn nhiều thời gian nhất nhưng lại có tác động sâu sắc nhất đến kết quả cuối cùng. Các quyết định trong giai đoạn này có thể ảnh hưởng đến hiệu suất mô hình nhiều hơn cả việc lựa chọn một thuật toán phức tạp. Ví dụ, việc áp dụng One-Hot Encoding cho một đặc trưng có hàng nghìn danh mục sẽ tạo ra hàng nghìn cột mới, dẫn đến "lời nguyền chiều dữ liệu". Điều này làm cho không gian đặc trưng trở nên thưa thớt, khiến các thuật toán dựa trên khoảng cách như KNN kém hiệu quả, đồng thời làm tăng nguy cơ quá khớp và thời gian huấn luyện cho các mô hình phức tạp.

### **3.1. Tiền xử lý dữ liệu (Data Preprocessing)**

Tiền xử lý là quá trình chuyển đổi dữ liệu thô thành một định dạng sạch và phù hợp để đưa vào mô hình học máy.

#### **Scaling & Normalization**

Nhiều thuật toán học máy, đặc biệt là những thuật toán dựa trên khoảng cách (như K-Nearest Neighbors, SVM) hoặc tối ưu hóa bằng gradient descent (như mạng nơ-ron), hoạt động tốt hơn khi các đặc trưng đầu vào có cùng một thang đo.

* **Standardization (Chuẩn hóa Z-score):** Kỹ thuật này biến đổi dữ liệu sao cho nó có giá trị trung bình là 0 và độ lệch chuẩn là 1\. Công thức là z=(x−μ)/σ. Standardization đặc biệt hữu ích khi dữ liệu của bạn có phân phối gần giống phân phối Gaussian (hình chuông) và được ưu tiên cho các thuật toán như Hồi quy tuyến tính và Hồi quy logistic.12 Nó cũng ít bị ảnh hưởng bởi các giá trị ngoại lai (outliers) hơn so với Normalization.  
* **Normalization (Chuẩn hóa Min-Max):** Kỹ thuật này co giãn dữ liệu vào một phạm vi cố định, thường là từ 0 đến 1\. Công thức là Xnorm​=(X−Xmin​)/(Xmax​−Xmin​). Normalization hữu ích khi phân phối của dữ liệu không rõ hoặc không phải là Gaussian, và cho các thuật toán không đưa ra giả định về phân phối dữ liệu, chẳng hạn như K-Nearest Neighbors (KNN) và mạng nơ-ron.12

#### **Encoding Categorical Data**

Các mô hình học máy yêu cầu đầu vào là số, do đó các đặc trưng dạng danh mục (categorical) cần được chuyển đổi.

* **Label Encoding:** Gán một số nguyên duy nhất cho mỗi danh mục (ví dụ: "Thấp" \-\> 0, "Trung bình" \-\> 1, "Cao" \-\> 2). Phương pháp này chỉ nên được sử dụng cho dữ liệu có thứ tự (ordinal), vì nó có thể tạo ra một mối quan hệ thứ tự giả tạo mà mô hình có thể học sai nếu áp dụng cho dữ liệu không có thứ tự (nominal).14  
* **One-Hot Encoding:** Tạo ra các cột nhị phân (0/1) mới cho mỗi giá trị danh mục. Ví dụ, đặc trưng "Màu sắc" với các giá trị {"Đỏ", "Xanh"} sẽ được chuyển thành hai cột "Màu\_Đỏ" và "Màu\_Xanh". Phương pháp này phù hợp cho dữ liệu nominal nhưng có thể làm tăng đáng kể số chiều của dữ liệu nếu đặc trưng có nhiều danh mục khác nhau.14

#### **Handling Missing Values**

Dữ liệu trong thực tế thường không đầy đủ. Có nhiều chiến lược để xử lý các giá trị bị thiếu:

* **Xóa bỏ:** Xóa các hàng hoặc cột có giá trị thiếu. Đây là cách đơn giản nhất nhưng có thể làm mất thông tin quan trọng, chỉ nên áp dụng khi lượng dữ liệu thiếu là rất nhỏ.  
* **Điền giá trị (Imputation):** Thay thế các giá trị thiếu bằng một giá trị thống kê như trung bình (mean), trung vị (median), hoặc yếu vị (mode) của cột đó. Trung vị thường là lựa chọn tốt hơn trung bình khi dữ liệu có outliers.  
* **Điền giá trị dự đoán (Predictive Imputation):** Sử dụng một mô hình học máy khác để dự đoán các giá trị bị thiếu dựa trên các đặc trưng khác.

#### **Handling Outliers**

Outliers là các điểm dữ liệu khác biệt đáng kể so với phần còn lại. Chúng có thể là lỗi dữ liệu hoặc là các sự kiện hiếm nhưng có thật (ví dụ: giao dịch gian lận). Việc xử lý chúng phụ thuộc vào bối cảnh:

* **Phát hiện:** Sử dụng các phương pháp thống kê như khoảng tứ phân vị (IQR) hoặc Z-score để xác định outliers.  
* **Xử lý:** Có thể loại bỏ, thay thế (ví dụ: bằng giá trị giới hạn trên/dưới), hoặc giữ lại nếu chúng mang thông tin quan trọng.

### **3.2. Kỹ thuật đặc trưng (Feature Engineering)**

Đây là quá trình sử dụng kiến thức chuyên môn về lĩnh vực để tạo ra các đặc trưng mới từ dữ liệu thô, giúp mô hình học máy hoạt động hiệu quả hơn.

* **Feature Creation:** Tạo ra các đặc trưng mới có ý nghĩa hơn. Ví dụ, từ đặc trưng "ngày giao dịch", ta có thể tạo ra "ngày trong tuần", "có phải cuối tuần không". Từ "tổng giá" và "số lượng", ta có thể tạo ra "đơn giá".  
* **Feature Selection:** Lựa chọn một tập hợp con các đặc trưng quan trọng nhất từ bộ đặc trưng ban đầu. Điều này giúp giảm độ phức tạp của mô hình, giảm thời gian huấn luyện và tránh quá khớp. Các phương pháp bao gồm:  
  * **Filter methods:** Dựa trên các chỉ số thống kê (như tương quan, chi-squared) để xếp hạng và chọn đặc trưng.  
  * **Wrapper methods:** Sử dụng một mô hình để đánh giá các tập con đặc trưng khác nhau.  
  * **Embedded methods:** Quá trình lựa chọn đặc trưng được tích hợp vào chính quá trình huấn luyện mô hình (ví dụ: Lasso Regression, Random Forest feature importance).  
* **Dimensionality Reduction:** Các kỹ thuật như PCA được sử dụng để giảm số lượng đặc trưng trong khi vẫn giữ lại phần lớn thông tin.

### **3.3. Phân chia dữ liệu (Data Splitting)**

Để đánh giá hiệu suất thực sự của một mô hình, điều quan trọng là phải kiểm tra nó trên dữ liệu mà nó chưa từng thấy trong quá trình huấn luyện.

#### **Train/Validation/Test Split**

Tập dữ liệu thường được chia thành ba phần riêng biệt:

* **Tập huấn luyện (Train set):** Phần lớn nhất của dữ liệu, được sử dụng để "dạy" hoặc huấn luyện các tham số của mô hình.  
* **Tập xác thực (Validation set):** Được sử dụng để tinh chỉnh các siêu tham số của mô hình (ví dụ: chọn learning rate tốt nhất) và để lựa chọn giữa các mô hình khác nhau.  
* **Tập kiểm tra (Test set):** Được giữ hoàn toàn riêng biệt và chỉ được sử dụng một lần duy nhất ở cuối cùng để đánh giá hiệu suất cuối cùng của mô hình đã được lựa chọn và tinh chỉnh.

#### **Cross-Validation (Kiểm định chéo)**

Khi dữ liệu khan hiếm, việc chia thành ba tập có thể không hiệu quả. Kiểm định chéo là một kỹ thuật mạnh mẽ hơn để đánh giá mô hình.

* **K-Fold Cross-Validation:** Dữ liệu được chia thành k phần (folds) bằng nhau. Quá trình huấn luyện và đánh giá được lặp lại k lần. Trong mỗi lần lặp, một phần được sử dụng làm tập xác thực, và k-1 phần còn lại được sử dụng làm tập huấn luyện. Hiệu suất cuối cùng là trung bình của k lần đánh giá.  
* **Stratified K-Fold Cross-Validation:** Một biến thể của K-Fold, đặc biệt quan trọng cho các bài toán phân loại với dữ liệu mất cân bằng. Nó đảm bảo rằng tỷ lệ của mỗi lớp trong từng fold là tương tự như tỷ lệ trong toàn bộ tập dữ liệu, giúp việc đánh giá trở nên đáng tin cậy hơn.16

## **Phần 4: Học có giám sát (Supervised Learning)**

Học có giám sát là xương sống của nhiều ứng dụng học máy trong thực tế, từ dự đoán giá cổ phiếu đến chẩn đoán y khoa. Việc lựa chọn thuật toán phù hợp không chỉ dựa trên độ chính xác mà còn phụ thuộc vào các yếu tố như khả năng diễn giải, yêu cầu tính toán và đặc điểm của dữ liệu. Trong các ngành công nghiệp được quản lý chặt chẽ như tài chính, khả năng giải thích lý do một khoản vay bị từ chối có thể quan trọng hơn việc tăng một vài phần trăm độ chính xác. Điều này cho thấy bối cảnh kinh doanh và các quy định pháp lý đóng vai trò quyết định trong việc lựa chọn kỹ thuật, đôi khi ưu tiên các mô hình đơn giản hơn như Hồi quy Logistic so với các mô hình "hộp đen" phức tạp như XGBoost.

### **4.1. Hồi quy (Dự đoán giá trị liên tục)**

Các thuật toán hồi quy được sử dụng khi biến mục tiêu là một giá trị liên tục.

#### **Linear Regression (Hồi quy tuyến tính)**

* **Ý nghĩa:** Đây là thuật toán hồi quy cơ bản nhất, tìm cách mô hình hóa mối quan hệ giữa các biến độc lập và một biến phụ thuộc bằng cách tìm một đường thẳng (hoặc một siêu phẳng trong không gian nhiều chiều) phù hợp nhất với dữ liệu.  
* **Hàm mất mát:** Thường sử dụng Sai số bình phương trung bình (Mean Squared Error \- MSE) để đo lường sự khác biệt giữa các giá trị dự đoán và giá trị thực tế.  
* **Giả định:** Hồi quy tuyến tính hoạt động tốt nhất khi các giả định của nó được đáp ứng: mối quan hệ tuyến tính giữa các biến, các biến độc lập không có tương quan cao với nhau (không có đa cộng tuyến), và phương sai của sai số không đổi (homoscedasticity).17  
* **Ưu điểm:** Đơn giản, nhanh, và rất dễ diễn giải. Các hệ số của mô hình cho biết chính xác mức độ ảnh hưởng của mỗi đặc trưng đến kết quả dự đoán.17  
* **Nhược điểm:** Mô hình quá đơn giản để nắm bắt các mối quan hệ phi tuyến phức tạp trong thế giới thực và rất nhạy cảm với các giá trị ngoại lai (outliers).17

#### **Polynomial Regression (Hồi quy đa thức)**

* **Ý nghĩa:** Là một sự mở rộng của hồi quy tuyến tính, cho phép mô hình hóa các mối quan hệ phi tuyến bằng cách thêm các đặc trưng đa thức (ví dụ: x2,x3) vào mô hình. Về bản chất, nó vẫn là một mô hình tuyến tính đối với các hệ số.  
* **Ưu điểm:** Có thể phù hợp với các đường cong phức tạp hơn trong dữ liệu.  
* **Nhược điểm:** Rất dễ bị quá khớp (overfitting), đặc biệt với bậc đa thức cao.

#### **Regularized Regression (Hồi quy chính quy hóa)**

Các kỹ thuật này được sử dụng để chống lại hiện tượng quá khớp trong các mô hình hồi quy bằng cách thêm một "thành phần phạt" vào hàm mất mát để kiểm soát độ lớn của các hệ số.

* **Ridge Regression (L2 Regularization):** Thành phần phạt là tổng bình phương của các hệ số. Nó có xu hướng co các hệ số về gần 0 nhưng không bao giờ bằng 0, do đó nó giữ lại tất cả các đặc trưng trong mô hình.  
* **Lasso Regression (L1 Regularization):** Thành phần phạt là tổng giá trị tuyệt đối của các hệ số. Một đặc điểm quan trọng của Lasso là nó có thể ép một số hệ số về chính xác bằng 0, do đó có thể được sử dụng như một phương pháp lựa chọn đặc trưng (feature selection).  
* **ElasticNet:** Là sự kết hợp của cả hai kỹ thuật Ridge và Lasso, tận dụng ưu điểm của cả hai.

### **4.2. Phân loại (Dự đoán nhãn rời rạc)**

Các thuật toán phân loại được sử dụng khi biến mục tiêu là một danh mục hoặc một nhãn rời rạc.

#### **Logistic Regression (Hồi quy Logistic)**

* **Ý nghĩa:** Mặc dù có tên là "hồi quy", đây là một thuật toán phân loại. Nó sử dụng hàm sigmoid để ánh xạ đầu ra của một phương trình tuyến tính vào một khoảng xác suất (0, 1), ước tính xác suất một điểm dữ liệu thuộc về một lớp cụ thể.  
* **Ứng dụng:** Thường được sử dụng cho các bài toán phân loại nhị phân như phát hiện spam, dự đoán khách hàng rời bỏ.

#### **Decision Trees (Cây quyết định)**

* **Ý nghĩa:** Xây dựng một mô hình dự đoán dưới dạng một cấu trúc cây. Nó chia nhỏ tập dữ liệu thành các tập con nhỏ hơn dựa trên các ngưỡng của các đặc trưng, tạo ra một loạt các quy tắc "if-then-else".  
* **Ưu điểm:** Rất dễ hiểu và trực quan hóa, có thể xử lý cả dữ liệu số và dữ liệu danh mục.  
* **Nhược điểm:** Rất dễ bị quá khớp và không ổn định (một thay đổi nhỏ trong dữ liệu có thể dẫn đến một cây hoàn toàn khác).

| Thuật toán | Tiêu chí phân chia | Xử lý dữ liệu liên tục | Loại cây |
| :---- | :---- | :---- | :---- |
| **ID3** | Information Gain (dựa trên Entropy) | Không | Đa nhánh |
| **C4.5** | Gain Ratio (cải tiến của Information Gain) | Có (bằng cách tạo ngưỡng) | Đa nhánh |
| **CART** | Gini Impurity (phân loại), Variance Reduction (hồi quy) | Có (bằng cách tạo ngưỡng) | Nhị phân |

Bảng 4.1: So sánh các thuật toán Decision Tree phổ biến. Bảng này tóm tắt các khác biệt chính về tiêu chí phân chia và khả năng xử lý dữ liệu giữa ID3, C4.5 và CART, giúp người đọc lựa chọn thuật toán phù hợp dựa trên đặc điểm của bài toán.18

#### **Random Forest (Rừng ngẫu nhiên)**

* **Ý nghĩa:** Là một phương pháp ensemble thuộc loại bagging. Nó xây dựng nhiều cây quyết định trên các mẫu con ngẫu nhiên của dữ liệu và các tập con ngẫu nhiên của đặc trưng. Dự đoán cuối cùng được đưa ra bằng cách lấy trung bình (hồi quy) hoặc bỏ phiếu đa số (phân loại) từ tất cả các cây.  
* **Ưu điểm:** Mạnh mẽ, có độ chính xác cao và ít bị quá khớp hơn so với một cây quyết định đơn lẻ.  
* **Nhược điểm:** Mất đi tính dễ diễn giải của một cây quyết định đơn lẻ, trở thành một mô hình "hộp đen".

#### **Support Vector Machines (SVM)**

* **Ý nghĩa:** Tìm một siêu phẳng (hyperplane) trong không gian nhiều chiều để phân tách các lớp dữ liệu với một "lề" (margin) rộng nhất có thể. Các điểm dữ liệu nằm trên lề được gọi là các vector hỗ trợ (support vectors).  
* **Kernel Trick:** Đối với dữ liệu không thể phân tách tuyến tính, SVM sử dụng "kernel trick". Thay vì ánh xạ dữ liệu lên một không gian có số chiều cao hơn một cách tường minh (tốn kém tính toán), kernel trick cho phép SVM tính toán các mối quan hệ trong không gian chiều cao đó một cách hiệu quả bằng cách sử dụng các hàm kernel. Điều này cho phép SVM tìm ra các đường biên phân loại phi tuyến phức tạp.20  
* **Ưu điểm:** Rất hiệu quả trong không gian nhiều chiều và khi dữ liệu không phân tách tuyến tính.  
* **Nhược điểm:** Tốn tài nguyên tính toán và bộ nhớ, khó mở rộng cho các tập dữ liệu rất lớn.

#### **K-Nearest Neighbors (KNN)**

* **Ý nghĩa:** Là một thuật toán "lười học" (lazy learning). Để phân loại một điểm dữ liệu mới, nó tìm k điểm dữ liệu gần nhất trong tập huấn luyện và gán nhãn dựa trên bỏ phiếu đa số của k hàng xóm đó.  
* **Ưu điểm:** Cực kỳ đơn giản và trực quan.  
* **Nhược điểm:** Chậm trong giai đoạn dự đoán vì phải tính toán khoảng cách đến tất cả các điểm trong tập huấn luyện. Nhạy cảm với các đặc trưng không liên quan và thang đo của dữ liệu.

#### **Naïve Bayes**

* **Ý nghĩa:** Dựa trên định lý Bayes với giả định "ngây thơ" về sự độc lập của các đặc trưng.  
* **Ưu điểm:** Rất nhanh, yêu cầu ít dữ liệu huấn luyện, và hoạt động đặc biệt tốt cho các bài toán phân loại văn bản.  
* **Nhược điểm:** Giả định độc lập hiếm khi đúng trong thực tế, điều này có thể ảnh hưởng đến hiệu suất trên một số loại dữ liệu.

### **4.3. Phương pháp Ensemble**

Phương pháp Ensemble kết hợp nhiều mô hình học máy (gọi là "base learners") để tạo ra một mô hình dự đoán mạnh mẽ hơn.

* **Bagging (Bootstrap Aggregating):** Huấn luyện nhiều mô hình độc lập trên các mẫu con được lấy ngẫu nhiên (có lặp lại) từ tập dữ liệu huấn luyện. Kết quả cuối cùng được tổng hợp bằng cách bỏ phiếu hoặc lấy trung bình. Random Forest là ví dụ tiêu biểu.  
* **Boosting:** Huấn luyện các mô hình một cách tuần tự. Mỗi mô hình sau tập trung vào việc sửa chữa những sai lầm của các mô hình trước đó bằng cách chú trọng hơn vào các điểm dữ liệu bị phân loại sai. Các thuật toán boosting nổi tiếng bao gồm AdaBoost, Gradient Boosting, và các triển khai hiệu suất cao như **XGBoost**, LightGBM, và CatBoost. XGBoost đặc biệt phổ biến nhờ các ưu điểm vượt trội như xử lý song song, chính quy hóa tích hợp, và khả năng xử lý giá trị thiếu một cách thông minh.21  
* **Stacking:** Kết hợp các dự đoán từ nhiều mô hình khác nhau (base models) bằng cách sử dụng một mô hình khác (meta-model) để học cách đưa ra dự đoán cuối cùng.

### **4.4. Các độ đo đánh giá (Evaluation Metrics)**

Lựa chọn độ đo phù hợp là rất quan trọng để đánh giá đúng hiệu suất của mô hình.

#### **Độ đo cho Hồi quy**

* **MSE (Mean Squared Error):** Sai số bình phương trung bình. Rất nhạy cảm với các giá trị ngoại lai lớn.  
* **MAE (Mean Absolute Error):** Sai số tuyệt đối trung bình. Ít nhạy cảm với outliers hơn và dễ diễn giải hơn.  
* **R² Score (Hệ số xác định):** Đo lường tỷ lệ phần trăm phương sai của biến phụ thuộc được giải thích bởi mô hình. Giá trị càng gần 1 càng tốt.

#### **Độ đo cho Phân loại**

* **Accuracy (Độ chính xác):** Tỷ lệ dự đoán đúng trên tổng số dự đoán. Dễ hiểu nhưng có thể gây hiểu lầm nghiêm trọng trên các tập dữ liệu mất cân bằng.  
* **Precision, Recall, F1-Score:**  
  * **Precision (Độ chính xác):** Trong số những gì được dự đoán là tích cực, có bao nhiêu là đúng? Quan trọng khi chi phí của một dự đoán sai tích cực (False Positive) là cao (ví dụ: đánh dấu nhầm email quan trọng là spam).  
  * **Recall (Độ phủ):** Trong số tất cả các trường hợp thực sự tích cực, mô hình đã tìm thấy bao nhiêu? Quan trọng khi chi phí của việc bỏ lỡ một trường hợp tích cực (False Negative) là cao (ví dụ: bỏ sót một bệnh nhân ung thư).  
  * **F1-Score:** Trung bình điều hòa của Precision và Recall, hữu ích khi cần cân bằng cả hai.  
* **ROC-AUC:** Đường cong ROC (Receiver Operating Characteristic) biểu diễn sự đánh đổi giữa tỷ lệ dương tính thật (Recall) và tỷ lệ dương tính giả. AUC (Area Under the Curve) là diện tích dưới đường cong này, đo lường khả năng tổng thể của mô hình trong việc phân biệt giữa các lớp.  
* **Confusion Matrix (Ma trận nhầm lẫn):** Một bảng phân tích chi tiết các kết quả dự đoán, bao gồm True Positives (TP), True Negatives (TN), False Positives (FP), và False Negatives (FN).

## **Phần 5: Học không giám sát (Unsupervised Learning)**

Học không giám sát mở ra khả năng khám phá các cấu trúc và mẫu tiềm ẩn trong dữ liệu mà không cần đến nhãn được gán trước. Không giống như học có giám sát, nơi có một "câu trả lời đúng" rõ ràng, kết quả của các thuật toán không giám sát thường phụ thuộc nhiều vào các giả định của thuật toán và sự lựa chọn tham số. Ví dụ, K-Means sẽ luôn trả về K cụm hình cầu, ngay cả khi dữ liệu có cấu trúc khác, trong khi DBSCAN phụ thuộc vào việc lựa chọn cẩn thận các tham số eps và min\_samples. Điều này nhấn mạnh vai trò không thể thiếu của chuyên gia lĩnh vực trong việc diễn giải và xác thực tính hữu ích của các kết quả, biến quá trình phân tích thành một vòng lặp giữa thuật toán và kiến thức chuyên môn.

### **5.1. Phân cụm (Clustering)**

Phân cụm là nhiệm vụ nhóm một tập hợp các đối tượng sao cho các đối tượng trong cùng một nhóm (cụm) có nhiều điểm tương đồng hơn so với các đối tượng ở các nhóm khác.

#### **K-Means**

* **Cách hoạt động:** Là một thuật toán phân cụm dựa trên centroid, hoạt động theo các bước lặp đi lặp lại:  
  1. **Chọn K:** Người dùng phải xác định trước số lượng cụm K mong muốn.23  
  2. **Khởi tạo Centroids:** Chọn ngẫu nhiên K điểm dữ liệu làm tâm cụm (centroids) ban đầu.23  
  3. **Gán điểm:** Gán mỗi điểm dữ liệu vào cụm có centroid gần nhất, thường dựa trên khoảng cách Euclid.23  
  4. **Cập nhật Centroids:** Tính toán lại vị trí của mỗi centroid bằng cách lấy giá trị trung bình của tất cả các điểm dữ liệu được gán cho cụm đó.23  
  5. **Lặp lại:** Lặp lại các bước 3 và 4 cho đến khi vị trí của các centroids không còn thay đổi đáng kể, tức là thuật toán đã hội tụ.  
* **Ưu điểm:** Nhanh, đơn giản và có khả năng mở rộng tốt cho các tập dữ liệu lớn.  
* **Nhược điểm:** Cần phải chỉ định trước số cụm K, nhạy cảm với việc khởi tạo centroids ban đầu và các giá trị ngoại lai, và có xu hướng tạo ra các cụm có dạng hình cầu và kích thước tương đương.

#### **Hierarchical Clustering (Phân cụm phân cấp)**

* **Cách hoạt động:** Xây dựng một hệ thống phân cấp các cụm dưới dạng một cây (dendrogram). Có hai cách tiếp cận chính:  
  * **Agglomerative (Gom cụm):** Bắt đầu với mỗi điểm dữ liệu là một cụm riêng, sau đó liên tục hợp nhất các cặp cụm gần nhất cho đến khi chỉ còn một cụm duy nhất.  
  * **Divisive (Phân chia):** Bắt đầu với tất cả các điểm dữ liệu trong một cụm, sau đó liên tục chia nhỏ các cụm cho đến khi mỗi điểm là một cụm riêng.  
* **Ưu điểm:** Không cần chỉ định trước số lượng cụm; dendrogram cho phép trực quan hóa cấu trúc phân cấp và lựa chọn số cụm phù hợp.  
* **Nhược điểm:** Độ phức tạp tính toán cao (thường là O(n2logn) hoặc O(n3)), không phù hợp với các tập dữ liệu lớn.25

#### **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**

* **Cách hoạt động:** Là một thuật toán phân cụm dựa trên mật độ. Nó nhóm các điểm dữ liệu nằm gần nhau trong các khu vực có mật độ cao và đánh dấu các điểm nằm trong các khu vực có mật độ thấp là nhiễu (noise) hoặc outliers.  
* **Ưu điểm:** Có thể tìm thấy các cụm có hình dạng bất kỳ, không cần chỉ định trước số lượng cụm, và mạnh mẽ với các giá trị ngoại lai.25  
* **Nhược điểm:** Hiệu suất phụ thuộc nhiều vào việc lựa chọn hai tham số: eps (bán kính lân cận) và minPts (số điểm tối thiểu trong một lân cận). Khó hoạt động tốt với các cụm có mật độ khác nhau.

| Thuật toán | Nguyên tắc hoạt động | Yêu cầu số cụm (K) | Hình dạng cụm | Xử lý Outliers | Độ phức tạp tính toán |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **K-Means** | Dựa trên Centroid | Có, phải xác định trước | Hình cầu, kích thước tương đương | Nhạy cảm | Thấp (O(n⋅K⋅d⋅i)) |
| **Hierarchical** | Dựa trên phân cấp | Không, chọn sau từ dendrogram | Bất kỳ (phụ thuộc linkage) | Nhạy cảm | Cao (O(n2logn) đến O(n3)) |
| **DBSCAN** | Dựa trên Mật độ | Không, tự động xác định | Bất kỳ | Mạnh mẽ, xác định là nhiễu | Trung bình (O(nlogn) với index) |

Bảng 5.1: So sánh các thuật toán Phân cụm. Bảng này cung cấp một cái nhìn tổng quan về sự khác biệt cốt lõi giữa K-Means, Hierarchical Clustering và DBSCAN, giúp người dùng lựa chọn phương pháp phù hợp nhất với đặc điểm dữ liệu và mục tiêu phân tích của họ.25

### **5.2. Giảm chiều dữ liệu**

#### **PCA (Principal Component Analysis)**

Như đã thảo luận trong phần nền tảng toán học, PCA là một kỹ thuật giảm chiều tuyến tính. Sau khi thực hiện PCA, việc diễn giải các thành phần chính là rất quan trọng để hiểu được ý nghĩa của không gian đặc trưng mới. Điều này được thực hiện bằng cách kiểm tra **tương quan (correlations)** giữa các biến ban đầu và các thành phần chính mới. Một thành phần chính được "định nghĩa" bởi các biến ban đầu có tương quan mạnh nhất với nó. Ví dụ, nếu Thành phần chính 1 có tương quan dương cao với các biến "thu nhập", "chi tiêu", và "số lượng giao dịch", ta có thể diễn giải nó như một chỉ số tổng hợp đại diện cho "sức mua của khách hàng".5

#### **t-SNE / UMAP**

* **Ý nghĩa:** là các kỹ thuật giảm chiều phi tuyến, được thiết kế chủ yếu cho mục đích **trực quan hóa** dữ liệu nhiều chiều trong không gian 2D hoặc 3D. Chúng cố gắng bảo toàn cấu trúc lân cận cục bộ của dữ liệu.  
* **Ưu điểm:** Tạo ra các hình ảnh trực quan rất rõ ràng và dễ hiểu về cách các cụm dữ liệu được phân tách.  
* **Nhược điểm:** Tốn tài nguyên tính toán, và khoảng cách giữa các cụm trong không gian t-SNE/UMAP không nhất thiết phản ánh khoảng cách thực tế trong không gian ban đầu. Chúng không phù hợp để sử dụng cho các nhiệm vụ khác ngoài trực quan hóa.

#### **Autoencoders**

* **Ý nghĩa:** Là một loại mạng nơ-ron được sử dụng cho học không giám sát. Nó bao gồm hai phần: một **bộ mã hóa (encoder)** nén dữ liệu đầu vào thành một biểu diễn có chiều thấp hơn (gọi là mã hóa hoặc không gian tiềm ẩn), và một **bộ giải mã (decoder)** cố gắng tái tạo lại dữ liệu đầu vào từ mã hóa đó.  
* **Ứng dụng:** Được sử dụng để giảm chiều, khử nhiễu dữ liệu, và phát hiện bất thường (một điểm dữ liệu không thể tái tạo tốt có thể là một bất thường).

### **5.3. Khai phá luật kết hợp (Association Rule Learning)**

#### **Apriori Algorithm**

* **Ý nghĩa:** Là một thuật toán kinh điển để tìm ra các luật kết hợp dạng "Nếu {A} thì {B}" trong cơ sở dữ liệu giao dịch. Nó hoạt động dựa trên nguyên tắc "apriori": nếu một tập hợp các mục (itemset) xuất hiện thường xuyên, thì tất cả các tập hợp con của nó cũng phải xuất hiện thường xuyên.  
* **Các độ đo chính:**  
  * **Support (Độ hỗ trợ):** Tỷ lệ giao dịch chứa một itemset cụ thể.  
  * **Confidence (Độ tin cậy):** Xác suất có điều kiện để tìm thấy mục B khi đã có mục A.  
  * **Lift (Độ nâng):** Đo lường mức độ một quy tắc thú vị hơn so với sự ngẫu nhiên. Lift \> 1 cho thấy A và B có xu hướng xuất hiện cùng nhau.

#### **FP-Growth**

* **Ý nghĩa:** Là một cải tiến của Apriori, hiệu quả hơn trong việc tìm kiếm các itemset thường xuyên. Thay vì quét cơ sở dữ liệu nhiều lần, nó sử dụng một cấu trúc cây nhỏ gọn gọi là FP-tree để lưu trữ thông tin.  
* **Ưu điểm:** Nhanh hơn đáng kể so với Apriori trên các tập dữ liệu lớn.

### **5.4. Đánh giá trong Học không giám sát**

Việc đánh giá các mô hình học không giám sát khó khăn hơn so với học có giám sát vì không có "nhãn" đúng để so sánh.

* **Độ đo cho Phân cụm:**  
  * **Internal (không cần nhãn):**  
    * **Silhouette Score:** Đo lường mức độ một điểm dữ liệu giống với cụm của chính nó so với các cụm khác. Giá trị từ \-1 đến 1, càng cao càng tốt.  
    * **Davies-Bouldin Index:** Đo lường tỷ lệ giữa khoảng cách trong cụm và khoảng cách giữa các cụm. Càng thấp càng tốt.  
  * **External (cần nhãn thực tế để so sánh):** Adjusted Rand Index (ARI), Normalized Mutual Information (NMI).  
* **Độ đo cho Giảm chiều:**  
  * **Explained Variance (PCA):** Tỷ lệ phần trăm phương sai của dữ liệu gốc được giữ lại bởi các thành phần chính.  
  * **Reconstruction Error (Autoencoder):** Sai số giữa dữ liệu đầu vào và dữ liệu được tái tạo, đo lường mức độ mất mát thông tin.

## **Phần 6: Mạng Nơ-ron & Học sâu (Cơ bản)**

Mạng nơ-ron nhân tạo (Artificial Neural Networks) là các mô hình tính toán lấy cảm hứng từ cấu trúc và chức năng của não bộ sinh học. Chúng bao gồm các đơn vị xử lý được kết nối với nhau gọi là "nơ-ron". Học sâu (Deep Learning) là một lĩnh vực con của học máy, sử dụng các mạng nơ-ron có nhiều lớp ẩn (sâu) để học các biểu diễn dữ liệu ngày càng phức tạp và trừu tượng.

### **6.1. Perceptron & Multilayer Perceptron (MLP)**

#### **Perceptron**

* Cấu trúc: Là dạng mạng nơ-ron đơn giản nhất, bao gồm một nơ-ron duy nhất. Nó nhận một vector đầu vào x, nhân với một vector trọng số w, cộng thêm một độ lệch (bias) b, và sau đó đưa kết quả qua một hàm kích hoạt (activation function) f để tạo ra đầu ra.

  y=f(i∑​wi​xi​+b)=f(w⋅x+b)  
* **Giới hạn:** Một Perceptron đơn lẻ chỉ có thể giải quyết các bài toán có thể phân tách tuyến tính (linearly separable). Nó có thể học các hàm logic như AND hoặc OR, nhưng không thể học hàm XOR.

#### **Multilayer Perceptron (MLP)**

* **Cấu trúc:** Để vượt qua giới hạn của Perceptron, MLP được giới thiệu bằng cách thêm một hoặc nhiều **lớp ẩn (hidden layers)** vào giữa lớp đầu vào và lớp đầu ra. Mỗi nơ-ron trong một lớp được kết nối đầy đủ (fully-connected) với tất cả các nơ-ron trong lớp tiếp theo.  
* **Khả năng:** Việc thêm các lớp ẩn và sử dụng các hàm kích hoạt phi tuyến cho phép MLP học các mối quan hệ phi tuyến phức tạp và giải quyết các bài toán mà Perceptron đơn không thể.  
* **Huấn luyện:** MLP được huấn luyện bằng thuật toán lan truyền ngược (backpropagation).

### **6.2. Hàm kích hoạt (Activation Functions)**

Hàm kích hoạt quyết định đầu ra của một nơ-ron. Vai trò quan trọng của chúng là đưa tính phi tuyến vào mạng, cho phép mạng học các mẫu phức tạp. Nếu không có hàm kích hoạt phi tuyến, một mạng nơ-ron sâu sẽ chỉ tương đương với một mô hình tuyến tính đơn lớp.

| Hàm Kích hoạt | Công thức | Phạm vi đầu ra | Ưu điểm | Nhược điểm | Khi dùng |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Sigmoid** | σ(x)=1+e−x1​ | (0, 1\) | Mượt, có thể diễn giải là xác suất | Gây ra vấn đề Vanishing Gradient, đầu ra không phải zero-centered 27 | Lớp đầu ra của bài toán phân loại nhị phân |
| **Tanh** | tanh(x)=ex+e−xex−e−x​ | (-1, 1\) | Đầu ra zero-centered, giúp hội tụ nhanh hơn Sigmoid | Vẫn bị Vanishing Gradient 27 | Các lớp ẩn trong các mạng nông |
| **ReLU** | ReLU(x)=max(0,x) |  | Lựa chọn mặc định cho hầu hết các lớp ẩn |  |  |
| **Leaky ReLU** | x nếu x\>0, ngược lại αx | (−∞, ∞) | Giải quyết vấn đề "Dying ReLU" | Hiệu quả không phải lúc nào cũng vượt trội ReLU | Khi gặp vấn đề "Dying ReLU" |
| **Softmax** | ∑j​exj​exi​​ | (0, 1), tổng bằng 1 | Chuyển đổi logits thành phân phối xác suất | Chỉ dùng ở lớp cuối | Lớp đầu ra của bài toán phân loại đa lớp |

*Bảng 6.1: So sánh các hàm kích hoạt phổ biến. Bảng này cung cấp một cái nhìn tổng quan về các đặc tính, ưu nhược điểm và trường hợp sử dụng điển hình của các hàm kích hoạt, giúp người đọc lựa chọn hàm phù hợp cho kiến trúc mạng của mình.*

### **6.3. Lan truyền ngược & Huấn luyện**

Quá trình huấn luyện một mạng nơ-ron là một vòng lặp gồm hai giai đoạn: lan truyền xuôi và lan truyền ngược.

#### **Lan truyền xuôi (Forward Pass)**

Dữ liệu đầu vào được đưa qua mạng, từ lớp đầu vào, qua các lớp ẩn, đến lớp đầu ra. Tại mỗi lớp, đầu ra được tính toán bằng cách nhân đầu vào của lớp đó với ma trận trọng số, cộng với bias, và sau đó đưa qua hàm kích hoạt. Quá trình này tạo ra một dự đoán ở lớp đầu ra. Sau đó, một hàm mất mát (ví dụ: Cross-Entropy cho phân loại, MSE cho hồi quy) được sử dụng để so sánh dự đoán này với giá trị thực tế và tính toán sai số.29

#### **Lan truyền ngược (Backward Pass \- Backpropagation)**

Đây là thuật toán cốt lõi cho phép mạng học. Nó tính toán gradient của hàm mất mát đối với từng trọng số và bias trong mạng. Bằng cách sử dụng quy tắc chuỗi của giải tích, sai số được "lan truyền ngược" từ lớp đầu ra về các lớp trước đó. Gradient này cho biết mỗi trọng số cần được điều chỉnh như thế nào (tăng hay giảm) và mức độ bao nhiêu để giảm thiểu sai số tổng thể.29

#### **Cập nhật trọng số**

Sau khi tính toán gradient, các trọng số được cập nhật bằng một thuật toán tối ưu hóa, phổ biến nhất là một biến thể của Gradient Descent.

w:=w−η∂w∂L​

Trong đó η là tốc độ học và ∂w∂L​ là gradient của hàm mất mát L theo trọng số w. Các optimizer hiện đại như Adam thường được sử dụng để quá trình cập nhật này hiệu quả và ổn định hơn.11

### **6.4. Quá khớp & Chính quy hóa**

Mạng nơ-ron, với số lượng tham số lớn, rất dễ bị **quá khớp (overfitting)**—tức là nó học thuộc lòng dữ liệu huấn luyện, bao gồm cả nhiễu, và do đó hoạt động kém trên dữ liệu mới. Chính quy hóa (Regularization) là một tập hợp các kỹ thuật được sử dụng để chống lại hiện tượng này.

* **Dropout:** Là một kỹ thuật chính quy hóa đơn giản nhưng rất hiệu quả. Trong mỗi lần lặp huấn luyện, một tỷ lệ nơ-ron ngẫu nhiên trong một lớp sẽ bị "tắt" (đầu ra của chúng được đặt bằng 0). Điều này buộc mạng không được phụ thuộc quá nhiều vào bất kỳ nơ-ron nào và phải học các đặc trưng mạnh mẽ và dư thừa hơn. Nó có thể được xem như việc huấn luyện một tập hợp lớn các mạng con khác nhau và lấy trung bình kết quả của chúng.31  
* **L1/L2 Regularization:** Tương tự như trong hồi quy, thêm một thành phần phạt vào hàm mất mát dựa trên độ lớn của các trọng số, khuyến khích mạng sử dụng các trọng số nhỏ hơn.  
* **Early Stopping:** Theo dõi hiệu suất của mô hình trên tập xác thực trong quá trình huấn luyện và dừng lại khi hiệu suất này bắt đầu giảm, ngăn không cho mô hình tiếp tục học và quá khớp.  
* **Batch Normalization:** Chuẩn hóa đầu ra của một lớp trước khi đưa vào lớp tiếp theo. Nó giúp ổn định và tăng tốc quá trình huấn luyện, đồng thời có tác dụng chính quy hóa nhẹ.  
* **Data Augmentation:** Tạo ra các dữ liệu huấn luyện mới bằng cách áp dụng các phép biến đổi ngẫu nhiên lên dữ liệu hiện có (ví dụ: xoay, lật, cắt ảnh). Điều này làm tăng sự đa dạng của tập huấn luyện và giúp mô hình tổng quát hóa tốt hơn.

## **Phần 7: Đánh giá và Xác thực Mô hình**

Xây dựng một mô hình học máy không chỉ là việc huấn luyện nó mà còn phải đảm bảo rằng nó hoạt động tốt trên dữ liệu thực tế. Giai đoạn đánh giá và xác thực là rất quan trọng để hiểu được hiệu suất thực sự của mô hình, chẩn đoán các vấn đề như quá khớp hoặc dưới khớp, và chọn ra mô hình tốt nhất cho bài toán.

### **7.1. Đánh đổi Bias–Variance**

Hiệu suất của một mô hình học máy bị chi phối bởi một sự đánh đổi cơ bản giữa hai nguồn lỗi: bias và variance.

* **Bias (Độ chệch):** Là sai số do các giả định sai lầm trong mô hình. Một mô hình có bias cao quá đơn giản và không thể nắm bắt được các quy luật phức tạp trong dữ liệu, dẫn đến hiện tượng **dưới khớp (underfitting)**. Dấu hiệu là mô hình có sai số cao trên cả tập huấn luyện và tập kiểm tra.34  
* **Variance (Phương sai):** Là sai số do sự nhạy cảm của mô hình với các biến động nhỏ trong tập dữ liệu huấn luyện. Một mô hình có variance cao quá phức tạp và học cả nhiễu trong dữ liệu, dẫn đến hiện tượng **quá khớp (overfitting)**. Dấu hiệu là mô hình có sai số rất thấp trên tập huấn luyện nhưng sai số cao trên tập kiểm tra.34

**Sự đánh đổi (Tradeoff):**

* Khi tăng độ phức tạp của mô hình, bias có xu hướng giảm nhưng variance lại tăng lên.  
* Khi giảm độ phức tạp của mô hình, variance giảm nhưng bias lại tăng.  
  Mục tiêu là tìm ra một mô hình có độ phức tạp vừa phải, cân bằng được cả hai nguồn lỗi này để đạt được sai số tổng thể thấp nhất trên dữ liệu chưa thấy. Các kỹ thuật như chính quy hóa và kiểm định chéo là công cụ để quản lý sự đánh đổi này.

### **7.2. Các chiến lược xác thực**

#### **Hold-out Validation**

Phương pháp đơn giản nhất, chia dữ liệu thành ba tập: huấn luyện, xác thực, và kiểm tra. Tuy nhiên, kết quả đánh giá có thể phụ thuộc nhiều vào cách chia dữ liệu ngẫu nhiên.

#### **Cross-Validation (CV)**

Cung cấp một ước tính hiệu suất mạnh mẽ và ổn định hơn.

* **K-Fold CV:** Chia dữ liệu thành k phần. Lặp lại k lần, mỗi lần sử dụng một phần làm tập xác thực và k-1 phần còn lại để huấn luyện.  
* **Stratified K-Fold:** Quan trọng cho dữ liệu mất cân bằng, đảm bảo tỷ lệ các lớp được duy trì trong mỗi phần chia.16  
* **Leave-One-Out (LOO):** Một trường hợp đặc biệt của K-Fold CV với k bằng số lượng mẫu. Rất tốn kém về mặt tính toán nhưng hữu ích cho các tập dữ liệu cực nhỏ.

### **7.3. Tinh chỉnh siêu tham số (Hyperparameter Tuning)**

Siêu tham số là các tham số của mô hình không được học trong quá trình huấn luyện mà phải được thiết lập trước. Việc tìm ra bộ siêu tham số tối ưu là rất quan trọng để tối đa hóa hiệu suất.

* **Grid Search:** Thử nghiệm một cách toàn diện tất cả các kết hợp có thể có của các siêu tham số được chỉ định. Đảm bảo tìm ra kết hợp tốt nhất trong lưới, nhưng rất tốn tài nguyên.  
* **Random Search:** Thử nghiệm một số lượng kết hợp ngẫu nhiên từ không gian siêu tham số. Thường hiệu quả hơn Grid Search, đặc biệt khi một số siêu tham số quan trọng hơn những siêu tham số khác.  
* **Bayesian Optimization:** Một cách tiếp cận thông minh hơn, sử dụng kết quả của các lần thử trước đó để xây dựng một mô hình xác suất của hàm mục tiêu và quyết định điểm nào cần thử nghiệm tiếp theo để tối đa hóa khả năng cải thiện.

### **7.4. Tinh chỉnh ngưỡng (Threshold Tuning)**

Nhiều mô hình phân loại (như Hồi quy Logistic) không trả về một nhãn cứng mà là một xác suất. Ngưỡng mặc định để chuyển đổi xác suất này thành nhãn thường là 0.5, nhưng đây không phải lúc nào cũng là lựa chọn tối ưu.

* **Tác động:** Việc điều chỉnh ngưỡng này cho phép chúng ta đánh đổi giữa Precision và Recall.  
  * **Giảm ngưỡng (\< 0.5):** Sẽ phân loại nhiều mẫu hơn là "dương tính", làm tăng Recall nhưng giảm Precision. Hữu ích khi việc bỏ sót một trường hợp dương tính (False Negative) là rất nguy hiểm (ví dụ: chẩn đoán ung thư).  
  * **Tăng ngưỡng (\> 0.5):** Sẽ yêu cầu mô hình "chắc chắn" hơn trước khi phân loại là "dương tính", làm tăng Precision nhưng giảm Recall. Hữu ích khi một dự đoán sai dương tính (False Positive) gây ra nhiều phiền toái (ví dụ: lọc nhầm email quan trọng).  
* **Công cụ:** Đường cong ROC và đường cong Precision-Recall là các công cụ hữu ích để chọn ngưỡng tối ưu dựa trên yêu cầu của bài toán.

## **Phần 8: Các khía cạnh thực tiễn trong Học Máy**

Việc đưa một mô hình học máy từ phòng thí nghiệm ra môi trường sản xuất đòi hỏi phải giải quyết nhiều vấn đề thực tiễn ngoài việc chỉ tối ưu hóa độ chính xác. Các vấn đề này bao gồm khả năng diễn giải, xử lý dữ liệu đặc biệt, khả năng mở rộng, và các cân nhắc về đạo đức.

### **8.1. Khả năng diễn giải mô hình (Model Interpretability)**

Trong nhiều ứng dụng, đặc biệt là trong các lĩnh vực có rủi ro cao như tài chính và y tế, việc hiểu **tại sao** một mô hình đưa ra một dự đoán cụ thể cũng quan trọng như chính dự đoán đó.

* **Feature Importance:** Các mô hình dựa trên cây (Random Forest, XGBoost) có thể cung cấp một thước đo về tầm quan trọng của mỗi đặc trưng đối với quyết định chung của mô hình.  
* **SHAP (SHapley Additive exPlanations):** Dựa trên lý thuyết trò chơi, SHAP gán cho mỗi đặc trưng một giá trị "đóng góp" vào một dự đoán cụ thể cho một điểm dữ liệu. Nó cung cấp cả giải thích cục bộ (cho từng dự đoán) và toàn cục (tầm quan trọng trung bình của đặc trưng) và có nền tảng lý thuyết vững chắc về tính nhất quán.36  
* **LIME (Local Interpretable Model-agnostic Explanations):** Là một kỹ thuật giải thích cục bộ. Để giải thích một dự đoán, LIME tạo ra các mẫu dữ liệu mới xung quanh điểm dữ liệu đó, nhận dự đoán của mô hình "hộp đen" trên các mẫu này, và sau đó huấn luyện một mô hình đơn giản, có thể diễn giải (như hồi quy tuyến tính) để mô phỏng hành vi của mô hình phức tạp tại khu vực lân cận đó.36

### **8.2. Xử lý dữ liệu mất cân bằng (Imbalanced Data)**

Trong nhiều bài toán thực tế (phát hiện gian lận, chẩn đoán bệnh hiếm), số lượng mẫu của một lớp (lớp thiểu số) ít hơn đáng kể so với các lớp khác. Nếu không xử lý, mô hình sẽ có xu hướng bỏ qua lớp thiểu số và chỉ dự đoán lớp đa số, dẫn đến độ chính xác (accuracy) cao nhưng vô dụng.

* **Resampling:**  
  * **Oversampling (Lấy mẫu quá mức):** Tăng số lượng mẫu của lớp thiểu số. Một kỹ thuật phổ biến là **SMOTE (Synthetic Minority Over-sampling Technique)**, tạo ra các mẫu tổng hợp mới bằng cách nội suy giữa các mẫu thiểu số hiện có và các hàng xóm gần nhất của chúng.37  
  * **Undersampling (Lấy mẫu dưới mức):** Giảm số lượng mẫu của lớp đa số.  
* **Class Weights:** Điều chỉnh hàm mất mát để phạt nặng hơn những lỗi sai trên lớp thiểu số, buộc mô hình phải chú ý hơn đến chúng.  
* **Sử dụng các độ đo phù hợp:** Tránh sử dụng Accuracy. Thay vào đó, tập trung vào Precision, Recall, F1-Score, và ROC-AUC.

### **8.3. Khả năng mở rộng và hiệu quả**

Khi làm việc với các tập dữ liệu lớn hoặc các mô hình phức tạp, hiệu quả tính toán và khả năng mở rộng trở thành yếu tố quan trọng.

* **Batching:** Thay vì xử lý toàn bộ dữ liệu cùng một lúc, dữ liệu được chia thành các lô nhỏ (batches) để huấn luyện, giúp giảm yêu cầu về bộ nhớ.  
* **Parallelization (Song song hóa):** Tận dụng nhiều lõi CPU hoặc GPU để tăng tốc độ tính toán, đặc biệt là trong huấn luyện mạng nơ-ron sâu.  
* **Distributed ML (Học máy phân tán):** Sử dụng các framework như Apache Spark MLlib hoặc Horovod để huấn luyện mô hình trên một cụm nhiều máy tính, cho phép xử lý các tập dữ liệu khổng lồ.  
* **Model Compression (Nén mô hình):** Các kỹ thuật như lượng tử hóa (quantization) hoặc tỉa (pruning) được sử dụng để giảm kích thước của mô hình, giúp triển khai trên các thiết bị có tài nguyên hạn chế như điện thoại di động.

### **8.4. Triển khai trong môi trường sản xuất (Deployment in Production)**

* **Model Serving:** Đóng gói mô hình đã huấn luyện thành một dịch vụ có thể truy cập qua API (thường là REST API) bằng các framework như Flask, FastAPI, hoặc các máy chủ mô hình chuyên dụng như TensorFlow Serving.  
* **Monitoring (Giám sát):** Theo dõi liên tục hiệu suất của mô hình trong thực tế để phát hiện các vấn đề như **data drift** (phân phối của dữ liệu đầu vào thay đổi) và **model drift** (mối quan hệ giữa đầu vào và đầu ra thay đổi).  
* **CI/CD for ML (MLOps):** Áp dụng các nguyên tắc của DevOps vào quy trình học máy để tự động hóa việc xây dựng, kiểm thử, và triển khai mô hình, tạo ra một vòng đời sản phẩm ML bền vững và có thể lặp lại.

### **8.5. Quyền riêng tư và Đạo đức Dữ liệu**

* **Thiên vị trong dữ liệu (Bias in data):** Các mô hình học máy được huấn luyện trên dữ liệu lịch sử có thể học và khuếch đại các định kiến xã hội hiện có trong dữ liệu đó.  
* **Công bằng (Fairness):** Cần đảm bảo rằng các quyết định của mô hình là công bằng đối với các nhóm nhân khẩu học khác nhau (ví dụ: giới tính, chủng tộc).  
* **Học máy bảo vệ quyền riêng tư (Privacy-preserving ML):**  
  * **Differential Privacy:** Thêm nhiễu vào dữ liệu hoặc thuật toán để bảo vệ danh tính của các cá nhân.  
  * **Federated Learning:** Huấn luyện mô hình trực tiếp trên các thiết bị của người dùng (ví dụ: điện thoại) mà không cần gửi dữ liệu thô về máy chủ trung tâm.

## **Phần 9: Các chủ đề nâng cao trong Học Máy**

Khi đã nắm vững các nguyên tắc cơ bản, lĩnh vực học máy mở ra nhiều hướng đi nâng cao, mỗi hướng giải quyết các loại vấn đề phức tạp và chuyên biệt.

### **9.1. Học tăng cường (Reinforcement Learning \- RL)**

* **Ý tưởng chính:** Như đã giới thiệu, RL là về việc một agent học cách hành động trong một môi trường thông qua cơ chế thử và sai, nhằm tối đa hóa phần thưởng tích lũy dài hạn.  
* **Các thành phần chính:**  
  * **State (S):** Trạng thái hiện tại của môi trường.  
  * **Action (A):** Các hành động mà agent có thể thực hiện.  
  * **Reward (R):** Tín hiệu phản hồi từ môi trường sau một hành động.  
  * **Policy (π):** Chiến lược hoặc quy tắc mà agent sử dụng để chọn hành động từ một trạng thái.  
* **Thuật toán tiêu biểu: Q-Learning**  
  * Q-Learning là một thuật toán RL không cần mô hình (model-free), học một hàm chất lượng hành động (action-value function), ký hiệu là Q(s,a). Hàm này ước tính phần thưởng tích lũy tối đa có thể đạt được khi thực hiện hành động a từ trạng thái s và sau đó tuân theo chính sách tối ưu.1  
  * Nó sử dụng một bảng gọi là **Q-table** để lưu trữ các giá trị Q(s,a) cho tất cả các cặp trạng thái-hành động. Bảng này được cập nhật lặp đi lặp lại thông qua kinh nghiệm của agent theo phương trình Bellman.1  
* **Ứng dụng:** Trí tuệ nhân tạo chơi game (AlphaGo), điều khiển robot tự hành, hệ thống đề xuất động.

### **9.2. Mô hình sinh (Generative Models)**

Không giống như các mô hình phân biệt (discriminative models) học cách phân loại dữ liệu, các mô hình sinh học cách tạo ra dữ liệu mới có cùng phân phối với dữ liệu huấn luyện.

#### **Generative Adversarial Networks (GANs)**

* **Ý tưởng chính:** GAN bao gồm hai mạng nơ-ron cạnh tranh với nhau trong một trò chơi:  
  * **Generator (Bộ sinh):** Cố gắng tạo ra dữ liệu giả (ví dụ: hình ảnh) trông giống như thật.  
  * **Discriminator (Bộ phân biệt):** Cố gắng phân biệt giữa dữ liệu thật từ tập huấn luyện và dữ liệu giả do Generator tạo ra.  
* **Quá trình huấn luyện:** Hai mạng được huấn luyện đồng thời. Generator ngày càng giỏi hơn trong việc tạo ra dữ liệu giả để đánh lừa Discriminator, trong khi Discriminator ngày càng giỏi hơn trong việc phát hiện hàng giả. Quá trình này tiếp tục cho đến khi Generator tạo ra dữ liệu giả thực tế đến mức Discriminator không thể phân biệt được nữa.40  
* **Ứng dụng:** Tạo ra hình ảnh, video (deepfake), và âm nhạc nghệ thuật; tăng cường dữ liệu (data augmentation).

#### **Variational Autoencoders (VAEs)**

* **Ý tưởng chính:** VAE là một loại mô hình sinh dựa trên kiến trúc autoencoder. Nó học cách mã hóa dữ liệu đầu vào thành một không gian tiềm ẩn (latent space) tuân theo một phân phối xác suất (thường là Gaussian). Để tạo dữ liệu mới, nó lấy một mẫu ngẫu nhiên từ không gian tiềm ẩn này và đưa qua bộ giải mã.  
* **Ứng dụng:** Tạo hình ảnh, nén dữ liệu, và phát hiện bất thường.

### **9.3. Học chuyển giao (Transfer Learning)**

* **Ý tưởng chính:** Thay vì huấn luyện một mô hình từ đầu (tốn nhiều dữ liệu và tài nguyên), học chuyển giao tận dụng kiến thức đã được học từ một mô hình đã được huấn luyện trước (pre-trained model) trên một tập dữ liệu lớn và một nhiệm vụ liên quan.  
* **Quy trình:**  
  1. Lấy một mô hình đã được huấn luyện trước (ví dụ: ResNet được huấn luyện trên ImageNet, hoặc BERT được huấn luyện trên kho văn bản khổng lồ).  
  2. "Đóng băng" (freeze) các lớp đầu tiên của mô hình (những lớp này đã học các đặc trưng chung như cạnh, góc trong ảnh hoặc cấu trúc ngữ pháp trong văn bản).  
  3. Thay thế hoặc thêm các lớp cuối cùng để phù hợp với nhiệm vụ mới.  
  4. **Tinh chỉnh (fine-tuning):** Huấn luyện lại các lớp cuối (và có thể là một vài lớp trước đó) trên tập dữ liệu nhỏ hơn của nhiệm vụ mới.43  
* **Ưu điểm:** Tiết kiệm đáng kể thời gian và dữ liệu huấn luyện, thường mang lại hiệu suất tốt hơn, đặc biệt khi dữ liệu cho nhiệm vụ mới là hạn chế.  
* **Ứng dụng:** Hầu hết các ứng dụng hiện đại trong thị giác máy tính (Computer Vision) và xử lý ngôn ngữ tự nhiên (NLP) đều sử dụng học chuyển giao.

### **9.4. Học trực tuyến và Học tăng dần (Online & Incremental Learning)**

* **Ý tưởng chính:** Các mô hình học máy truyền thống được huấn luyện ngoại tuyến (offline) trên một tập dữ liệu tĩnh. Học trực tuyến cho phép mô hình cập nhật liên tục khi có dữ liệu mới đến dưới dạng một dòng (stream), mà không cần phải huấn luyện lại từ đầu trên toàn bộ dữ liệu.  
* **Kỹ thuật:** Stochastic Gradient Descent (SGD) về bản chất là một thuật toán học trực tuyến. Các thuật toán khác như Hoeffding Trees cũng được thiết kế cho môi trường này.  
* **Ứng dụng:** Dự đoán thị trường chứng khoán, đặt giá thầu quảng cáo thời gian thực, phân tích dữ liệu từ cảm biến IoT.

### **Case Study 1: Hệ thống Đề xuất của Netflix**

Hệ thống đề xuất của Netflix là một ví dụ điển hình về việc áp dụng nhiều kỹ thuật học máy phức tạp để giải quyết một bài toán kinh doanh cốt lõi: giữ chân người dùng bằng cách giúp họ tìm thấy nội dung yêu thích một cách dễ dàng. Hơn 80% thời lượng xem trên Netflix đến từ các đề xuất được cá nhân hóa.45

* **Thuật toán sử dụng:**  
  * **Lọc cộng tác (Collaborative Filtering):** Đây là nền tảng của hệ thống. Nó hoạt động dựa trên nguyên tắc "những người dùng có sở thích tương tự trong quá khứ sẽ có sở thích tương tự trong tương lai". Netflix xác định các nhóm người dùng có lịch sử xem và xếp hạng giống nhau, sau đó đề xuất các bộ phim mà những người dùng khác trong nhóm đã thích nhưng người dùng hiện tại chưa xem.45  
  * **Lọc dựa trên nội dung (Content-Based Filtering):** Đề xuất các mục tương tự như những gì người dùng đã thích trước đây. Nó phân tích các thuộc tính của nội dung (metadata) như thể loại, diễn viên, đạo diễn, và từ khóa mô tả để tìm ra sự tương đồng.45  
  * **Mô hình học sâu và Foundation Models:** Gần đây, Netflix đã chuyển sang các mô hình phức tạp hơn, bao gồm cả các "Foundation Model" lấy cảm hứng từ các mô hình ngôn ngữ lớn (LLM). Các mô hình này có thể xử lý lịch sử tương tác dài hạn của người dùng và tạo ra các biểu diễn nhúng (embeddings) phong phú cho cả người dùng và nội dung. Các embeddings này sau đó có thể được sử dụng trong nhiều ứng dụng hạ nguồn khác nhau, từ việc tạo ứng viên đề xuất đến tinh chỉnh cho các giao diện cụ thể.49  
* **Cá nhân hóa đa tầng:** Sự cá nhân hóa không chỉ dừng lại ở việc đề xuất phim gì. Nó bao gồm:  
  * **Sắp xếp các hàng (Rows):** Thứ tự các hàng trên trang chủ ("Thịnh hành", "Tiếp tục xem", "Vì bạn đã xem...") được cá nhân hóa cho từng người dùng.  
  * **Xếp hạng trong hàng:** Thứ tự các bộ phim trong mỗi hàng cũng được xếp hạng riêng cho từng người.  
  * **Cá nhân hóa ảnh bìa (Artwork Personalization):** Netflix thậm chí còn chọn ảnh bìa (thumbnail) khác nhau cho cùng một bộ phim để hiển thị cho những người dùng khác nhau, dựa trên sở thích của họ (ví dụ: hiển thị ảnh của diễn viên hài cho người thích phim hài, hoặc ảnh của cặp đôi lãng mạn cho người thích phim tình cảm).45  
* **Kiến trúc hệ thống:** Netflix sử dụng kiến trúc microservices, cho phép các thành phần khác nhau của hệ thống đề xuất (thu thập dữ liệu, huấn luyện ngoại tuyến, phục vụ trực tuyến) được phát triển và mở rộng một cách độc lập. Họ cũng sử dụng các lớp caching phức tạp (như EV cache) để đảm bảo độ trễ thấp khi phục vụ hàng triệu yêu cầu mỗi giây.47

### **Case Study 2: Phát hiện Gian lận trong Ngành Ngân hàng**

Phát hiện gian lận là một ứng dụng quan trọng của học máy trong ngành tài chính, nơi các giao dịch bất thường cần được xác định nhanh chóng và chính xác. Đây là một bài toán phân loại điển hình, nhưng có những thách thức riêng.

* **Thách thức:**  
  * **Dữ liệu mất cân bằng nghiêm trọng:** Số lượng giao dịch gian lận cực kỳ nhỏ so với giao dịch hợp lệ (ví dụ: \< 0.1%).  
  * **Tính thích ứng của kẻ gian:** Những kẻ gian lận liên tục thay đổi chiến thuật, đòi hỏi mô hình phải có khả năng thích ứng.  
  * **Yêu cầu thời gian thực:** Các quyết định phải được đưa ra trong mili giây.  
* **Mô hình và Kỹ thuật:**  
  * **Thuật toán:** Các mô hình ensemble như Random Forest và Gradient Boosting (đặc biệt là XGBoost, LightGBM) rất phổ biến vì hiệu suất cao. Các mô hình phát hiện bất thường (Anomaly Detection) như Isolation Forest hoặc Autoencoders cũng được sử dụng.  
  * **Kỹ thuật đặc trưng (Feature Engineering):** Đây là yếu tố quyết định thành công. Các đặc trưng được tạo ra để nắm bắt hành vi của người dùng, chẳng hạn như: số lượng giao dịch trong một khoảng thời gian ngắn, giá trị giao dịch so với mức trung bình, vị trí giao dịch, thời gian trong ngày, v.v.  
  * **Xử lý mất cân bằng:** Các kỹ thuật như SMOTE để tạo thêm mẫu gian lận, hoặc sử dụng class weights trong hàm mất mát là bắt buộc.  
* **Ví dụ thực tế:** Một ngân hàng toàn cầu đã hợp tác với Cognizant để xây dựng một giải pháp phát hiện gian lận séc dựa trên AI. Họ đã sử dụng một mạng nơ-ron trên nền tảng Google TensorFlow, được huấn luyện trên một cơ sở dữ liệu lịch sử các séc đã được quét, bao gồm cả các séc gian lận đã biết. Mô hình này gán một điểm tin cậy cho mỗi séc được quét và gắn cờ các trường hợp đáng ngờ trong thời gian thực.  
* **Kết quả:** Giải pháp đã giúp **giảm 50% các giao dịch gian lận**, tiết kiệm cho ngân hàng **20 triệu USD mỗi năm** và có thời gian phản hồi dưới 70 mili giây, xử lý tới 1.200 séc mỗi giây.51

### **Case Study 3: Nhận dạng Đối tượng cho Xe tự lái**

Xe tự lái dựa vào một loạt các cảm biến (camera, LiDAR, radar) và các thuật toán học máy phức tạp để "nhìn" và hiểu thế giới xung quanh. Nhận dạng đối tượng (Object Detection) là một nhiệm vụ cốt lõi, cho phép xe xác định và định vị các đối tượng khác như xe cộ, người đi bộ, người đi xe đạp, và biển báo giao thông.

* **Vai trò của Mạng Nơ-ron Tích chập (CNN):** CNN là kiến trúc nền tảng cho hầu hết các mô hình thị giác máy tính hiện đại. Các lớp đầu tiên của CNN học các đặc trưng cấp thấp như cạnh và màu sắc, trong khi các lớp sâu hơn học các đặc trưng phức tạp hơn như hình dạng của một chiếc xe hơi hoặc khuôn mặt của một người đi bộ.52  
* **Các kiến trúc phổ biến:**  
  * **Họ R-CNN (Region-based CNN):** Các mô hình như R-CNN, Fast R-CNN, và Faster R-CNN hoạt động theo hai giai đoạn: đầu tiên, đề xuất các "vùng quan tâm" (regions of interest) có khả năng chứa đối tượng, sau đó, sử dụng một CNN để phân loại đối tượng trong các vùng đó. Chúng thường có độ chính xác cao nhưng chậm hơn.52  
  * **YOLO (You Only Look Once):** Là một kiến trúc một giai đoạn (one-stage). Nó chia hình ảnh thành một lưới và dự đoán đồng thời các hộp giới hạn (bounding boxes) và xác suất lớp cho mỗi ô lưới trong một lần duy nhất. YOLO nổi tiếng vì tốc độ cực nhanh, rất phù hợp cho các ứng dụng thời gian thực như xe tự lái, mặc dù có thể kém chính xác hơn một chút so với các mô hình hai giai đoạn đối với các vật thể nhỏ.52  
  * **SSD (Single Shot MultiBox Detector):** Một kiến trúc một giai đoạn khác, cố gắng cân bằng giữa tốc độ của YOLO và độ chính xác của họ R-CNN.  
* **Thách thức:**  
  * **Điều kiện thời tiết và ánh sáng kém:** Mưa, tuyết, sương mù, hoặc ban đêm làm giảm chất lượng hình ảnh và gây khó khăn cho việc nhận dạng.  
  * **Vật thể bị che khuất (Occlusion):** Các đối tượng bị che khuất một phần rất khó để nhận dạng chính xác.  
  * **Tốc độ thời gian thực:** Hệ thống phải xử lý hàng chục khung hình mỗi giây để đưa ra quyết định kịp thời. Đây là lý do tại sao các mô hình như YOLO rất được ưa chuộng.

#### **Nguồn trích dẫn**

1. Q-learning \- Wikipedia, truy cập vào tháng 8 18, 2025, [https://en.wikipedia.org/wiki/Q-learning](https://en.wikipedia.org/wiki/Q-learning)  
2. A Gentle Introduction to Q-Learning \- MachineLearningMastery.com, truy cập vào tháng 8 18, 2025, [https://machinelearningmastery.com/a-gentle-introduction-to-q-learning/](https://machinelearningmastery.com/a-gentle-introduction-to-q-learning/)  
3. An introduction to Q-Learning: Reinforcement Learning \- FloydHub Blog, truy cập vào tháng 8 18, 2025, [https://floydhub.ghost.io/an-introduction-to-q-learning-reinforcement-learning/](https://floydhub.ghost.io/an-introduction-to-q-learning-reinforcement-learning/)  
4. Understanding the Role of Eigenvectors and Eigenvalues in PCA ..., truy cập vào tháng 8 18, 2025, [https://medium.com/@dareyadewumi650/understanding-the-role-of-eigenvectors-and-eigenvalues-in-pca-dimensionality-reduction-10186dad0c5c](https://medium.com/@dareyadewumi650/understanding-the-role-of-eigenvectors-and-eigenvalues-in-pca-dimensionality-reduction-10186dad0c5c)  
5. 11.4 \- Interpretation of the Principal Components | STAT 505, truy cập vào tháng 8 18, 2025, [https://online.stat.psu.edu/stat505/lesson/11/11.4](https://online.stat.psu.edu/stat505/lesson/11/11.4)  
6. Gradient descent (article) | Khan Academy, truy cập vào tháng 8 18, 2025, [https://www.khanacademy.org/math/multivariable-calculus/applications-of-multivariable-derivatives/optimizing-multivariable-functions/a/what-is-gradient-descent](https://www.khanacademy.org/math/multivariable-calculus/applications-of-multivariable-derivatives/optimizing-multivariable-functions/a/what-is-gradient-descent)  
7. What Is Gradient Descent in Machine Learning? | Coursera, truy cập vào tháng 8 18, 2025, [https://www.coursera.org/articles/what-is-gradient-descent](https://www.coursera.org/articles/what-is-gradient-descent)  
8. The Naïve Bayes Classifier \- MyEducator, truy cập vào tháng 8 18, 2025, [https://app.myeducator.com/reader/web/1755b/naives%20bayes/g74yi/](https://app.myeducator.com/reader/web/1755b/naives%20bayes/g74yi/)  
9. app.myeducator.com, truy cập vào tháng 8 18, 2025, [https://app.myeducator.com/reader/web/1755b/naives%20bayes/g74yi/\#:\~:text=The%20Na%C3%AFve%20Bayes%20classifier%20tweaks,part%20of%20a%20certain%20class.](https://app.myeducator.com/reader/web/1755b/naives%20bayes/g74yi/#:~:text=The%20Na%C3%AFve%20Bayes%20classifier%20tweaks,part%20of%20a%20certain%20class.)  
10. massedcompute.com, truy cập vào tháng 8 18, 2025, [https://massedcompute.com/faq-answers/?question=What%20are%20the%20key%20differences%20between%20Adam%20and%20SGD%20optimizers%20in%20large%20language%20model%20training?\#:\~:text=SGD%3A%20More%20sensitive%20to%20noisy,adaptive%20learning%20rates%20and%20momentum.](https://massedcompute.com/faq-answers/?question=What+are+the+key+differences+between+Adam+and+SGD+optimizers+in+large+language+model+training?#:~:text=SGD%3A%20More%20sensitive%20to%20noisy,adaptive%20learning%20rates%20and%20momentum.)  
11. Stochastic Gradient Descent (SGD) and Adam | by Hey Amit | Data ..., truy cập vào tháng 8 18, 2025, [https://medium.com/data-scientists-diary/stochastic-gradient-descent-sgd-and-adam-4fe496ef1bbf](https://medium.com/data-scientists-diary/stochastic-gradient-descent-sgd-and-adam-4fe496ef1bbf)  
12. When to Normalize or Standardize Data | Secoda, truy cập vào tháng 8 18, 2025, [https://www.secoda.co/learn/when-to-normalize-or-standardize-data](https://www.secoda.co/learn/when-to-normalize-or-standardize-data)  
13. www.datacamp.com, truy cập vào tháng 8 18, 2025, [https://www.datacamp.com/tutorial/normalization-vs-standardization\#:\~:text=Normalization%20can%20help%20adjust%20for,approach%20to%20fixing%20outlier%20problems.\&text=Often%20applied%20in%20algorithms%20like,be%20on%20a%20consistent%20scale.](https://www.datacamp.com/tutorial/normalization-vs-standardization#:~:text=Normalization%20can%20help%20adjust%20for,approach%20to%20fixing%20outlier%20problems.&text=Often%20applied%20in%20algorithms%20like,be%20on%20a%20consistent%20scale.)  
14. One Hot Encoding vs Label Encoding \- GeeksforGeeks, truy cập vào tháng 8 18, 2025, [https://www.geeksforgeeks.org/machine-learning/one-hot-encoding-vs-label-encoding/](https://www.geeksforgeeks.org/machine-learning/one-hot-encoding-vs-label-encoding/)  
15. One Hot Encoding vs Label Encoding \- GeeksforGeeks, truy cập vào tháng 8 18, 2025, [https://www.geeksforGeeks.org/machine-learning/one-hot-encoding-vs-label-encoding/](https://www.geeksforGeeks.org/machine-learning/one-hot-encoding-vs-label-encoding/)  
16. Stratified K Fold Cross Validation \- GeeksforGeeks, truy cập vào tháng 8 18, 2025, [https://www.geeksforgeeks.org/machine-learning/stratified-k-fold-cross-validation/](https://www.geeksforgeeks.org/machine-learning/stratified-k-fold-cross-validation/)  
17. Linear Regression \-Pros & Cons. Linear Regression is a statistical ..., truy cập vào tháng 8 18, 2025, [https://medium.com/@satyavishnumolakala/linear-regression-pros-cons-62085314aef0](https://medium.com/@satyavishnumolakala/linear-regression-pros-cons-62085314aef0)  
18. Comparative Study Id3, Cart And C4.5 Decision Tree Algorithm: A Survey \- IJAIST, truy cập vào tháng 8 18, 2025, [https://www.ijaist.com/wp-content/uploads/2018/08/ComparativeStudyId3CartAndC4.5DecisionTreeAlgorithmASurvey.pdf](https://www.ijaist.com/wp-content/uploads/2018/08/ComparativeStudyId3CartAndC4.5DecisionTreeAlgorithmASurvey.pdf)  
19. Decision Tree Algorithms \- GeeksforGeeks, truy cập vào tháng 8 18, 2025, [https://www.geeksforgeeks.org/machine-learning/decision-tree-algorithms/](https://www.geeksforgeeks.org/machine-learning/decision-tree-algorithms/)  
20. Support Vector Machine (SVM) Algorithm \- GeeksforGeeks, truy cập vào tháng 8 18, 2025, [https://www.geeksforgeeks.org/machine-learning/support-vector-machine-algorithm/](https://www.geeksforgeeks.org/machine-learning/support-vector-machine-algorithm/)  
21. What is XGBoost? | IBM, truy cập vào tháng 8 18, 2025, [https://www.ibm.com/think/topics/xgboost](https://www.ibm.com/think/topics/xgboost)  
22. What Is XGBoost and Why Does It Matter? | NVIDIA Glossary, truy cập vào tháng 8 18, 2025, [https://www.nvidia.com/en-us/glossary/xgboost/](https://www.nvidia.com/en-us/glossary/xgboost/)  
23. K-means clustering in Python \- Domino Data Lab, truy cập vào tháng 8 18, 2025, [https://domino.ai/blog/getting-started-with-k-means-clustering-in-python](https://domino.ai/blog/getting-started-with-k-means-clustering-in-python)  
24. What is k-means clustering? \- IBM, truy cập vào tháng 8 18, 2025, [https://www.ibm.com/think/topics/k-means-clustering](https://www.ibm.com/think/topics/k-means-clustering)  
25. Comparing DBSCAN, k-means, and Hierarchical Clustering: When ..., truy cập vào tháng 8 18, 2025, [https://hex.tech/blog/comparing-density-based-methods/](https://hex.tech/blog/comparing-density-based-methods/)  
26. (PDF) Performance Comparison of K-Means and DBScan Algorithms for Text Clustering Product Reviews \- ResearchGate, truy cập vào tháng 8 18, 2025, [https://www.researchgate.net/publication/362683121\_Performance\_Comparison\_of\_K-Means\_and\_DBScan\_Algorithms\_for\_Text\_Clustering\_Product\_Reviews](https://www.researchgate.net/publication/362683121_Performance_Comparison_of_K-Means_and_DBScan_Algorithms_for_Text_Clustering_Product_Reviews)  
27. Tanh vs. Sigmoid vs. ReLU \- GeeksforGeeks, truy cập vào tháng 8 18, 2025, [https://www.geeksforgeeks.org/deep-learning/tanh-vs-sigmoid-vs-relu/](https://www.geeksforgeeks.org/deep-learning/tanh-vs-sigmoid-vs-relu/)  
28. Neural networks: Activation functions | Machine Learning \- Google for Developers, truy cập vào tháng 8 18, 2025, [https://developers.google.com/machine-learning/crash-course/neural-networks/activation-functions](https://developers.google.com/machine-learning/crash-course/neural-networks/activation-functions)  
29. Backpropagation Step by Step \- HMKCODE, truy cập vào tháng 8 18, 2025, [https://hmkcode.com/ai/backpropagation-step-by-step/](https://hmkcode.com/ai/backpropagation-step-by-step/)  
30. Mastering Backpropagation: A Comprehensive Guide for Neural Networks \- DataCamp, truy cập vào tháng 8 18, 2025, [https://www.datacamp.com/tutorial/mastering-backpropagation](https://www.datacamp.com/tutorial/mastering-backpropagation)  
31. Dilution (neural networks) \- Wikipedia, truy cập vào tháng 8 18, 2025, [https://en.wikipedia.org/wiki/Dilution\_(neural\_networks)](https://en.wikipedia.org/wiki/Dilution_\(neural_networks\))  
32. Dropout in neural networks: what it is and how it works : r/learnmachinelearning \- Reddit, truy cập vào tháng 8 18, 2025, [https://www.reddit.com/r/learnmachinelearning/comments/x89qsi/dropout\_in\_neural\_networks\_what\_it\_is\_and\_how\_it/](https://www.reddit.com/r/learnmachinelearning/comments/x89qsi/dropout_in_neural_networks_what_it_is_and_how_it/)  
33. Dropout Regularization in Deep Learning | DigitalOcean, truy cập vào tháng 8 18, 2025, [https://www.digitalocean.com/community/tutorials/droput-regularization-deep-learning](https://www.digitalocean.com/community/tutorials/droput-regularization-deep-learning)  
34. What is Bias-Variance Tradeoff? | IBM, truy cập vào tháng 8 18, 2025, [https://www.ibm.com/think/topics/bias-variance-tradeoff](https://www.ibm.com/think/topics/bias-variance-tradeoff)  
35. Bias–variance tradeoff \- Wikipedia, truy cập vào tháng 8 18, 2025, [https://en.wikipedia.org/wiki/Bias%E2%80%93variance\_tradeoff](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)  
36. LIME vs SHAP: A Comparative Analysis of Interpretability Tools, truy cập vào tháng 8 18, 2025, [https://www.markovml.com/blog/lime-vs-shap](https://www.markovml.com/blog/lime-vs-shap)  
37. SMOTE: A Powerful Technique for Handling Imbalanced Data \- Medium, truy cập vào tháng 8 18, 2025, [https://medium.com/@thecontentfarmblog/smote-a-powerful-technique-for-handling-imbalanced-data-2375ad46103c](https://medium.com/@thecontentfarmblog/smote-a-powerful-technique-for-handling-imbalanced-data-2375ad46103c)  
38. Smote for Imbalanced Classification with Python, Technique, truy cập vào tháng 8 18, 2025, [https://www.analyticsvidhya.com/blog/2020/10/overcoming-class-imbalance-using-smote-techniques/](https://www.analyticsvidhya.com/blog/2020/10/overcoming-class-imbalance-using-smote-techniques/)  
39. Reinforcement Learning Explained Visually \- Q Learning, step-by-step | Ketan Doshi Blog, truy cập vào tháng 8 18, 2025, [https://ketanhdoshi.github.io/Reinforcement-Learning-Q-Learning/](https://ketanhdoshi.github.io/Reinforcement-Learning-Q-Learning/)  
40. What is a GAN? \- Generative Adversarial Networks Explained \- AWS, truy cập vào tháng 8 18, 2025, [https://aws.amazon.com/what-is/gan/](https://aws.amazon.com/what-is/gan/)  
41. Overview of GAN Structure | Machine Learning \- Google for Developers, truy cập vào tháng 8 18, 2025, [https://developers.google.com/machine-learning/gan/gan\_structure](https://developers.google.com/machine-learning/gan/gan_structure)  
42. A Gentle Introduction to Generative Adversarial Networks (GANs) \- MachineLearningMastery.com, truy cập vào tháng 8 18, 2025, [https://machinelearningmastery.com/what-are-generative-adversarial-networks-gans/](https://machinelearningmastery.com/what-are-generative-adversarial-networks-gans/)  
43. A Gentle Introduction to Transfer Learning with BERT and ResNet ..., truy cập vào tháng 8 18, 2025, [https://medium.com/@gaurav.duseja211/a-gentle-introduction-to-transfer-learning-with-bert-and-resnet-50-7c15576b436c](https://medium.com/@gaurav.duseja211/a-gentle-introduction-to-transfer-learning-with-bert-and-resnet-50-7c15576b436c)  
44. Transfer Learning: Leveraging Pretrained Models | by Jim Canary | Medium, truy cập vào tháng 8 18, 2025, [https://medium.com/@jimcanary/transfer-learning-leveraging-pretrained-models-153ab99b9b00](https://medium.com/@jimcanary/transfer-learning-leveraging-pretrained-models-153ab99b9b00)  
45. Netflix Content Recommendation System – Product Analytics Case Study \- HelloPM, truy cập vào tháng 8 18, 2025, [https://hellopm.co/netflix-content-recommendation-system-product-analytics-case-study/](https://hellopm.co/netflix-content-recommendation-system-product-analytics-case-study/)  
46. Netflix's Recommendation Systems: Entertainment Made for You, truy cập vào tháng 8 18, 2025, [https://illumin.usc.edu/netflixs-recommendation-systems-entertainment-made-for-you/](https://illumin.usc.edu/netflixs-recommendation-systems-entertainment-made-for-you/)  
47. System Design Netflix | A Complete Architecture \- GeeksforGeeks, truy cập vào tháng 8 18, 2025, [https://www.geeksforgeeks.org/system-design/system-design-netflix-a-complete-architecture/](https://www.geeksforgeeks.org/system-design/system-design-netflix-a-complete-architecture/)  
48. A Deep Dive Into Recommendation Algorithms With Netflix Case Study and NVIDIA Deep Learning Technology \- DZone, truy cập vào tháng 8 18, 2025, [https://dzone.com/articles/a-deep-dive-into-recommendation-algorithms-with-ne](https://dzone.com/articles/a-deep-dive-into-recommendation-algorithms-with-ne)  
49. Foundation Model for Personalized Recommendation | by Netflix Technology Blog, truy cập vào tháng 8 18, 2025, [https://netflixtechblog.com/foundation-model-for-personalized-recommendation-1a0bd8e02d39](https://netflixtechblog.com/foundation-model-for-personalized-recommendation-1a0bd8e02d39)  
50. Blueprints for recommender system architectures: 10th anniversary edition, truy cập vào tháng 8 18, 2025, [http://amatria.in/blog/RecsysArchitectures](http://amatria.in/blog/RecsysArchitectures)  
51. AI Machine Learning Aids Fraud Detection | Cognizant, truy cập vào tháng 8 18, 2025, [https://www.cognizant.com/us/en/case-studies/ai-machine-learning-fraud-detection](https://www.cognizant.com/us/en/case-studies/ai-machine-learning-fraud-detection)  
52. Car Object Detection: How Deep Learning Powers Self-Driving Cars, truy cập vào tháng 8 18, 2025, [https://www.labellerr.com/blog/how-object-detection-works-in-self-driving-cars-using-deep-learning/](https://www.labellerr.com/blog/how-object-detection-works-in-self-driving-cars-using-deep-learning/)  
53. How Self-Driving Cars Learn to See (Part 3): Eyes on the Road with Convolutional Networks, truy cập vào tháng 8 18, 2025, [https://medium.com/@nikhilnair8490/how-self-driving-cars-learn-to-see-part-3-eyes-on-the-road-with-convolutional-networks-d5f8bcac980f](https://medium.com/@nikhilnair8490/how-self-driving-cars-learn-to-see-part-3-eyes-on-the-road-with-convolutional-networks-d5f8bcac980f)  
54. Enhancing Object Detection in Self-Driving Cars Using a Hybrid Approach \- MDPI, truy cập vào tháng 8 18, 2025, [https://www.mdpi.com/2079-9292/12/13/2768](https://www.mdpi.com/2079-9292/12/13/2768)  
55. Object Detection for Autonomous Driving using YOLO algorithm \- ResearchGate, truy cập vào tháng 8 18, 2025, [https://www.researchgate.net/publication/352137641\_Object\_Detection\_for\_Autonomous\_Driving\_using\_YOLO\_algorithm](https://www.researchgate.net/publication/352137641_Object_Detection_for_Autonomous_Driving_using_YOLO_algorithm)  
56. Enhancing Object Detection Accuracy in Autonomous Vehicles Using Synthetic Data \- arXiv, truy cập vào tháng 8 18, 2025, [https://arxiv.org/html/2411.15602v1](https://arxiv.org/html/2411.15602v1)