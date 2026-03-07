## **Core topics in Machine Learning (ML) fundamentals**
---

## 1. **Introduction to Machine Learning**

* Definition and types of ML (Supervised, Unsupervised, Reinforcement Learning)
* Differences between AI, ML, and Deep Learning
* Typical ML workflow: Data → Model → Evaluation → Deployment

---

## 2. **Mathematical Foundations**

* **Linear Algebra:** Vectors, matrices, dot products, eigenvalues/eigenvectors
* **Calculus:** Derivatives, gradients, partial derivatives (for optimization)
* **Probability & Statistics:** Bayes’ theorem, distributions, expectation/variance
* **Optimization:** Gradient descent, stochastic gradient descent (SGD), convex optimization

---

## 3. **Data Handling**

* Data preprocessing (cleaning, normalization, scaling)
* Feature engineering (encoding categorical data, feature selection, dimensionality reduction)
* Handling missing values and outliers
* Train/test/validation splits and cross-validation

---

## 4. **Supervised Learning**

* **Regression:** Linear regression, logistic regression
* **Classification:** KNN, Decision Trees, Random Forest, SVM, Naïve Bayes
* **Ensemble Methods:** Bagging, Boosting (XGBoost, LightGBM), Stacking
* Evaluation metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC, MSE, RMSE

---

## 5. **Unsupervised Learning**

* **Clustering:** K-Means, Hierarchical, DBSCAN
* **Dimensionality Reduction:** PCA, t-SNE, Autoencoders
* Association rule mining (Apriori, FP-Growth)

---

## 6. **Neural Networks & Deep Learning (Basics)**

* Perceptron and multilayer perceptrons (MLPs)
* Activation functions (ReLU, Sigmoid, Tanh, Softmax)
* Backpropagation & gradient descent
* Overfitting and regularization (Dropout, L2/L1)

---

## 7. **Model Evaluation & Validation**

* Bias-variance tradeoff
* Overfitting vs underfitting
* Cross-validation strategies (k-fold, stratified, leave-one-out)
* Hyperparameter tuning (Grid search, Random search, Bayesian optimization)

---

## 8. **Practical Aspects**

* Model interpretability (SHAP, LIME, feature importance)
* Handling imbalanced data (SMOTE, class weights, undersampling/oversampling)
* Scalability and efficiency (batching, parallelization, distributed ML)

---

## 9. **Advanced Topics (Fundamental Awareness)**

* Reinforcement Learning (Q-learning, Policy Gradients)
* Generative models (GANs, VAEs)
* Transfer Learning & Pretrained Models
* Online learning and streaming data

---

✅ **In short:** ML fundamentals span **mathematics, data, algorithms, evaluation, and practical deployment issues**. Mastering these core topics will give you the foundation to dive deeper into specialized areas like NLP, Computer Vision, and Reinforcement Learning.

---

# 🔹 **1. Introduction to Machine Learning**

## **Supervised Learning**

* **Ý nghĩa:** Học từ dữ liệu có nhãn (input → output).
* **Khi dùng:** Dự đoán giá nhà (regression), phân loại email spam (classification).
* **Ví dụ thực tế:**

  * Y tế: dự đoán bệnh dựa trên kết quả xét nghiệm
  * Tài chính: phát hiện gian lận thẻ tín dụng
  * Marketing: dự đoán churn (khách sắp rời bỏ)

**Mục tiêu:** Tối ưu dự đoán **chính xác trên dữ liệu mới**.

---

## **Unsupervised Learning**

* **Ý nghĩa:** Không có nhãn, tìm cấu trúc ẩn trong dữ liệu.
* **Khi dùng:** Nhóm khách hàng (clustering), giảm chiều dữ liệu (PCA).
* **Ví dụ thực tế:**

  * E-commerce: gợi ý sản phẩm dựa trên hành vi người dùng giống nhau
  * Sinh học: phân nhóm gene có đặc tính tương đồng

**Mục tiêu:** **Khám phá dữ liệu** khi chưa biết rõ mục tiêu.

---

## **Reinforcement Learning**

* **Ý nghĩa:** Agent học cách hành động thông qua phần thưởng/phạt.
* **Khi dùng:** Game AI, robotics, hệ thống recommend liên tục.
* **Ví dụ thực tế:**

  * AlphaGo đánh cờ vây
  * Xe tự lái học cách lái an toàn

**Mục tiêu:** Tối đa hóa **tích lũy phần thưởng dài hạn**.

---

### **So sánh tổng quan:**

| Loại              | Input                 | Output                                   | Ứng dụng chính       | Metric thường dùng  |
| ----------------- | --------------------- | ---------------------------------------- | -------------------- | ------------------- |
| **Supervised**    | Có nhãn               | Giá trị dự đoán (continuous/categorical) | Dự đoán, phân loại   | Accuracy, F1, MSE   |
| **Unsupervised**  | Không nhãn            | Nhóm, cấu trúc ẩn                        | Khám phá, gợi ý      | Silhouette, inertia |
| **Reinforcement** | Trạng thái, hành động | Chính sách hành động                     | Game, control system | Reward, regret      |

---

# 🔹 **2. Mathematical Foundations**

## **Linear Algebra**

* **Vai trò:** Dữ liệu ML thường ở dạng vector/matrix → cần hiểu để xử lý.
* **Khái niệm quan trọng:**

  * Vector, matrix
  * Dot product, norm (độ dài vector)
  * Eigenvalue, eigenvector → dùng trong PCA

**Ví dụ thực tế:**

* Computer Vision: ảnh là ma trận pixel
* NLP: word embeddings là vector

**Tips:** Học **numpy**/R matrix manipulation sẽ thấy trực quan.

---

## **Calculus**

* **Vai trò:** Học máy = Tối ưu hàm loss → cần đạo hàm, gradient.
* **Khái niệm quan trọng:**

  * Gradient: vector hướng tăng nhanh nhất
  * Partial derivatives: tính theo từng biến
  * Chain rule: tính backpropagation trong neural network

**Ví dụ thực tế:**

* Deep Learning: backpropagation chính là áp dụng chain rule
* Logistic Regression: dùng gradient descent để tối ưu tham số

---

## **Probability & Statistics**

* **Vai trò:** Mọi dự đoán đều mang tính xác suất.
* **Khái niệm quan trọng:**

  * Bayes’ theorem → nền tảng của Naïve Bayes classifier
  * Expectation, variance → đánh giá dữ liệu
  * Distributions: Gaussian, Bernoulli, Poisson

**Ví dụ thực tế:**

* Spam detection: xác suất email là spam dựa trên từ khóa
* A/B testing: kiểm định giả thuyết (hypothesis testing)

---

## **Optimization**

* **Vai trò:** Cốt lõi của việc training model.
* **Kỹ thuật phổ biến:**

  * Gradient Descent, SGD (stochastic)
  * Momentum, Adam optimizer
  * Convex optimization (bài toán dễ giải hơn)

**Ví dụ thực tế:**

* Deep learning training → tối ưu loss function
* Linear regression → tìm hệ số bằng cách minimize MSE

---

### **Tóm tắt quyết định học toán trong ML:**

| Thành phần     | Tại sao quan trọng              | Liên quan đến ML                |
| -------------- | ------------------------------- | ------------------------------- |
| Linear Algebra | Biểu diễn dữ liệu & model       | Embedding, PCA, CNN             |
| Calculus       | Tối ưu hóa loss                 | Backpropagation                 |
| Probability    | Ra quyết định trong uncertainty | Naïve Bayes, Bayesian inference |
| Optimization   | Training hiệu quả               | Gradient descent, Adam          |

---
# 🔹 **3. Data Handling in ML**

Trước khi train model, **data preparation chiếm 70–80% công việc ML**. Nếu data bẩn → model sai lệch.

---

## **3.1 Data Preprocessing**

### **Scaling & Normalization**

* **Khi cần:** Khi dữ liệu có giá trị khác nhau quá lớn.
* **Kỹ thuật:**

  * **Standardization (Z-score):** đưa về mean = 0, std = 1.
  * **Min-Max Scaling:** scale về \[0,1].
  * **Log Transform:** xử lý phân phối lệch (skewed).

**Ví dụ:**

* Gradient Descent hội tụ nhanh hơn nếu features có scale tương đồng.
* KNN, SVM rất nhạy cảm với khoảng cách → cần scaling.

---

### **Encoding Categorical Data**

* **One-Hot Encoding:** chuyển category thành vector 0/1.
* **Label Encoding:** gán số thứ tự cho category.
* **Target Encoding:** thay bằng trung bình target (dùng cẩn thận).

**Ví dụ:**

* "Country" → \["VN", "US", "JP"] → one-hot thành \[1,0,0], \[0,1,0], \[0,0,1].
* Movie genre encoding cho recommendation system.

---

### **Handling Missing Values**

* **Options:**

  * Bỏ hàng/cột (nếu ít, không quan trọng).
  * Điền bằng mean/median/mode.
  * Predictive imputation (dùng model để dự đoán).

**Ví dụ:**

* Y tế: thiếu chỉ số xét nghiệm → dùng median.
* Finance: điền missing stock price bằng interpolation.

---

### **Handling Outliers**

* **Options:**

  * IQR method (Q1 - 1.5IQR, Q3 + 1.5IQR).
  * Z-score > 3 coi là outlier.
  * Winsorization (cắt về ngưỡng).

**Ví dụ:**

* Credit card fraud: giao dịch 1 tỷ trong lịch sử vài trăm nghìn → outlier thật.
* Housing price: nhà luxury cao bất thường → có thể giữ lại.

---

## **3.2 Feature Engineering**

### **Feature Creation**

* Tạo feature mới từ feature có sẵn.
* **Ví dụ:**

  * Từ "date" → tạo "day\_of\_week", "is\_weekend".
  * Từ "total\_price" và "quantity" → tạo "unit\_price".

---

### **Feature Selection**

* **Filter methods:** dựa trên thống kê (chi-square, correlation).
* **Wrapper methods:** chọn feature dựa trên model (RFE, forward selection).
* **Embedded methods:** feature importance từ model (Lasso, Random Forest).

**Ví dụ:**

* NLP: chọn top 1000 words quan trọng thay vì full vocab.
* Genomics: chọn gene quan trọng để dự đoán bệnh.

---

### **Dimensionality Reduction**

* **PCA (Principal Component Analysis):** tìm trục chính giảm chiều.
* **t-SNE / UMAP:** giảm chiều để visualize.
* **Autoencoder:** neural net để compress dữ liệu.

**Ví dụ:**

* Hình ảnh 1000x1000 pixel → PCA giảm còn vài trăm chiều.
* Customer segmentation visualization.

---

## **3.3 Data Splitting**

### **Train/Test/Validation**

* **Train set:** học tham số.
* **Validation set:** chọn hyperparameters.
* **Test set:** kiểm tra cuối cùng.

### **Cross-Validation**

* **K-fold CV:** chia thành k tập, xoay vòng.
* **Stratified CV:** giữ tỉ lệ class cân bằng.
* **Leave-One-Out (LOO):** extreme CV.

**Ví dụ:**

* Kaggle competitions: dùng stratified k-fold CV để đánh giá ổn định.
* NLP: LOO dùng khi dataset cực nhỏ.

---

### **Tóm tắt quyết định xử lý data:**

| Vấn đề                  | Kỹ thuật                        | Khi nào dùng                                                       |
| ----------------------- | ------------------------------- | ------------------------------------------------------------------ |
| Feature scale khác biệt | Standardization, Min-Max        | Model dựa trên khoảng cách, gradient descent                       |
| Categorical data        | One-hot, Label, Target encoding | One-hot cho dữ liệu ít class, Target encoding cho high cardinality |
| Missing values          | Mean/Median, Predictive         | Khi dữ liệu không hoàn chỉnh                                       |
| Outliers                | IQR, Z-score, Winsorization     | Tuỳ bối cảnh: gian lận giữ lại, noise thì loại                     |
| Quá nhiều features      | PCA, Feature selection          | Tránh curse of dimensionality                                      |
| Data split              | Train/Val/Test, CV              | Đảm bảo model generalize tốt                                       |

---

# 🔹 **4. Supervised Learning**

Supervised learning = học từ dữ liệu có nhãn (input → output).
Chia thành **Regression** (output liên tục) và **Classification** (output rời rạc).

---

## **4.1 Regression (Dự đoán giá trị liên tục)**

### **Linear Regression**

* **Ý nghĩa:** Tìm đường thẳng/hyperplane fit dữ liệu.
* **Loss:** Mean Squared Error (MSE).
* **Ưu điểm:** Đơn giản, dễ giải thích.
* **Nhược điểm:** Giả định tuyến tính, nhạy cảm với outliers.

**Ví dụ:**

* Dự đoán giá nhà từ diện tích.
* Dự đoán lượng mưa từ nhiệt độ & độ ẩm.

---

### **Polynomial Regression**

* **Ý nghĩa:** Mở rộng linear bằng polynomial features (x², x³,...).
* **Ưu điểm:** Fit được quan hệ phi tuyến.
* **Nhược điểm:** Dễ overfit.

**Ví dụ:**

* Growth curve trong sinh học.
* Quan hệ phức tạp giữa giá cổ phiếu & thời gian.

---

### **Regularized Regression**

* **Ridge (L2):** giảm magnitude coefficients → hạn chế overfitting.
* **Lasso (L1):** ép nhiều coefficients về 0 → feature selection.
* **ElasticNet:** kết hợp Ridge + Lasso.

**Ví dụ:**

* Dự đoán giá cổ phiếu với hàng ngàn features.
* Y tế: chọn chỉ số máu quan trọng để dự đoán bệnh.

---

### **Khi nào dùng Regression?**

* Output là số thực liên tục (doanh thu, giá, thời gian).
* Khi cần dự đoán *how much / how many*.

---

## **4.2 Classification (Dự đoán nhãn rời rạc)**

### **Logistic Regression**

* **Ý nghĩa:** Ước lượng xác suất P(y=1|x).
* **Output:** Sigmoid → \[0,1].
* **Ví dụ:** Spam detection, bệnh có/không.

---

### **Decision Trees**

* **Ý nghĩa:** Chia nhánh dữ liệu theo feature threshold.
* **Ưu điểm:** Dễ hiểu, visualize.
* **Nhược điểm:** Dễ overfit, không ổn định.
* **Ví dụ:** Approve loan, churn prediction.

---

### **Random Forest**

* **Ý nghĩa:** Tập hợp nhiều cây → giảm variance.
* **Ưu điểm:** Robust, ít overfit.
* **Nhược điểm:** Mất tính dễ giải thích.
* **Ví dụ:** Fraud detection, credit scoring.

---

### **Support Vector Machines (SVM)**

* **Ý nghĩa:** Tìm hyperplane tối ưu phân tách class.
* **Ưu điểm:** Hiệu quả với dữ liệu không tuyến tính (kernel trick).
* **Nhược điểm:** Tốn tài nguyên, khó scale big data.
* **Ví dụ:** Phân loại văn bản, nhận diện khuôn mặt.

---

### **K-Nearest Neighbors (KNN)**

* **Ý nghĩa:** Classify dựa trên majority vote của k neighbors.
* **Ưu điểm:** Đơn giản, trực quan.
* **Nhược điểm:** Chậm khi dataset lớn.
* **Ví dụ:** Recommendation, phân loại ảnh.

---

### **Naïve Bayes**

* **Ý nghĩa:** Áp dụng Bayes theorem với assumption độc lập.
* **Ưu điểm:** Nhanh, hiệu quả cho text classification.
* **Nhược điểm:** Assumption độc lập hiếm khi đúng.
* **Ví dụ:** Sentiment analysis, spam filtering.

---

### **Neural Networks (cơ bản)**

* **Ý nghĩa:** Layers of perceptrons học mapping phức tạp.
* **Ưu điểm:** Capture non-linear patterns.
* **Nhược điểm:** Tốn compute, dễ overfit nếu data nhỏ.
* **Ví dụ:** Image classification, NLP tasks.

---

### **Khi nào dùng Classification?**

* Output là discrete categories (yes/no, loại sản phẩm).
* Khi cần trả lời *which class does it belong to?*.

---

## **4.3 Ensemble Methods**

### **Bagging (Bootstrap Aggregating)**

* Train nhiều models trên bootstrap samples, vote trung bình.
* **Ví dụ:** Random Forest.

---

### **Boosting**

* Train models sequentially, mỗi model tập trung sửa lỗi model trước.
* **Ví dụ:** AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost.
* **Ứng dụng:** Fraud detection, Kaggle competitions.

---

### **Stacking**

* Combine predictions của nhiều base models bằng meta-model.
* **Ví dụ:** Dùng Logistic Regression trên output của RF + XGB.
* **Ứng dụng:** NLP competitions, production ML.

---

## **4.4 Evaluation Metrics**

### **Regression Metrics**

* **MSE (Mean Squared Error):** Nhạy với outliers.
* **MAE (Mean Absolute Error):** Robust hơn, dễ hiểu.
* **R² Score:** Tỉ lệ variance giải thích được.

---

### **Classification Metrics**

* **Accuracy:** dễ hiểu, nhưng misleading nếu dataset imbalance.
* **Precision / Recall / F1:** dùng tuỳ business case (như phần bạn ví dụ).
* **ROC-AUC:** đo khả năng phân biệt class.
* **Confusion Matrix:** breakdown TP, FP, FN, TN.

---

### **Metric chọn theo ngữ cảnh:**

| Tình huống           | Metric ưu tiên      | Lý do                       |
| -------------------- | ------------------- | --------------------------- |
| Dataset balance      | Accuracy            | Tổng quan dễ hiểu           |
| Dataset imbalance    | Precision/Recall/F1 | Accuracy dễ đánh lừa        |
| Y tế, bảo mật        | Recall              | FN nguy hiểm                |
| Spam, Ads, Recommend | Precision           | FP gây phiền                |
| Tìm kiếm, IR         | F1                  | Cần trade-off               |
| Ranking problems     | ROC-AUC             | Đánh giá khả năng phân biệt |

---

✅ **Kết luận Supervised Learning:**

* Regression → khi output là **continuous**.
* Classification → khi output là **categorical**.
* Ensemble → khi cần **boost performance**.
* Evaluation metric chọn theo **business impact của lỗi**.

---

 **Clustering → Dimensionality Reduction → Association Rules → Evaluation**.

---

# 🔹 **5. Unsupervised Learning**

Unsupervised learning = học từ **dữ liệu không có nhãn**.
Mục tiêu: **khám phá cấu trúc ẩn** hoặc **giảm phức tạp** dữ liệu.

---

## **5.1 Clustering (Phân nhóm dữ liệu)**

### **K-Means**

* **Cách hoạt động:** chọn K cluster, lặp lại: gán điểm vào cluster gần nhất → cập nhật centroid.
* **Ưu điểm:** Nhanh, dễ hiểu.
* **Nhược điểm:** Cần chọn K, nhạy cảm outliers.

**Ví dụ:**

* Marketing: phân nhóm khách hàng (VIP, bình thường).
* Image compression: gom pixel vào nhóm màu.

---

### **Hierarchical Clustering**

* **Cách hoạt động:** gom nhóm dần (agglomerative) hoặc tách dần (divisive).
* **Ưu điểm:** Không cần chọn K trước.
* **Nhược điểm:** Chậm với dữ liệu lớn.

**Ví dụ:**

* Sinh học: phân loại cây, gene, loài.
* NLP: nhóm tài liệu theo chủ đề.

---

### **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**

* **Cách hoạt động:** gom cụm theo mật độ điểm.
* **Ưu điểm:** Không cần K, nhận diện noise/outliers.
* **Nhược điểm:** Nhạy với tham số (eps, minPts).

**Ví dụ:**

* Phát hiện vùng địa chấn từ dữ liệu GPS.
* Anomaly detection trong dữ liệu giao dịch.

---

### **So sánh các phương pháp clustering:**

| Thuật toán   | Khi nào dùng                              | Ưu điểm                  | Nhược điểm        |
| ------------ | ----------------------------------------- | ------------------------ | ----------------- |
| K-Means      | Data lớn, phân bố rõ ràng                 | Nhanh, dễ hiểu           | Cần K, nhạy noise |
| Hierarchical | Khi muốn quan sát dendrogram              | Không cần K trước        | Chậm              |
| DBSCAN       | Khi data có noise, hình dạng cụm phức tạp | Tìm cụm bất kỳ hình dạng | Nhạy tham số      |

---

## **5.2 Dimensionality Reduction**

### **PCA (Principal Component Analysis)**

* **Ý nghĩa:** Tìm trục mới (principal components) tối đa hóa variance.
* **Ưu điểm:** Giảm chiều hiệu quả, dễ implement.
* **Nhược điểm:** Tuyến tính, khó giải thích.

**Ví dụ:**

* Ảnh 1000x1000 pixel → PCA giảm còn 50 chiều.
* Finance: giảm số lượng chỉ số kinh tế để phân tích.

---

### **t-SNE / UMAP**

* **Ý nghĩa:** non-linear dimensionality reduction → visualization.
* **Ưu điểm:** Hiển thị dữ liệu high-dim thành 2D/3D rõ ràng.
* **Nhược điểm:** tốn compute, không tốt cho prediction.

**Ví dụ:**

* NLP: visualize word embeddings.
* Bioinformatics: visualize gene expression data.

---

### **Autoencoders**

* **Ý nghĩa:** Neural net học cách nén & giải nén dữ liệu.
* **Ưu điểm:** Giảm chiều, phát hiện anomaly.
* **Nhược điểm:** Cần nhiều data, training phức tạp.

**Ví dụ:**

* Anomaly detection trong IoT.
* Denoising ảnh.

---

## **5.3 Association Rule Learning**

### **Apriori Algorithm**

* **Ý nghĩa:** tìm ra luật “nếu A → thì B” trong dữ liệu.
* **Metric:** Support, Confidence, Lift.
* **Ví dụ:** Market basket analysis: “mua bia → thường mua snack”.

---

### **FP-Growth**

* **Ý nghĩa:** Giống Apriori nhưng tối ưu hơn, dùng FP-tree.
* **Ưu điểm:** Nhanh hơn Apriori với dữ liệu lớn.
* **Ví dụ:** Recommendation systems cho e-commerce.

---

## **5.4 Evaluation Metrics for Unsupervised Learning**

⚠️ Vì không có nhãn → evaluation khó hơn supervised.

### **Clustering Metrics**

* **Internal (không cần nhãn):**

  * Silhouette Score (–1 đến 1): đo độ gọn và tách biệt cụm.
  * Davies-Bouldin Index: càng thấp càng tốt.
* **External (có nhãn ground truth):**

  * Adjusted Rand Index (ARI).
  * Normalized Mutual Information (NMI).

**Ví dụ chọn metric:**

* Customer segmentation → dùng Silhouette.
* So sánh với cluster thật (chẩn đoán y tế) → dùng ARI/NMI.

---

### **Dimensionality Reduction Metrics**

* **Explained Variance (PCA):** bao nhiêu % variance được giữ lại.
* **Reconstruction Error (Autoencoder):** dữ liệu nén có khôi phục tốt không.

---

### **Association Rule Metrics**

* **Support:** % transactions chứa {A,B}.
* **Confidence:** Xác suất có B khi có A.
* **Lift:** mức độ mạnh của quan hệ so với random.

**Ví dụ:**

* Market basket analysis: Lift > 1 = có ý nghĩa.

---

### **Tóm tắt quyết định Unsupervised Learning:**

| Bài toán               | Kỹ thuật            | Metric quan trọng         |
| ---------------------- | ------------------- | ------------------------- |
| Phân nhóm khách hàng   | K-Means, DBSCAN     | Silhouette, ARI           |
| Visualization high-dim | PCA, t-SNE          | Explained variance        |
| Anomaly detection      | Autoencoder, DBSCAN | Reconstruction error      |
| Market basket analysis | Apriori, FP-Growth  | Support, Confidence, Lift |

---

✅ **Kết luận Unsupervised Learning:**

* Clustering để **tìm nhóm tự nhiên** trong dữ liệu.
* Dimensionality reduction để **giảm chiều / visualize**.
* Association rules để **khai thác mối quan hệ ẩn**.
* Evaluation chọn theo **ngữ cảnh + mục tiêu**.

---

# 🔹 **6. Neural Networks & Deep Learning (Basics)**

Neural networks = tập hợp các **“neuron nhân tạo”** mô phỏng não bộ.
Deep learning = neural networks nhiều tầng, học representation phức tạp.

---

## **6.1 Perceptron & Multilayer Perceptron (MLP)**

### **Perceptron**

* **Input:** vector $x$.
* **Weights:** vector $w$.
* **Output:** $y = f(w \cdot x + b)$.
* **f:** activation function (bật/tắt neuron).

**Giới hạn:** Chỉ giải quyết được bài toán **linear separable** (vd: AND/OR nhưng không phải XOR).

---

### **Multilayer Perceptron (MLP)**

* Thêm **hidden layers** để học quan hệ phi tuyến.
* Dùng backpropagation để train.

**Ví dụ:**

* Input: ảnh 28x28 (MNIST).
* Hidden: nhiều layer fully-connected.
* Output: class (0–9).

---

## **6.2 Activation Functions**

🔑 Activation giúp mạng học **non-linear patterns**.

| Activation     | Công thức                         | Khi dùng                          | Ưu / Nhược điểm                                  |
| -------------- | --------------------------------- | --------------------------------- | ------------------------------------------------ |
| **Sigmoid**    | $\sigma(x) = 1 / (1+e^{-x})$      | Output xác suất                   | Dễ hiểu, nhưng vanishing gradient                |
| **Tanh**       | $(e^x - e^{-x}) / (e^x + e^{-x})$ | Hidden layers cũ                  | Better than sigmoid, vẫn vanishing gradient      |
| **ReLU**       | $\max(0, x)$                      | Hidden layers modern              | Nhanh, giảm vanishing gradient, nhưng dying ReLU |
| **Leaky ReLU** | $x$ if $x>0$, else $0.01x$        | Fix dying ReLU                    | Cho gradient nhỏ khi x<0                         |
| **Softmax**    | $e^{x_i}/\sum e^{x_j}$            | Output classification multi-class | Trả phân phối xác suất                           |

**Ví dụ:**

* Spam classification → Sigmoid cho output 0/1.
* ImageNet classification → Softmax cho 1000 classes.

---

## **6.3 Backpropagation & Training**

### **Forward Pass**

* Input → hidden layers → output.
* Tính loss: $L(y_{true}, y_{pred})$.

### **Backward Pass (Backpropagation)**

* Chain rule của đạo hàm để tính gradient từng layer.
* Update weights:

  $$
  w := w - \eta \frac{\partial L}{\partial w}
  $$

  ($\eta$ = learning rate).

### **Optimizers**

* **SGD:** cơ bản.
* **Momentum:** tăng tốc hội tụ.
* **Adam:** adaptive learning rate, phổ biến nhất.

---

## **6.4 Overfitting & Regularization**

⚠️ Neural nets rất dễ **overfit**.

### **Regularization Techniques**

* **Dropout:** random tắt neuron trong training → tránh co-dependency.
* **L1/L2 Regularization:** penalty lên weights.
* **Early stopping:** dừng training khi val\_loss ngừng giảm.
* **Batch Normalization:** chuẩn hoá input layer → hội tụ nhanh, giảm overfit.
* **Data Augmentation:** tăng dữ liệu train (ảnh xoay, lật, dịch).

**Ví dụ:**

* Image classification: dùng augmentation (rotate, crop).
* Text classification: dùng dropout trong RNN/LSTM.

---

## **6.5 Practical Use Cases**

* **Computer Vision (CV):**

  * CNNs → nhận diện ảnh, object detection.
  * Ví dụ: FaceID, tự động gắn tag ảnh.

* **Natural Language Processing (NLP):**

  * RNN/LSTM → chuỗi văn bản.
  * Transformers (sau này) → GPT, BERT.

* **Recommendation Systems:**

  * Deep neural nets để học user–item embeddings.

* **Finance:**

  * Fraud detection với autoencoder (anomaly).
  * Stock price trend prediction.

---

### **Tóm tắt quyết định Neural Networks:**

| Thành phần            | Vai trò             | Khi nào quan trọng                 |
| --------------------- | ------------------- | ---------------------------------- |
| Perceptron            | Building block      | Linear separation                  |
| MLP                   | Học non-linear      | Classification cơ bản              |
| Activation            | Non-linear patterns | Sigmoid/Tanh/Softmax/ReLU          |
| Backprop + Optimizers | Tối ưu loss         | Adam phổ biến nhất                 |
| Regularization        | Chống overfit       | Dropout, BatchNorm, Early stopping |
| Use cases             | Ứng dụng đa dạng    | CV, NLP, Recommendation            |

---

✅ **Kết luận Deep Learning Basics:**

* **Perceptron** → nền tảng.
* **MLP + activation** → học phi tuyến.
* **Backpropagation** → cách học.
* **Regularization** → chống overfit.
* Ứng dụng mạnh trong **CV, NLP, Finance, Recommendation**.

---

# 🔹 **7. Model Evaluation & Validation**

---

## **7.1 Bias–Variance Tradeoff**

### **Bias**

* **Ý nghĩa:** Sai lệch do model **quá đơn giản** (underfitting).
* **Ví dụ:** Linear regression cố fit dữ liệu phi tuyến.
* **Dấu hiệu:** Train error & test error đều cao.

---

### **Variance**

* **Ý nghĩa:** Sai lệch do model **quá phức tạp** (overfitting).
* **Ví dụ:** Decision tree quá sâu học noise thay vì pattern.
* **Dấu hiệu:** Train error thấp nhưng test error cao.

---

### **Tradeoff**

* **Bias cao, Variance thấp → underfit**.
* **Bias thấp, Variance cao → overfit**.
* **Giải pháp:** Regularization, cross-validation, ensemble.

**Minh hoạ thực tế:**

* Bác sĩ chẩn đoán bệnh:

  * **Bias cao:** chỉ xét 1 triệu chứng → bỏ sót bệnh.
  * **Variance cao:** xét quá nhiều yếu tố nhỏ lẻ → kết luận lung tung.

---

## **7.2 Validation Strategies**

### **Hold-out Validation**

* Chia dataset thành **Train / Validation / Test**.
* Đơn giản nhưng phụ thuộc cách chia.

---

### **Cross-Validation (CV)**

* **K-Fold CV:** chia thành k phần, train trên k-1, validate trên phần còn lại, xoay vòng.
* **Stratified K-Fold:** giữ tỉ lệ class.
* **Leave-One-Out (LOO):** extreme CV, dùng khi dataset nhỏ.

**Ví dụ:**

* Kaggle competitions → stratified k-fold CV để giảm variance trong scoring.
* NLP dataset nhỏ → LOO CV để tận dụng data.

---

### **Nested Cross-Validation**

* Dùng cho hyperparameter tuning → vòng ngoài train/test, vòng trong chọn tham số.
* Tránh overfitting khi tune hyperparameters.

---

## **7.3 Hyperparameter Tuning**

### **Grid Search**

* Test mọi combination → exhaustive nhưng tốn tài nguyên.

### **Random Search**

* Chọn random combination → nhanh hơn, hiệu quả với high-dimensional space.

### **Bayesian Optimization**

* Dự đoán vùng promising → thử nghiệm thông minh hơn.

### **Modern Tools**

* **Optuna, Hyperopt, Ray Tune** → tune tự động, distributed.

**Ví dụ:**

* XGBoost: tune `max_depth`, `eta`, `colsample_bytree`.
* Deep Learning: tune learning rate, batch size, dropout.

---

## **7.4 Threshold Tuning**

Nhiều classification model trả về **probability** → cần chọn threshold.

* **Default:** 0.5.
* **Điều chỉnh:**

  * Tăng recall → hạ threshold (nhiều positive hơn).
  * Tăng precision → tăng threshold (ít positive hơn).

**Ví dụ:**

* Medical diagnosis (ung thư): threshold thấp để không bỏ sót.
* Email spam: threshold cao để tránh lọc nhầm email quan trọng.

---

## **7.5 Model Selection in Practice**

* Không có **1 model tốt nhất** → phụ thuộc **data + business goal**.
* Chọn model dựa trên:

  * Metric quan trọng (Accuracy vs Recall vs Precision).
  * Data size (small → simpler model, big → deep learning).
  * Compute resources.

**Ví dụ:**

* Banking fraud detection: chọn Recall > Precision.
* Recommendation ads: chọn Precision > Recall.

---

### **Tóm tắt quyết định Evaluation & Validation:**

| Vấn đề                 | Giải pháp              | Khi nào dùng                         |
| ---------------------- | ---------------------- | ------------------------------------ |
| Underfit (bias cao)    | Model phức tạp hơn     | Khi train & test error cao           |
| Overfit (variance cao) | Regularization, CV     | Khi train error thấp, test error cao |
| Small dataset          | Cross-validation, LOO  | Khi data khan hiếm                   |
| Large dataset          | Hold-out + K-fold      | Khi data dư dả                       |
| Hyperparameter tuning  | Grid, Random, Bayesian | Tuỳ compute resources                |
| Threshold tuning       | ROC curve, PR curve    | Khi cần tối ưu Precision/Recall      |

---

✅ **Kết luận Evaluation & Validation:**

* Luôn kiểm tra **bias-variance tradeoff**.
* Dùng **cross-validation** để đánh giá ổn định.
* **Hyperparameter tuning** & **threshold tuning** giúp model phù hợp hơn business context.
* **Không bao giờ chọn model chỉ theo Accuracy** → phải gắn với **mục tiêu kinh doanh**.

---
# 🔹 **8. Practical Aspects in Machine Learning**

---

## **8.1 Model Interpretability**

### **Feature Importance**

* Với tree-based models (Random Forest, XGBoost) → dễ dàng lấy importance scores.
* Giúp biết feature nào ảnh hưởng nhiều nhất đến prediction.

### **SHAP (SHapley Additive exPlanations)**

* Dựa trên lý thuyết trò chơi.
* Mỗi feature có “đóng góp” vào prediction của từng instance.

### **LIME (Local Interpretable Model-agnostic Explanations)**

* Giải thích prediction cục bộ (local).
* Huấn luyện một model đơn giản xung quanh điểm cần giải thích.

**Ví dụ:**

* Finance: giải thích tại sao khách bị từ chối cho vay (model interpretability = bắt buộc vì quy định).
* Healthcare: giải thích dự đoán ung thư dựa vào yếu tố nào.

---

## **8.2 Handling Imbalanced Data**

⚠️ Thực tế, nhiều bài toán có **class imbalance** (ví dụ: 99% normal, 1% fraud).

### **Techniques**

* **Resampling:**

  * Oversampling (SMOTE – Synthetic Minority Oversampling).
  * Undersampling majority class.
* **Class Weights:** phạt nặng hơn cho minority misclassification.
* **Anomaly Detection Models:** (One-Class SVM, Isolation Forest).

**Ví dụ:**

* Fraud detection: chỉ 0.1% giao dịch gian lận → phải balance data.
* Medical rare disease prediction: bệnh hiếm chiếm <1% dân số.

---

## **8.3 Scalability & Efficiency**

### **Batching**

* Chia data lớn thành batch nhỏ để training → giảm memory usage.

### **Parallelization**

* Dùng multi-core CPU/GPU để tăng tốc training.

### **Distributed ML**

* Dùng frameworks:

  * **Apache Spark MLlib** → big data ML.
  * **Horovod, Ray** → distributed deep learning.

### **Model Compression**

* Quantization, pruning, distillation → giảm kích thước model.

**Ví dụ:**

* Training NLP models với hàng trăm triệu samples.
* Deploy deep learning trên mobile → cần quantization.

---

## **8.4 Deployment in Production**

### **Model Serving**

* REST API (Flask, FastAPI, Django).
* Model servers (TensorFlow Serving, TorchServe).

### **Monitoring**

* Monitor accuracy drift, data drift.
* Re-train khi dữ liệu thay đổi.

### **CI/CD for ML (MLOps)**

* Automate: data ingestion → training → testing → deployment.
* Tools: MLflow, Kubeflow, Airflow.

**Ví dụ:**

* E-commerce: update recommendation model mỗi tuần.
* Finance: update risk model khi thị trường thay đổi.

---

### **8.5 Data Privacy & Ethics**

* **Bias in data:** Model có thể phản ánh định kiến xã hội.
* **Fairness:** Đảm bảo công bằng cho các nhóm (gender, race).
* **Privacy-preserving ML:**

  * Differential Privacy.
  * Federated Learning (train trên device, không gửi data).

**Ví dụ:**

* Healthcare: phải tuân thủ HIPAA (US) hoặc GDPR (EU).
* Banking: mô hình tín dụng không được phân biệt giới tính.

---

### **Tóm tắt quyết định Practical Aspects:**

| Vấn đề               | Giải pháp                      | Khi nào dùng                 |
| -------------------- | ------------------------------ | ---------------------------- |
| Model khó giải thích | SHAP, LIME                     | Y tế, tài chính (regulation) |
| Dataset imbalance    | SMOTE, Class Weights           | Fraud, rare disease          |
| Dataset cực lớn      | Batch training, distributed ML | Big data pipelines           |
| Deploy model         | REST API, TF Serving, MLflow   | Production systems           |
| Ethical concerns     | Bias check, Federated Learning | Healthcare, HR, Banking      |

---

✅ **Kết luận Practical Aspects:**

* **Interpretability** quan trọng trong ngành regulated (finance, healthcare).
* **Imbalanced data** cần xử lý đúng, không thể chỉ dùng accuracy.
* **Scalability & MLOps** giúp ML vận hành trong production.
* **Ethics & privacy** ngày càng quan trọng trong AI.

---

# 🔹 **9. Advanced Topics in ML**

---

## **9.1 Reinforcement Learning (RL)**

### **Ý tưởng chính**

* Agent học bằng cách **tương tác với môi trường**.
* Nhận **reward** nếu hành động tốt, **penalty** nếu sai.
* Mục tiêu: tối đa hóa **cumulative reward**.

### **Thành phần**

* **State (S):** tình trạng hiện tại (ví dụ: vị trí robot).
* **Action (A):** hành động agent có thể thực hiện.
* **Reward (R):** feedback từ môi trường.
* **Policy (π):** chiến lược chọn action.

### **Thuật toán chính**

* **Q-Learning:** bảng Q-value để chọn action tốt nhất.
* **Deep Q-Networks (DQN):** dùng deep learning thay Q-table.
* **Policy Gradient (REINFORCE, PPO):** học trực tiếp policy.

**Ứng dụng:**

* **Game AI:** AlphaGo, Dota 2 bot.
* **Robotics:** xe tự lái, cánh tay robot.
* **Recommendation:** gợi ý nội dung theo thời gian thực.

---

## **9.2 Generative Models**

### **Generative Adversarial Networks (GANs)**

* Gồm **Generator** (tạo dữ liệu giả) và **Discriminator** (phân biệt thật/giả).
* Cạnh tranh nhau cho đến khi generator tạo dữ liệu giống thật.

**Ứng dụng:**

* Deepfake video, hình ảnh nghệ thuật AI.
* Data augmentation (tạo dữ liệu y tế giả để train).

---

### **Variational Autoencoders (VAEs)**

* Encoder → latent space → Decoder.
* Tạo dữ liệu mới bằng sampling từ latent space.

**Ứng dụng:**

* Tạo hình ảnh y tế giả phục vụ nghiên cứu.
* Recommendation (generate embedding cho sản phẩm mới).

---

## **9.3 Transfer Learning**

### **Ý tưởng chính**

* Dùng model pretrained trên dataset lớn → fine-tune trên dataset nhỏ.

### **Ưu điểm**

* Tiết kiệm dữ liệu, compute.
* Cải thiện performance.

### **Ứng dụng**

* **Computer Vision:** Dùng ResNet pretrained trên ImageNet để phân loại ảnh y tế.
* **NLP:** Dùng BERT/GPT pretrained để sentiment analysis.
* **Speech:** Whisper (OpenAI) cho speech-to-text, fine-tune cho ngôn ngữ ít tài nguyên.

---

## **9.4 Online & Incremental Learning**

### **Ý tưởng chính**

* Học từ dữ liệu **streaming**, cập nhật model liên tục.

### **Kỹ thuật**

* **Stochastic Gradient Descent (SGD):** tự nhiên phù hợp online learning.
* **Hoeffding Trees:** incremental decision trees.
* **Bandit Algorithms:** chọn action tối ưu với dữ liệu streaming (multi-armed bandits).

### **Ứng dụng**

* Stock market prediction.
* Real-time ad bidding.
* IoT sensor data analysis.

---

## **9.5 Meta Learning ("Learning to Learn")**

### **Ý tưởng**

* Thay vì train model mới từ đầu, ML học cách học nhanh hơn.

### **Thuật toán**

* Model-Agnostic Meta Learning (MAML).
* Few-shot learning, one-shot learning.

### **Ứng dụng**

* Computer vision: nhận diện đối tượng mới với rất ít ảnh.
* NLP: học từ vài câu ví dụ (prompt-based learning).

---

### **Tóm tắt quyết định Advanced ML:**

| Kỹ thuật               | Ý tưởng chính              | Ứng dụng                             |
| ---------------------- | -------------------------- | ------------------------------------ |
| Reinforcement Learning | Học qua phần thưởng/phạt   | Game AI, Robotics, Recommender       |
| GANs                   | Generator vs Discriminator | Ảnh giả, Deepfake, Data augmentation |
| VAEs                   | Latent space sampling      | Image generation, Embedding          |
| Transfer Learning      | Pretrained → Fine-tune     | CV, NLP, Speech                      |
| Online Learning        | Update theo stream         | Stock, Ads, IoT                      |
| Meta Learning          | Học cách học               | Few-shot, One-shot learning          |

---

✅ **Kết luận Advanced Topics:**

* **RL** → khi có feedback loop.
* **Generative Models** → khi cần tạo dữ liệu mới.
* **Transfer Learning** → khi data ít.
* **Online Learning** → khi dữ liệu đến liên tục.
* **Meta Learning** → khi cần học từ rất ít dữ liệu.

---
