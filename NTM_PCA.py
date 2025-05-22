import numpy as np
import pandas as pd
import warnings
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

# Bước 1: Đọc tệp .txt
data = []
with open('vidu4_lin_reg.txt', 'r') as f:
    lines = f.readlines()
    for line in lines[1:]:  # Bỏ qua dòng tiêu đề
        values = line.strip().split()  # Giả sử các giá trị được phân cách bằng khoảng trắng
        data.append([float(value) for value in values])  # Chuyển đổi thành số

# Chuyển đổi thành DataFrame
df = pd.DataFrame(data, columns=['ID', 'TUOI', 'BIM', 'HA', 'GLUCOSE', 'CHOLESTEROL', 'BEDAYNTM'])
df['BEDAYNTM'].value_counts()

Y = df['BEDAYNTM']
X = df[['TUOI', 'BIM', 'HA', 'GLUCOSE', 'CHOLESTEROL']]

#Chia dữ liệu train&test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Xây dựng mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)
# Tính Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error Before PCA(MSE):", mse)

#Chuẩn hóa dữ liệu
feature = StandardScaler().fit_transform(X)
#Áp dụng PCA
pca = PCA(n_components=2)
result = pca.fit_transform(feature)
Xnew = pd.DataFrame(result)

# Hiển thị trực quan kết quả
plt.figure(figsize=(8, 6))
plt.scatter(result[:, 0], result[:, 1], c='blue', edgecolor='k', s=50)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA - 2D Projection of Features')
plt.grid(True)
plt.show()

# Chia dữ liệu train&test
X_train, X_test, Y_train, Y_test = train_test_split(Xnew, Y, test_size=0.2, random_state=0)

#Áp dụng mô hình Linear Regression sau khi PCA
lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)

#Dự đoán đầu ra
Y_pred = lin_reg.predict(X_test)

# Tính Mean Squared Error (MSE)
mse_pca = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error After PCA(MSE):", mse)

#So sánh hệ số MSE
if abs(mse) > abs(mse_pca):
    print("\nĐộ sai lệch của dữ liệu gốc lơn hơn.")
elif abs(mse) < abs(mse_pca):
    print("\nĐộ sai lệch sau khi thực hiện PCA lớn hơn.")
else:
    print("\nĐộ sai lệch sau khi PCA không thay đổi.")





