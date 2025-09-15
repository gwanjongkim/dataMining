import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


df = pd.read_csv("여성 신체 데이터.csv", encoding = 'utf-8', low_memory=False)
df = df.apply(pd.to_numeric, errors='coerce')
data_list = df.values.tolist()

data_arr = np.array(data_list)

X = data_arr[1:, [6, 12, 15]]   #[나이, 허리둘레, 키]
Y = data_arr[1:, 11]#상체 옷 사이즈 

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, Y)

# 예측할 새로운 사람 바로 입력
predicted_class = knn.predict([[26, 60, 167]])[0]
print(predicted_class)