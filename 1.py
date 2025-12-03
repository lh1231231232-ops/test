# 1. 引入必要的库
from sklearn.tree import DecisionTreeClassifier # 引入一个叫"决策树"的算法
import pandas as pd

# 2. 准备数据 (通常这里是 pd.read_csv 读取文件)
# 假设我们有4个地点的历史数据
# 特征：[坡度(度), 降雨量(mm)]
X = [
    [10, 50],   # 地点A: 坡平、雨小
    [45, 200],  # 地点B: 坡陡、雨大
    [15, 60],   # 地点C: 坡平、雨小
    [50, 220]   # 地点D: 坡陡、雨大
]

# 标签：0代表没滑坡，1代表滑坡
y = [0, 1, 0, 1]

# 3. 建立模型 (初始化)
# 就像雇佣了一个还没培训的实习生
clf = DecisionTreeClassifier()

# 4. 训练模型 (Training) - 这一步最关键！
# 告诉模型：X是条件，y是对应的结果，请找规律。
clf.fit(X, y)

print("模型训练完毕！我已经学会判断滑坡了。")

# -----------------------------

# 5. 实际应用 (Prediction)
# 现在来了一个新地点：坡度40度，降雨180mm。它会滑坡吗？
new_location = [[40, 180]]

prediction = clf.predict(new_location)

if prediction[0] == 1:
    print(f"预测结果：危险！可能会滑坡 (结果代码: {prediction[0]})")
else:
    print(f"预测结果：安全 (结果代码: {prediction[0]})")