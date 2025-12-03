import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # 用于拆分试卷
from sklearn.ensemble import RandomForestClassifier  # 这次我们要用更强大的“随机森林”
from sklearn.metrics import accuracy_score, confusion_matrix # 用于给模型打分

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# ==========================================
# 1. 制造“模拟数据” (假装这是你从野外采集回来的Excel表)
# ==========================================
# 假设我们采集了 1000 个滑坡点的数据
np.random.seed(42) # 保证每次运行生成的随机数一样

data = {
    '坡度': np.random.randint(5, 60, 1000),       # 坡度 5-60 度
    '降雨量': np.random.randint(50, 300, 1000),   # 降雨 50-300 mm
    '岩性硬度': np.random.randint(1, 5, 1000),    # 1=极软, 5=极硬
    '植被覆盖率': np.random.rand(1000)            # 0.0 - 1.0
}

df = pd.DataFrame(data)

# 强行制造一个“规律”作为标签 (现实中这个规律是未知的)
# 规律：坡度>30 且 降雨>150 容易滑坡(1)，否则不容易(0)
# 这里加一点随机噪音，模拟现实世界的复杂性
df['是否滑坡'] = ((df['坡度'] > 30) & (df['降雨量'] > 150)).astype(int)

print("--- 数据预览 (前5行) ---")
print(df.head())
print("-" * 30)

# ==========================================
# 2. 准备数据 (X 和 y)
# ==========================================
X = df[['坡度', '降雨量', '岩性硬度', '植被覆盖率']] # 特征
y = df['是否滑坡']                                  # 标签 (答案)

# ==========================================
# 3. 关键步骤：拆分 训练集 vs 测试集
# ==========================================
# test_size=0.3 意思是：70%的数据拿去学习，30%的数据留着期末考试
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"原始数据共 {len(df)} 条")
print(f"训练集用于学习: {len(X_train)} 条")
print(f"测试集用于考试: {len(X_test)} 条")

# ==========================================
# 4. 训练模型 (只允许看训练集!)
# ==========================================
# 随机森林 (Random Forest) 是地质灾害领域最好用的模型之一
model = RandomForestClassifier(n_estimators=100) 
model.fit(X_train, y_train)

# ==========================================
# 5. 期末考试 (在测试集上预测)
# ==========================================
y_pred = model.predict(X_test)

# ==========================================
# 6. 老师批改试卷 (评估)
# ==========================================
accuracy = accuracy_score(y_test, y_pred)
print("-" * 30)
print(f"模型在测试集上的准确率: {accuracy * 100:.2f}%")

# 看看模型觉得哪些特征最重要？
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
print("\n模型认为最重要的致灾因子：")
print(feature_importance.sort_values(ascending=False))





# ... (接你之前的代码) ...

# 1. 打印详细的分类报告
print("--- 详细评估报告 ---")
# 重点看 Recall (召回率) 和 F1-score
print(classification_report(y_test, y_pred, target_names=['安全(0)', '滑坡(1)']))

# 2. 画出混淆矩阵 (让它可视化)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
# annot=True 表示在格子里显示数字，fmt='d' 表示显示整数
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Pred:安全', 'Pred:滑坡'], 
            yticklabels=['True:安全', 'True:滑坡'])
plt.xlabel('模型预测结果')
plt.ylabel('真实地质情况')
plt.title('混淆矩阵 (Confusion Matrix)')
plt.show()