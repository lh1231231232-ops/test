import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 设置绘图风格和字体（防止乱码，这里用英文标签以防万一）
plt.style.use('seaborn-v0_8')
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# ==========================================
# 1. 构造一个“真实的”地质数据集
# ==========================================
np.random.seed(2025)
n_samples = 2000

# 构造特征
data = {
    # 坡度：正态分布，平均30度
    'Slope': np.random.normal(30, 10, n_samples), 
    
    # 降雨量：平均100mm
    'Rainfall': np.random.normal(100, 30, n_samples),
    
    # 岩性：这是一个文本特征！计算机看不懂！
    'Lithology': np.random.choice(['Granite', 'Mudstone', 'Limestone'], n_samples),
    
    # 断层距离：0-5000米
    'Fault_Distance': np.random.uniform(0, 5000, n_samples)
}

df = pd.DataFrame(data)

# 构造真实的滑坡逻辑（加入概率，模拟真实世界的复杂性）
# 逻辑：泥岩(Mudstone)容易滑，坡度大容易滑，离断层近容易滑
# 但我们加入 random.rand 随机噪声，让规律不再绝对
def simulate_landslide(row):
    score = 0
    score += (row['Slope'] - 20) * 0.5            # 坡度越大越危险
    score += (row['Rainfall'] - 80) * 0.2         # 雨越大越危险
    if row['Lithology'] == 'Mudstone': score += 20 # 泥岩很危险
    if row['Lithology'] == 'Granite': score -= 10  # 花岗岩很稳
    if row['Fault_Distance'] < 500: score += 15    # 离断层近危险
    
    # 转化为概率 (Sigmoid函数思想)
    prob = 1 / (1 + np.exp(-0.1 * score))
    
    # 掷骰子决定是否滑坡 (不再是 100% 确定)
    return 1 if np.random.rand() < prob else 0

df['Label'] = df.apply(simulate_landslide, axis=1)

print("--- 原始数据预览 (注意 Lithology 是文字) ---")
print(df.head())

# ==========================================
# 2. 特征工程：独热编码 (One-Hot Encoding)
# ==========================================
# 这一步非常重要！把 "Mudstone", "Granite" 变成 0 和 1
# pd.get_dummies 会自动把一列文字拆成多列数字
df_encoded = pd.get_dummies(df, columns=['Lithology'])

print("\n--- 编码后的数据 (计算机能读了) ---")
print(df_encoded.head())

# 假设 df 中 Label=1 是滑坡，Label=0 是非滑坡

# 1. 把两类分开
df_slide = df[df['Label'] == 1]
df_safe = df[df['Label'] == 0]

# 2. 从几万个安全点里，随机抽取和滑坡点一样多的数量 (Downsampling)
df_safe_sample = df_safe.sample(n=len(df_slide), random_state=42)

# 3. 拼在这个一起，组成新的训练数据
df_balanced = pd.concat([df_slide, df_safe_sample])

# 4. 打乱顺序
df_balanced = df_balanced.sample(frac=1).reset_index(drop=True)

print(f"平衡后的数据：滑坡 {len(df_slide)} 个，非滑坡 {len(df_safe_sample)} 个")
# 然后用 df_balanced 去训练模型



# ==========================================
# 3. 训练与测试
# ==========================================
X = df_encoded.drop('Label', axis=1) # 所有列除了Label都是特征
y = df_encoded['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ==========================================
# 4. 结果评估
# ==========================================
acc = accuracy_score(y_test, y_pred)
print(f"\n模型准确率: {acc*100:.2f}% (注意看是不是100%了？)")

# 画混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='OrRd',
            xticklabels=['Safe', 'Landslide'],
            yticklabels=['Safe', 'Landslide'])
plt.title(f'Confusion Matrix (Acc: {acc:.2f})')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# 查看哪个因素最重要
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n致灾因子重要性排序：")
print(importances)


from sklearn.metrics import roc_curve, auc

# ==========================================
# 1. 获取概率值 (这是画 ROC 的关键！)
# ==========================================
# model.predict_proba(X_test) 会返回两列：
# 第0列是“不滑坡”的概率，第1列是“滑坡”的概率
# 我们只需要第1列（关注滑坡的概率）
y_scores = model.predict_proba(X_test)[:, 1]

# ==========================================
# 2. 计算 FPR, TPR 和 AUC 值
# ==========================================
# FPR: 假阳性率 (误报率)
# TPR: 真阳性率 (召回率/命中率)
fpr, tpr, thresholds = roc_curve(y_test, y_scores)

# 计算面积 (AUC)
roc_auc = auc(fpr, tpr)
print(f"--- 关键指标 ---")
print(f"AUC 值: {roc_auc:.4f}")

# ==========================================
# 3. 绘制 ROC 曲线
# ==========================================
plt.figure(figsize=(8, 6))

# 画模型曲线
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')

# 画对角线 (纯瞎猜的线)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (误报率)')
plt.ylabel('True Positive Rate (命中率)')
plt.title('Receiver Operating Characteristic (ROC) - 滑坡预测模型')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()


from sklearn.model_selection import GridSearchCV

# 1. 定义我们要尝试的“配方表” (参数空间)
# 就像做实验设计(DOE)一样
param_grid = {
    'n_estimators': [50, 100, 200],      # 尝试种多少棵树：50, 100, 还是 200？
    'max_depth': [None, 10, 20],         # 树最大能长多深？太深容易死记硬背(过拟合)
    'min_samples_split': [2, 5, 10]      # 至少多少个样本才允许分叉？
}

print("开始自动调参，可能需要一点时间，请稍等...")

# 2. 建立自动搜寻器
# cv=5 意思是：做5次交叉验证（模拟5次期末考试取平均分），防止偶然性
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                            param_grid=param_grid,
                            cv=5, 
                            scoring='roc_auc', # 我们现在的目标是把 AUC 刷高
                            n_jobs=-1)         # 调用电脑所有CPU核心加速

# 3. 开始跑实验
grid_search.fit(X_train, y_train)

# 4. 公布最佳配方
print("-" * 30)
print(f"之前的 AUC 可能是 0.73 左右")
print(f"调参后最佳 AUC 得分: {grid_search.best_score_:.4f}")
print("-" * 30)
print("夺冠的参数组合 (Best Parameters):")
print(grid_search.best_params_)

# 5. 用最强模型再考一次试
best_model = grid_search.best_estimator_
y_pred_new = best_model.predict(X_test)
acc_new = accuracy_score(y_test, y_pred_new)
print(f"\n在测试集上的最终准确率: {acc_new*100:.2f}%")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 定义地图网格 (分辨率 50x50)
# ==========================================
N = 50
# 生成从西到东(x轴)，从南到北(y轴)的坐标
x_coords = np.linspace(0, 100, N)
y_coords = np.linspace(0, 100, N)
#织成一张网
xx, yy = np.meshgrid(x_coords, y_coords)

# 拉平成一条长线，方便做成 DataFrame
flat_x = xx.ravel()
flat_y = yy.ravel()

# ==========================================
# 2. 为地图上的每个点模拟地质特征
# ==========================================
# 为了让地图看起来有规律，我们设定：
# - 越往东边走(x越大)，坡度越陡
# - 越往北边走(y越大)，降雨越多
# - 假设区域中心有一条断层，离中心越近，Fault_Distance越小
# - 假设整个区域岩性都是泥岩 (最危险的岩性)

grid_data = pd.DataFrame({
    # 模拟坡度：受X坐标影响 + 一点随机噪声
    'Slope': flat_x * 0.6 + np.random.normal(0, 5, N*N),
    
    # 模拟降雨：受Y坐标影响 + 一点随机噪声
    'Rainfall': flat_y * 2.5 + np.random.normal(0, 20, N*N),
    
    # 模拟断层距离：计算到地图中心点(50,50)的欧式距离
    'Fault_Distance': np.sqrt((flat_x - 50)**2 + (flat_y - 50)**2) * 50,
    
    # 模拟岩性：全部设为泥岩 (Mudstone)
    # 注意：这里必须和训练时的 One-Hot 编码列名完全一致！
    'Lithology_Granite': 0,
    'Lithology_Limestone': 0,
    'Lithology_Mudstone': 1
})

# ***关键一步***：确保列的顺序和训练时一模一样，否则模型会认错人
grid_data = grid_data[X_train.columns]

print(f"地图网格准备就绪，共 {len(grid_data)} 个像元待预测...")

# ==========================================
# 3. 调用你的最强模型进行全图扫描
# ==========================================
# 我们要的是概率 (predict_proba)，取第1列（滑坡的概率）
map_probs_flat = best_model.predict_proba(grid_data)[:, 1]

# 把预测出的长条概率，重新折叠回 50x50 的方块形状
map_probs_2d = map_probs_flat.reshape(N, N)

# ==========================================
# 4. 绘制滑坡易发性热力图 (LSM)
# ==========================================
plt.figure(figsize=(10, 8))

# RdYlGn_r: 红-黄-绿 的反向渐变色 (Red-Yellow-Green reversed)
# 红色代表高概率(1.0)，绿色代表低概率(0.0)
sns.heatmap(map_probs_2d, cmap='RdYlGn_r', 
            vmin=0, vmax=1, # 固定色标范围在 0~1 之间
            cbar_kws={'label': '滑坡预测概率 (Probability)'})

plt.title('区域滑坡易发性评价图 (Simulated LSM)', fontsize=15)
# 为了美观，关掉坐标轴刻度
plt.xticks([])
plt.yticks([])
# 如果汉字乱码，请用下面这行英文标题
# plt.title('Landslide Susceptibility Map (LSM)', fontsize=15)

plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 假设 X 是你的特征数据 (DataFrame格式)
# 为了演示，我们先创建一个包含高度相关特征的假数据
# 你可以直接用你之前的 df 或 X_train
data_check = pd.DataFrame({
    '坡度': np.random.rand(100),
    '降雨量': np.random.rand(100),
    '断层距离': np.random.rand(100),
    # 制造一个"废话"特征：起伏度 = 坡度 * 2 + 一点点误差
    # 这意味着"起伏度"和"坡度"是高度相关的
    '起伏度': lambda x: x['坡度'] * 2 + np.random.normal(0, 0.01, 100)
})
data_check['起伏度'] = data_check['坡度'] * 2 + np.random.normal(0, 0.01, 100)

# ==========================================
# 1. 计算相关性矩阵 (Pearson系数)
# ==========================================
# corr() 函数会自动计算每两列之间的关系
# 范围是 -1 到 1
# 1 = 完全正相关 (你大我也大)
# 0 = 没关系
# -1 = 完全负相关 (你大我就小)
correlation_matrix = data_check.corr()

print("相关性数值矩阵：")
print(correlation_matrix)

# ==========================================
# 2. 画出热力图 (论文里的标准配图)
# ==========================================
plt.figure(figsize=(8, 6))

# annot=True: 在格子里显示数字
# cmap='coolwarm': 红色代表正相关，蓝色代表负相关
# vmin=-1, vmax=1: 锁死颜色范围
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)

plt.title('因子相关性分析 (Feature Correlation Matrix)')
plt.show()

from sklearn.metrics import cohen_kappa_score

# y_test 是真实答案，y_pred 是模型预测结果(0和1)
kappa = cohen_kappa_score(y_test, y_pred)

print(f"模型的 Kappa 系数: {kappa:.4f}")