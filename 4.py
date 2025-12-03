import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    cohen_kappa_score
)
from xgboost import XGBClassifier



# ==============================
# 全局画图风格
# ==============================
plt.style.use('seaborn-v0_8')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False    # 显示负号

np.random.seed(2025)

# ==========================================
# 1. 构造一个“真实的”地质数据集
# ==========================================
n_samples = 2000

data = {
    'Slope': np.random.normal(30, 10, n_samples),                # 坡度：正态分布，平均 30°
    'Rainfall': np.random.normal(100, 30, n_samples),            # 降雨量：平均 100mm
    'Lithology': np.random.choice(['Granite', 'Mudstone', 'Limestone'], n_samples),
    'Fault_Distance': np.random.uniform(0, 5000, n_samples)      # 断层距离：0-5000 m
}

df = pd.DataFrame(data)

def simulate_landslide(row):
    """根据地质条件模拟滑坡概率，并掷骰子决定是否发生滑坡"""
    score = 0
    score += (row['Slope'] - 20) * 0.5           # 坡度越大越危险
    score += (row['Rainfall'] - 80) * 0.2        # 雨越大越危险
    if row['Lithology'] == 'Mudstone':
        score += 20                               # 泥岩很危险
    if row['Lithology'] == 'Granite':
        score -= 10                               # 花岗岩较稳定
    if row['Fault_Distance'] < 500:
        score += 15                               # 离断层近很危险

    # Sigmoid 转成概率
    prob = 1 / (1 + np.exp(-0.1 * score))
    # 掷骰子
    return 1 if np.random.rand() < prob else 0

df['Label'] = df.apply(simulate_landslide, axis=1)

print("--- 原始数据预览 (注意 Lithology 是文字) ---")
print(df.head())

# ==========================================
# 2. 类别平衡（下采样）在原始 df 上进行
# ==========================================
df_slide = df[df['Label'] == 1]   # 滑坡
df_safe  = df[df['Label'] == 0]   # 未滑坡

print(f"\n原始数据：滑坡 {len(df_slide)} 个，未滑坡 {len(df_safe)} 个")

# 选取较少的那一类为基准，做对称下采样
n_min = min(len(df_slide), len(df_safe))

df_slide_sample = df_slide.sample(n=n_min, random_state=42)
df_safe_sample  = df_safe.sample(n=n_min, random_state=42)

df_balanced = pd.concat([df_slide_sample, df_safe_sample]) \
                .sample(frac=1, random_state=42) \
                .reset_index(drop=True)

print(f"平衡后的数据：滑坡 {len(df_slide_sample)} 个，未滑坡 {len(df_safe_sample)} 个")

# ==========================================
# 3. 特征工程：独热编码 & 划分训练/测试集
# ==========================================
df_balanced_encoded = pd.get_dummies(df_balanced, columns=['Lithology'])

print("\n--- 编码后的数据 (计算机能读了) ---")
print(df_balanced_encoded.head())

X = df_balanced_encoded.drop('Label', axis=1)
y = df_balanced_encoded['Label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# # ==========================================
# # 4. 基础随机森林模型训练与评估
# # ==========================================
# base_model = RandomForestClassifier(n_estimators=100, random_state=42)
# base_model.fit(X_train, y_train)
# y_pred_base = base_model.predict(X_test)



# ==========================================
# 4. 基础 XGBoost 模型训练与评估
# ==========================================
base_model = XGBClassifier(
    n_estimators=200,       # 基础树的数量
    max_depth=4,           # 树深度
    learning_rate=0.1,     # 学习率
    subsample=0.8,         # 行采样比例
    colsample_bytree=0.8,  # 列采样比例
    eval_metric='logloss', # 避免版本警告
    random_state=42
)

base_model.fit(X_train, y_train)
y_pred_base = base_model.predict(X_test)





acc_base = accuracy_score(y_test, y_pred_base)
print(f"\n基础随机森林模型 准确率: {acc_base*100:.2f}%")
print("\n分类报告：")
print(classification_report(y_test, y_pred_base))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred_base)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='OrRd',
            xticklabels=['Safe', 'Landslide'],
            yticklabels=['Safe', 'Landslide'])
plt.title(f'Confusion Matrix (Acc: {acc_base:.2f})')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# 特征重要性
importances = pd.Series(base_model.feature_importances_, index=X.columns) \
                .sort_values(ascending=False)
print("\n致灾因子重要性排序：")
print(importances)

# ==========================================
# 5. ROC 曲线 & AUC（基础模型）
# ==========================================
y_scores_base = base_model.predict_proba(X_test)[:, 1]  # 滑坡的概率
fpr, tpr, thresholds = roc_curve(y_test, y_scores_base)
roc_auc_base = auc(fpr, tpr)
print(f"\n基础模型 AUC 值: {roc_auc_base:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, lw=2, label=f'Base ROC (AUC = {roc_auc_base:.2f})')
plt.plot([0, 1], [0, 1], lw=2, linestyle='--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (误报率)')
plt.ylabel('True Positive Rate (命中率)')
plt.title('Receiver Operating Characteristic (ROC) - 基础滑坡预测模型')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

# # ==========================================
# # 6. GridSearchCV 自动调参（以 AUC 为目标）
# # ==========================================
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5, 10]
# }

# print("\n开始自动调参 (GridSearchCV)...")

# grid_search = GridSearchCV(
#     estimator=RandomForestClassifier(random_state=42),
#     param_grid=param_grid,
#     cv=5,
#     scoring='roc_auc',
#     n_jobs=-1
# )



# ==========================================
# 6. GridSearchCV 自动调参（以 AUC 为目标）—— XGBoost 版本
# ==========================================
param_grid = {
    'n_estimators': [100, 200, 300],    # 树的数量
    'max_depth': [3, 4, 5],             # 树深度
    'learning_rate': [0.05, 0.1, 0.2],  # 学习率
    'subsample': [0.8, 1.0],            # 行采样比例
    'colsample_bytree': [0.8, 1.0]      # 列采样比例
}

print("\n开始自动调参 (GridSearchCV, XGBoost)...")

grid_search = GridSearchCV(
    estimator=XGBClassifier(
        eval_metric='logloss',
        random_state=42
    ),
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("-" * 40)
print(f"基础模型 AUC: {roc_auc_base:.4f}")
print(f"调参后交叉验证平均 AUC: {grid_search.best_score_:.4f}")
print("-" * 40)
print("最佳参数组合 (Best Parameters):")
print(grid_search.best_params_)

best_model = grid_search.best_estimator_





grid_search.fit(X_train, y_train)

print("-" * 40)
print(f"基础模型 AUC: {roc_auc_base:.4f}")
print(f"调参后交叉验证平均 AUC: {grid_search.best_score_:.4f}")
print("-" * 40)
print("最佳参数组合 (Best Parameters):")
print(grid_search.best_params_)

best_model = grid_search.best_estimator_

# 在测试集上评估最优模型
y_pred_best = best_model.predict(X_test)
acc_best = accuracy_score(y_test, y_pred_best)
print(f"\n最优模型在测试集上的准确率: {acc_best*100:.2f}%")

y_scores_best = best_model.predict_proba(X_test)[:, 1]
fpr_best, tpr_best, _ = roc_curve(y_test, y_scores_best)
roc_auc_best = auc(fpr_best, tpr_best)
print(f"最优模型在测试集上的 AUC: {roc_auc_best:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(fpr_best, tpr_best, lw=2, label=f'Best ROC (AUC = {roc_auc_best:.2f})')
plt.plot([0, 1], [0, 1], lw=2, linestyle='--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (误报率)')
plt.ylabel('True Positive Rate (命中率)')
plt.title('Receiver Operating Characteristic (ROC) - 最优滑坡预测模型')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

# Kappa 系数（用最优模型）
kappa = cohen_kappa_score(y_test, y_pred_best)
print(f"\n最优模型的 Kappa 系数: {kappa:.4f}")

# ==========================================
# 7. 基于最优模型的 LSM（滑坡易发性图）
# ==========================================
N = 50  # 地图分辨率 50x50
x_coords = np.linspace(0, 100, N)
y_coords = np.linspace(0, 100, N)
xx, yy = np.meshgrid(x_coords, y_coords)

flat_x = xx.ravel()
flat_y = yy.ravel()

# 模拟地质特征：
# - 越往东 (x 越大)，坡度越陡
# - 越往北 (y 越大)，降雨越多
# - 地图中心 (50, 50) 附近断层活动强
# - 假设岩性为泥岩 (最危险)，便于体现高风险区
grid_data = pd.DataFrame({
    'Slope': flat_x * 0.6 + np.random.normal(0, 5, N*N),
    'Rainfall': flat_y * 2.5 + np.random.normal(0, 20, N*N),
    'Fault_Distance': np.sqrt((flat_x - 50)**2 + (flat_y - 50)**2) * 50,
    'Lithology_Granite': 0,
    'Lithology_Limestone': 0,
    'Lithology_Mudstone': 1
})

# 列顺序必须和训练时完全一致
grid_data = grid_data[X.columns]

print(f"\n地图网格准备就绪，共 {len(grid_data)} 个像元待预测...")

map_probs_flat = best_model.predict_proba(grid_data)[:, 1]
map_probs_2d = map_probs_flat.reshape(N, N)

plt.figure(figsize=(10, 8))
sns.heatmap(
    map_probs_2d,
    cmap='RdYlGn_r',
    vmin=0,
    vmax=1,
    cbar_kws={'label': '滑坡预测概率 (Probability)'}
)
plt.title('区域滑坡易发性评价图 (Simulated LSM)', fontsize=15)
plt.xticks([])
plt.yticks([])
plt.show()

# ==========================================
# 8. 因子相关性分析示例（防止多重共线性）
# ==========================================
# 这里用平衡后的数据构造一个高度相关的“起伏度”特征做示例
corr_data = df_balanced[['Slope', 'Rainfall', 'Fault_Distance']].copy()
corr_data['Relief'] = corr_data['Slope'] * 2 + np.random.normal(0, 1, len(corr_data))

correlation_matrix = corr_data.corr()
print("\n因子相关性数值矩阵：")
print(correlation_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    vmin=-1,
    vmax=1
)
plt.title('因子相关性分析 (Feature Correlation Matrix)')
plt.show()
