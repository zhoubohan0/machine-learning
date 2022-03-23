from sklearn import linear_model
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np

def regressioner(method = 'ridge'):
    # cv表示交叉验证自动设置参数
    '''线性回归'''
    if method == 'linear':
        reg = linear_model.LinearRegression()
    '''Ridge回归：将回归系数缩小为零。alpha参数越大，偏差越大，方差越小，默认为1，调参trade-off'''
    if method == 'ridge':
        reg = linear_model.RidgeCV()
    '''LASSO回归：仅选择出最重要的信息性特征，将非信息特征系数减小至0,当重要特征很稀疏时推荐使用'''
    if method == 'lasso':
        reg = linear_model.LassoCV()
    '''弹性网络回归：同时使用L1,L2正则化，像LASSO删除无效特征，继承岭回归稳定性，适用于重要特征不多的稀疏矩阵'''
    if method in 'elasticnet':
        reg = linear_model.ElasticNetCV()  # 慎用好像有问题
    return reg

# 数据集
X = np.random.random((1000, 10))
y = np.dot(X, np.ones(10)) + np.random.random(1000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# 回归器
reg = regressioner('ridge')
reg.fit(X_train, y_train)
coef = reg.coef_                             # 回归系数
y_pred = reg.predict(X_test)                 # 预测值

# 回归评价
mse = mean_squared_error(y_test, y_pred)     # 损失函数：均方误差
R2 = 1 - mse / np.var(y_test)                # R^2：R方值,在函数内部进行predict(X_test)，再与y_test计算R^2
                                             # R^2=r2_score(y_test, y_pred)
# 超参数调参
alphas = np.logspace(-5, 5, 10,base=10)  # 从10^(-5)到10^(5)按照log_base_stop倍数产生长度为10的等比数列
scores = [reg.set_params(alpha=alpha).fit(X_train, y_train).score(X_test, y_test)for alpha in alphas]  # 计算每个α下的评分指标
alpha_best = alphas[np.argmax(scores)]      # 找到最佳α


