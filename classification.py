import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier,OneVsRestClassifier,OutputCodeClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import make_multilabel_classification,load_digits
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

def To1(X):
    # 将属性[最小,最大]归一化到[0,1],用于1/2维数据
    return MinMaxScaler(feature_range=(0,1)).fit_transform(X)

def onehot(y):  # y需二维
    return OneHotEncoder().fit_transform(y).toarray()

def classifier(method='svm'):  # 多分类分类器
    if method == 'svm':
        clf = svm.SVC(kernel='rbf', C=1, gamma=0.1)
    elif method in 'decisiontree':
        clf = DecisionTreeClassifier(criterion='gini',splitter="random",max_depth=10,max_features = 'auto')  # 最后两个参数用于剪枝需调参
    elif method in 'rf':
        clf = RandomForestClassifier(n_estimators=100,max_depth=10,max_features = 'auto')
    elif method in '':
        pass
    return clf


def plot_PR(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", linewidth=2, label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", linewidth=2, label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1.1])
    plt.show()


def plot_ROC(FPR, TPR, label=None):
    plt.plot(FPR, TPR, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()


if __name__ == '__main__':
    # 手写数字数据集
    digits = load_digits()
    images = digits.images.reshape((len(digits.images),-1))
    labels = digits.target

    # 预处理：归一化，独热编码
    images = To1(images)

    # 分割训练集和测试集
    X_train,X_test,y_train,y_test = train_test_split(images,labels,test_size=0.2,random_state=0)

    clf = classifier()
    clf.fit(X_train,y_train)

    # 分类结果
    y_pred = clf.predict(X_test)               # 分类类别
    acc = accuracy_score(y_test,y_pred)        # 准确率
    # index = clf.apply(X_test)                # 叶结点索引,树才有

    # 分类评估
    # https://www.jianshu.com/p/8d1dd2d37f87
    cm = confusion_matrix(y_test,y_pred)       # 混淆矩阵
    # importances = clf.feature_importances_   # 每一个特征对分类的贡献率

    # K折交叉验证评估准确性
    # https://blog.csdn.net/marsjhao/article/details/78678276
    scores = cross_val_score(clf,images,labels,cv=10)
    print(f'最大分类准确率：{max(scores)}，平均分类准确率：{np.mean(scores):.3f}')

    # 自动调参
    C_range = np.logspace(-2,10,10)
    gamma_range = np.logspace(-9,3,10)
    gs = GridSearchCV(clf,param_grid={'C':C_range,'gamma':gamma_range},cv=10)
    gs.fit(images,labels)
    # 调参结果
    print(f'最佳参数：{gs.best_params_}，最佳准确率：{gs.best_score_}')

    # PR曲线，POC曲线，AUC
    y_scores = cross_val_predict(clf, X_test, y_test, cv=3,method="decision_function")
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)  # PR
    FPR, TPR, _ = roc_curve(y_test, y_scores)                                   # ROC
    AUC = auc(FPR, TPR)                                                         # AUC

    # print(f'PR and ROC curve:')
    # plot_PR(precisions, recalls,thresholds)
    # plot_ROC(FPR, TPR)
    # print(f'AUC: {AUC}')

