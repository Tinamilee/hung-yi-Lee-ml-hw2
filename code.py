#C:\Users\lenovo\Desktop\机器学习作业\收入预测\code.py
#by LogisticRegression
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
import matplotlib as mpl
#解决中文显示问题
mpl.rcParams["font.sans-serif"] = [u"SimHei"]
mpl.rcParams["axes.unicode_minus"] = False

df=pd.read_table('X_train',sep=',',header=None,skiprows=[0])
X=df
df=pd.read_table('Y_train',sep=',',header=None,skiprows=[0])
Y=df
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.0001,random_state=0)

ss = StandardScaler()
X_train = ss.fit_transform(X_train)     #先拟合数据在进行标准化
X_test = ss.transform(X_test)       #数据标准化
'''
在调参时如果我们主要的目的只是为了解决过拟合，一般penalty选择L2正则化就够了。
但是如果选择L2正则化发现还是过拟合，即预测效果差的时候，就可以考虑L1正则化。
另外，如果模型的特征非常多，我们希望一些不重要的特征系数归零，
从而让模型系数稀疏化的话，也可以使用L1正则化。
'''
'''
model = LogisticRegression(penalty='l1')
model.fit(X_train,Y_train)
r=model.score(X_train,Y_train)
sc=model.score(X_test,Y_test)
print(r)
print(sc)

test=pd.read_table('X_test',sep=',',header=None,skiprows=[0])
test = ss.transform(test)
Y_predict = model.predict(test)
np.savetxt('data.csv',(Y_predict),fmt='%d')
'''
#构建并训练模型
##  multi_class:分类方式选择参数，有"ovr(默认)"和"multinomial"两个值可选择，在二元逻辑回归中无区别
##  cv:几折交叉验证
##  solver:优化算法选择参数，当penalty为"l1"时，参数只能是"liblinear(坐标轴下降法)"
##  "lbfgs"和"cg"都是关于目标函数的二阶泰勒展开
##  当penalty为"l2"时，参数可以是"lbfgs(拟牛顿法)","newton_cg(牛顿法变种)","seg(minibactch随机平均梯度下降)"
##  维度<10000时，选择"lbfgs"法，维度>10000时，选择"cs"法比较好，显卡计算的时候，lbfgs"和"cs"都比"seg"快
##  penalty:正则化选择参数，用于解决过拟合，可选"l1","l2"
##  tol:当目标函数下降到该值是就停止，叫：容忍度，防止计算的过多
lr = LogisticRegressionCV(multi_class="ovr",fit_intercept=True,Cs=np.logspace(-2,2,20),cv=3,penalty="l1",solver="liblinear",tol=0.01)

#X = ss.fit_transform(X)     #先 拟合数据在进行标准化
#re = lr.fit(X,Y)

re = lr.fit(X_train,Y_train)
r = re.score(X_train,Y_train)
sc = re.score(X_test,Y_test)
print(r)
print(sc)

test=pd.read_table('X_test',sep=',',header=None,skiprows=[0])
test = ss.transform(test)
Y_predict = re.predict(test)
np.savetxt('data.csv',(Y_predict),fmt='%d')