import numpy as np

##导入数据
def load_dataset(filename):
    num_feat=len(open(filename).readline().split('\t'))-1
    data_mat=[];label_mat=[]
    fr=open(filename)
    for line in fr.readlines():
        linearr=[]
        curline=line.strip().split('\t')
        for i in range(num_feat):
            linearr.append(float(curline[i]))
        data_mat.append(linearr)
        label_mat.append(float(curline[-1]))
    return data_mat,label_mat

#计算最佳拟合直线

def standregres(x_arr,y_arr):
    x_mat=np.mat(x_arr);y_mat=np.mat(y_arr).T
    # print(x_mat)
    # print(y_mat)
    xTx=x_mat.T*x_mat
    # print(xTx)
    if np.linalg.det(xTx)==0:     #判断行列式是否为0
        print('this matrix is singular(奇异矩阵，即该矩阵的秩不是满秩） cannot do reverse')
    else:
        ws=np.linalg.solve(xTx,x_mat.T*y_mat)
        return ws


#局部加权线性回归函数
#它是一种非参数学习方法，在预测新样本值时候每次都会重新训练数据得到新的参数值，也就是说每次预测新样本都会依赖训练
#数据集合，所以每次得到的参数值都是不确定的。

def lwlr(test_point,x_arr,y_arr,k=1):
    x_mat=np.mat(x_arr)
    y_mat=np.mat(y_arr).T
    m=np.shape(x_mat)[0]
    weights=np.mat(np.eye(m))
    for j in range(m):
        diff_mat=test_point-x_mat[j,:]  #待预测点与样本点的距离
        weights[j,j]=np.exp(diff_mat*diff_mat.T/-2*k**2)
        print(weights)
    xTx = x_mat.T * (weights*x_mat)
    # print(xTx)
    if np.linalg.det(xTx) == 0:  # 判断行列式是否为0
        print('this matrix is singular(奇异矩阵，即该矩阵的秩不是满秩） cannot do reverse')
    else:
        ws = np.linalg.solve(xTx, x_mat.T *(weights*y_mat))
        return ws


#用于为数据集中的每个点调用lwlr()，有助于求解K的大小

def lwlr_test(test_arr,x_arr,y_arr,k=1):
    m=np.shape(test_arr)[0]
    y_hat=np.zeros(m)
    for i in range(m):
        y_hat[i]=lwlr(test_arr[i],x_arr,y_arr,k)
    return y_hat



#当数据的特征大于样本数，也就是说输入的数据矩阵不是满秩矩阵，因此不能用以上两种方法进行求解。
#此时，引入岭回归概念，就是在数据矩阵上加入一个lanmadaI使得矩阵非奇异。

def ridgeregres(x_mat,y_mat,lam=0.2):
    xTx=x_mat.T*x_mat
    denom=xTx+np.eye(shape(x_mat)[1])*lam
    if np.linalg.det(denom) ==0:
        print('this matrix is singluar,cannot do inverse')
    ws = denom.I*(x_mat.T * y_mat)
    return ws

def ridgetest(x_arr,y_arr):
    x_mat=np.mat(x_arr)
    y_mat=np.mat(y_arr).T
    y_mean=np.mean(y_mat,0)  # 压缩行，对各列求均值
    y_mat=y_mat-y_mean
    x_mean=np.mean(x_mat,0)
    x_var=np.var(x_mat,0)
    x_mat=(x_mat-x_mean)/x_var
    numtestpts=30
    wmat=np.zeros(numtestpts,shape(x_mat)[1])
    for i in range(numtestpts):
        ws=ridgeregres(x_mat,y_mat,exp(i-10))
        wmat[i,:]=ws.T
    return wmat












if __name__ == '__main__':
    x_arr,y_arr=load_dataset('E:/机器学习实践源代码/ex0.txt')
    # print(x_arr)
    # print(y_arr)
    # ws=standregres(x_arr,y_arr)
    # print(ws)
    # y_mat=np.mat(y_arr)
    # x_mat=np.mat(x_arr)
    # y_hat=x_mat*ws
    # print(y_mat)
    # corrcoef=np.corrcoef(y_hat.T,y_mat)  #计算相关系数
    # print(corrcoef)
    ws=lwlr(x_arr[0],x_arr,y_arr,1)
    print(ws)