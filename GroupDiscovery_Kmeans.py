import matplotlib.pyplot as plt
import numpy as np
import xlrd
from sklearn import preprocessing
from matplotlib import animation

def normalize(X, axis=-1, p=2):
    lp_norm = np.atleast_1d(np.linalg.norm(X, p, axis))
    lp_norm[lp_norm == 0] = 1
    return X / np.expand_dims(lp_norm, axis)


# 计算一个样本与数据集中所有样本的欧氏距离的平方
def euclidean_distance(one_sample, X):
    one_sample = one_sample.reshape(1, -1)
    X = X.reshape(X.shape[0], -1)
    distances = np.power(np.tile(one_sample, (X.shape[0], 1)) - X, 2).sum(axis=1)
    return distances


class Kmeans():
    """Kmeans聚类算法.
    Parameters:
    -----------
    k: int
        聚类的数目.
    max_iterations: int
        最大迭代次数.
    varepsilon: float
        判断是否收敛, 如果上一次的所有k个聚类中心与本次的所有k个聚类中心的差都小于varepsilon,
        则说明算法已经收敛

    初始值：k=4, max_iterations=500, varepsilon=0.0001
    """

    def __init__(self, k, max_iterations, varepsilon):
        self.k = k
        self.max_iterations = max_iterations
        self.varepsilon = varepsilon

    # 从所有样本中随机选取self.k样本作为初始的聚类中心
    def init_random_centroids(self, X):
        n_samples, n_features = np.shape(X)
        centroids = np.zeros((self.k, n_features))
        for i in range(self.k):
            centroid = X[np.random.choice(range(n_samples))]
            centroids[i] = centroid
        return centroids

    # 返回距离该样本最近的一个中心索引[0, self.k)
    def _closest_centroid(self, sample, centroids):
        distances = euclidean_distance(sample, centroids)
        closest_i = np.argmin(distances)
        return closest_i

    # 将所有样本进行归类，归类规则就是将该样本归类到与其最近的中心
    def create_clusters(self, centroids, X):
        n_samples = np.shape(X)[0]
        clusters = [[] for _ in range(self.k)]
        for sample_i, sample in enumerate(X):
            centroid_i = self._closest_centroid(sample, centroids)
            clusters[centroid_i].append(sample_i)
        return clusters

    # 对中心进行更新
    def update_centroids(self, clusters, X):
        n_features = np.shape(X)[1]
        centroids = np.zeros((self.k, n_features))
        for i, cluster in enumerate(clusters):
            centroid = np.mean(X[cluster], axis=0)
            centroids[i] = centroid
        return centroids

    # 将所有样本进行归类，其所在的类别的索引就是其类别标签
    def get_cluster_labels(self, clusters, X):
        y_pred = np.zeros(np.shape(X)[0])
        for cluster_i, cluster in enumerate(clusters):
            for sample_i in cluster:
                y_pred[sample_i] = cluster_i
        return y_pred

    # 对整个数据集X进行Kmeans聚类，返回其聚类的标签
    def predict(self, X):
        # 从所有样本中随机选取self.k样本作为初始的聚类中心
        centroids = self.init_random_centroids(X)

        # 迭代，直到算法收敛(上一次的聚类中心和这一次的聚类中心几乎重合)或者达到最大迭代次数
        for _ in range(self.max_iterations):
            # 将所有进行归类，归类规则就是将该样本归类到与其最近的中心
            clusters = self.create_clusters(centroids, X)
            former_centroids = centroids

            # 计算新的聚类中心
            centroids = self.update_centroids(clusters, X)

            # 如果聚类中心几乎没有变化，说明算法已经收敛，退出迭代
            diff = centroids - former_centroids
            if diff.any() < self.varepsilon:
                break

        return self.get_cluster_labels(clusters, X)



# 读取数据
index = 0
wk = xlrd.open_workbook(r'C:\Users\26303\Desktop\AI_Modern_Methods\人工智能现代方法行人分组作业\GroupDiscovery\TrajectoryData_students003\students003.xls')
sheets = wk.sheet_by_name('Sheet1')
ws = wk.sheet_by_index(0)
# 获取总行数
nrows = ws.nrows


def update(frame):
        TimeStamp = frame
        # 分帧读取数据
        data = []
        global index
        for i in range(index, nrows):
            time = sheets.cell_value(i, 0)
            if time == TimeStamp * 10:
                index = index + 1
                rowx = sheets.cell_value(i, 2)
                rowy = sheets.cell_value(i, 3)
                data.append([rowx, rowy])
                print(index)
            else:
                break

        # 标准化数据集
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        A = np.array(data)
        X = min_max_scaler.fit_transform(A)
        num, dim = X.shape
        clf = Kmeans(3, 500, 0.0001)

        # 返回聚类标签
        y_predict = clf.predict(X)
        print(y_predict)
        color = ['r', 'g', 'b', 'c', 'y', 'm', 'k']
        ax.clear()
        for p in range(0, num):
            y = y_predict[p]
            scatter = ax.scatter(A[p, 0], A[p, 1], c=color[int(y)])
        return scatter

def init():
    scatter = ax.scatter([], [])
    return scatter

fig, ax = plt.subplots()
scatter =  ax.scatter([], [])
ani = animation.FuncAnimation(fig, update, 540, interval=20, init_func=init)
plt.show()
ani.save(r'Pedestrian_Animation.mp4', writer='ffmpeg' , fps=1000/20)
