import matplotlib.pyplot as plt
import numpy as np
import xlrd
from sklearn import preprocessing
from sklearn.cluster import DBSCAN
from matplotlib import animation



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

        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        A = np.array(data)
        X = min_max_scaler.fit_transform(A)

        pedestrian_db = DBSCAN(eps=0.2, min_samples=4).fit_predict(X)
        ax.clear()
        ax.scatter(x=A[:, 0], y=A[:, 1], s=50, c=pedestrian_db)
        ax.set_title('DBSCAN Results', fontsize=10)
        return scatter

def init():
    scatter = ax.scatter([], [])
    return scatter,

fig, ax = plt.subplots()
scatter =  ax.scatter([], [])
ani = animation.FuncAnimation(fig, update, 540, interval=50, init_func=init)
plt.show()
Writer = animation.writers['ffmpeg']  # 需安装ffmpeg
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
ani.save("Pedestrian_Animation.mp4", writer=writer)

