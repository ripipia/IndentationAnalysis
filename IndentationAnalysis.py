import numpy as np
import open3d as o3d
import csv
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


##csv파일 불러와서 list로 저장하기
IDTpointsList = []

inputfile = "Y2_4KA.csv"
x_index = 0
y_index = 0
f = open(inputfile, 'r')
rdr = csv.reader(f)

#range_x = (950, 1050)
#range_y = (100, 200)
range_x = (0, 2001)
range_y = (0, 315)


for line in rdr:
    for indentation in line:
        if range_x[0] < x_index and x_index < range_x[1] and range_y[0] < y_index and y_index < range_y[1]:
            IDTpoint = [x_index * 1, y_index/315*1000, float(indentation)*1000]
            IDTpointsList.append(IDTpoint)
        x_index = x_index + 1
    x_index = 0
    y_index = y_index + 1

##List를 np로 저장하기
pc_array = np.array(IDTpointsList, dtype=np.float32)

##np를 pcd 형태로 변환하기
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pc_array)

##노멀 계산
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

##다운 샘플링(outlier를 찾기위한 시간을 단축)
uni_down_pcd = pcd.uniform_down_sample(10)


##아웃라이어 찾기
def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

inlier, ind = uni_down_pcd.remove_statistical_outlier(nb_neighbors=1000,std_ratio=0.10)
display_inlier_outlier(uni_down_pcd, ind)

##아웃라이어(압흔)을 제외한 point cloud, 즉 철판 표면을 np로 저장
inlier_numpy = np.asarray(inlier.points)
trainX, trainY = np.split(inlier_numpy,[2], axis = 1)


##np로 저장된 즉 철판 표면 데이터를 선형회귀분석하여 평면을 찾음
line_fitter = LinearRegression()
line_fitter.fit(trainX, trainY)

##찾은 평면을 가시화하기 위해 평면위의 점 계산
results = line_fitter.predict(trainX)
points_on_plane = np.concatenate((trainX, results), axis=1)

##평면위의 점과 철판 표면 데이터를 병합
plane_and_inlier = np.concatenate((inlier_numpy, points_on_plane), axis=0)

##np를 pcd 형태로 변환하고 가시화
pai = o3d.geometry.PointCloud()
pai.points = o3d.utility.Vector3dVector(plane_and_inlier)
o3d.visualization.draw_geometries([pai])

#평면의 기울기만큼 데이터 보정

#기준설정 후 기준보다 작은 점들의 부피 계산


#pcd를 메쉬화
'''
alpha = 100
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd,alpha)
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh], mesh_show_back_face = True)
'''

'''
x = []
y = []
z = []

inputfile = "Y2_4KA.csv"
x_index = 0
y_index = 0
f = open(inputfile, 'r')
rdr = csv.reader(f)

#df = pd.read_csv(inputfile, header=None)

for line in rdr:
    for indentation in line:
        x.append(x_index)
        y.append(y_index)
        z.append(float(indentation))
        x_index = x_index + 1
    x_index = 0
    y_index = y_index + 1

x = np.array(x)
x.shape = (630000,1)
y = np.array(y)
y.shape = (630000,1)
z = np.array(z)
z.shape = (630000,1)


# Figure
fig = plt.figure(figsize = (8, 8))

# 3DAxes
ax = fig.add_subplot(111, projection='3d')

# Axes
ax.set_title("", size = 20)

#
ax.set_xlabel("x", size = 14, color = "r")
ax.set_ylabel("y", size = 14, color = "r")
ax.set_zlabel("z", size = 14, color = "r")

#
#ax.set_xticks([-5.0, -2.5, 0.0, 2.5, 5.0])
#ax.set_yticks([-5.0, -2.5, 0.0, 2.5, 5.0])

ax.scatter(x, y, z, s = 0.1, c = "blue")

plt.show()
'''

##포인트클라우드 및 노멀 가시화
'''
print("Load a ply point cloud, print it, and render it")
ply_point_cloud = o3d.data.PLYPointCloud()
pcd = o3d.io.read_point_cloud(ply_point_cloud.path)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
o3d.visualization.draw_geometries([pcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])
'''

##점을 메쉬화 하는 방법
'''
verts = np.array(
    [[-1, 0, 0], [0, 1, 0], [1, 0, 0], [0, -1, 0], [0, 0, 1]],
    dtype=np.float64,
)
triangles = np.array([[0, 1, 3], [1, 2, 3], [1, 3, 4]])
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(verts)
mesh.triangles = o3d.utility.Vector3iVector(triangles)
mesh.compute_vertex_normals()
mesh.rotate(
    mesh.get_rotation_matrix_from_xyz((np.pi / 4, 0, np.pi / 4)),
    center=mesh.get_center(),
)
o3d.visualization.draw(mesh, raw_mode=True)
'''

##가시화
#o3d.visualization.draw(pcd, raw_mode=True)
#o3d.visualization.draw_geometries([pcd], point_show_normal = True)
#o3d.visualization.draw_geometries([pcd])