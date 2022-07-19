import numpy as np
import open3d as o3d
import csv
import copy
import math
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

##csv파일 불러와서 list로 저장하기
IDTpointsList = []

#inputfile = "Y2_4KA.csv"
inputfile = "Y2_2.5KA.csv"
#inputfile = "H101_7.0KA_X.csv"
#inputfile = "H127_5.5KA_x.csv" #센서값 오류
x_index = 0
y_index = 0
row_number = 0
column_number = 0
f = open(inputfile, 'r')
rdr = csv.reader(f)

#range_x = (950, 1050)
#range_y = (100, 200)
range_x = (0, 2001)
range_y = (0, 400)

#X, Y 간격을 각각 1, 1000/315, 1000으로 가정
space_X = 1.0
space_Y = 1000.0/315.0
#Z좌표를 1000배
space_Z = 1000.0

last_indentation = 0;
for line in rdr:
    x_index = 0
    for indentation in line:
        if range_x[0] <= x_index and x_index <= range_x[1] and range_y[0] <= y_index and y_index <= range_y[1]:
            if math.fabs(float(indentation) + 100) < 0.01:
                print(x_index, y_index, "element is outlier.")
                IDTpoint = [x_index * space_X, y_index * space_Y, float(last_indentation) * space_Z]
                IDTpointsList.append(IDTpoint)
            else:
                IDTpoint = [x_index * space_X, y_index * space_Y, float(indentation) * space_Z]
                IDTpointsList.append(IDTpoint)
                last_indentation = indentation
        x_index = x_index + 1
    y_index = y_index + 1
column_number = x_index
row_number = y_index

##List를 np로 저장하기
pc_array = np.array(IDTpointsList, dtype=np.float32)

##np를 pcd 형태로 변환하기
pcd_all_points_raw = o3d.geometry.PointCloud()
pcd_all_points_raw.points = o3d.utility.Vector3dVector(pc_array)
o3d.visualization.draw_geometries([pcd_all_points_raw])

##노멀 계산
pcd_all_points_raw.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

##다운 샘플링(outlier를 찾기위한 시간을 단축)
pcd_all_points_down_sample = pcd_all_points_raw.uniform_down_sample(10)
o3d.visualization.draw_geometries([pcd_all_points_down_sample])

##RANSAC을 이용한 평면 찾기
plane_model, inliers = pcd_all_points_down_sample.segment_plane(distance_threshold=40.0,
                                         ransac_n=3,
                                         num_iterations=1000)
[a, b, c, d] = plane_model
print(f"Plane equation: {a}x + {b}y + {c}z + {d} = 0")

inlier_cloud = pcd_all_points_down_sample.select_by_index(inliers)
inlier_cloud.paint_uniform_color([1.0, 0, 0])
outlier_cloud = pcd_all_points_down_sample.select_by_index(inliers, invert=True)
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

z_vector = np.array([0,0,1], dtype=np.float32)
normal_vector = np.array([a,b,c], dtype=np.float32)
#normal_vector = np.array([0,1,0], dtype=np.float32)
rotation_axis = np.cross(normal_vector, z_vector, axisa=- 1 , axisb=- 1 , axisc=- 1 , axis=None)
#벡터 크기 계산하는 법 np.sqrt(x.dot(x)) 또는 np.linalg.norm(x)
mag_z_vector = np.sqrt(z_vector.dot(z_vector))
mag_normal_vector = np.sqrt(normal_vector.dot(normal_vector))
mag_rotation_axis = np.sqrt(rotation_axis.dot(rotation_axis))
rotation_angle = math.asin( mag_rotation_axis / (mag_normal_vector*mag_z_vector) )
rotation_axis = rotation_axis / mag_rotation_axis * rotation_angle

'''
##아웃라이어 찾기
def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

pcd_inlier, ind = pcd_all_points_down_sample.remove_statistical_outlier(nb_neighbors=1000, std_ratio=0.10)
display_inlier_outlier(pcd_all_points_down_sample, ind)

##아웃라이어만 가시화
o3d.visualization.draw_geometries([pcd_inlier])

##아웃라이어(압흔)을 제외한 point cloud, 즉 철판 표면을 np로 저장
np_inlier_points = np.asarray(pcd_inlier.points)
np_inlier_XY, np_inlier_Y = np.split(np_inlier_points, [2], axis = 1)

##np로 저장된 즉 철판 표면 데이터를 선형회귀분석하여 평면을 찾음
line_fitter = LinearRegression()
line_fitter.fit(np_inlier_XY, np_inlier_Y)
print(line_fitter.coef_, line_fitter.intercept_)

##찾은 평면을 가시화하기 위해 평면위의 점 계산
np_on_plane_Y = line_fitter.predict(np_inlier_XY)
np_on_plane_points = np.concatenate((np_inlier_XY, np_on_plane_Y), axis=1)

##평면위의 점을 pcd형태로 변환하고 가시화
pcd_on_plane_points = o3d.geometry.PointCloud()
pcd_on_plane_points.points = o3d.utility.Vector3dVector(np_on_plane_points)
o3d.visualization.draw_geometries([pcd_on_plane_points])

##평면위의 점과 철판 표면 데이터를 병합
np_inlier_and_plane = np.concatenate((np_inlier_points, np_on_plane_points), axis=0)

##병합한 데이터를 pcd 형태로 변환하고 가시화
pcd_inlier_and_plane = o3d.geometry.PointCloud()
pcd_inlier_and_plane.points = o3d.utility.Vector3dVector(np_inlier_and_plane)
o3d.visualization.draw_geometries([pcd_inlier_and_plane])
'''

#평면의 기울기만큼 데이터 보정(다운샘플링한 점들만 대상)
pcd_all_points_down_sample_r = copy.deepcopy(pcd_all_points_down_sample)
R = pcd_all_points_down_sample_r.get_rotation_matrix_from_axis_angle(rotation_axis)
pcd_all_points_down_sample_r.rotate(R, pcd_all_points_down_sample.get_center() )
o3d.visualization.draw_geometries([pcd_all_points_down_sample, pcd_all_points_down_sample_r])

##RANSAC을 이용한 평면 찾기(검산)
plane_model, inliers = pcd_all_points_down_sample_r.segment_plane(distance_threshold=40.0,
                                         ransac_n=3,
                                         num_iterations=1000)
[a, b, c, d] = plane_model
print(f"Plane equation: {a}x + {b}y + {c}z + {d} = 0")

inlier_cloud = pcd_all_points_down_sample_r.select_by_index(inliers)
inlier_cloud.paint_uniform_color([1.0, 0, 0])
outlier_cloud = pcd_all_points_down_sample_r.select_by_index(inliers, invert=True)
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


#평면의 기울기만큼 데이터 보정(모든 점 대상)
pcd_all_points_r = copy.deepcopy(pcd_all_points_raw)
R = pcd_all_points_r.get_rotation_matrix_from_axis_angle(rotation_axis)
pcd_all_points_r.rotate(R, pcd_all_points_raw.get_center() )
o3d.visualization.draw_geometries([pcd_all_points_raw, pcd_all_points_r])

#point cloud를 X, Y, Z좌표별로 2차원 numpy 배열로 변환
np_all_points_r = np.asarray(pcd_all_points_r.points)
np_all_points_r_X = np_all_points_r[:,0]
np_all_points_r_Y = np_all_points_r[:,1]
np_all_points_r_Z = np_all_points_r[:,2]
np_all_points_r_X=np_all_points_r_X.reshape(row_number, column_number)
np_all_points_r_Y=np_all_points_r_Y.reshape(row_number, column_number)
np_all_points_r_Z=np_all_points_r_Z.reshape(row_number, column_number)

#최대 Z값을 가진 행, 열을 찾기
index_max = np_all_points_r_Z.argmax()

def calc_row_column(_index, _row_number, _column_number):
    row, column = divmod(_index, _column_number)
    return row, column

row, column = calc_row_column(index_max, row_number, column_number)

#최대 Z값을 가진 열을 추출
np_center_line = np_all_points_r_Z[row, :]

#최개 Z값을 가진 열의 기울기를 계산하기 위하여 점 100개를 묶어 smoothing
np_center_line_smoothed = savgol_filter(np_center_line, 100, 3)  # window size 51, polynomial order 3

#최개 Z값을 가진 열의 기울기를 계산
diff_np_center_line = np.delete(np_center_line_smoothed,0) - np.delete(np_center_line_smoothed,column_number-1)

plt.plot(np_center_line)
plt.show()

plt.plot(np_center_line_smoothed)
plt.show()

plt.plot(diff_np_center_line)
plt.show()

#기울기의 전체 평균값과, 최대 기울기의 중간값을 계산(alpha가 파라메터) = 압흔의 시작부분을 찾는 기준
alpha = 0.5
mean_of_diff = np.mean(diff_np_center_line)
max_of_diff = np.max(diff_np_center_line)
min_of_diff = np.min(diff_np_center_line)
mid_of_diff_positive = (1 - alpha) * mean_of_diff + alpha * max_of_diff
mid_of_diff_negative = (1 - alpha) * mean_of_diff + alpha * min_of_diff


#압흔이 시작하고 끝나는 지점의 포인트를 찾기.
indentation_start_index = 0
indentation_end_index = 0
number_of_mid_of_diff_positive = 0
number_of_mid_of_diff_negative = 0
for index in range(len(diff_np_center_line)-1):
    if diff_np_center_line[index] < mid_of_diff_positive and diff_np_center_line[index + 1] > mid_of_diff_positive:
        indentation_start_index = index
        number_of_mid_of_diff_positive = number_of_mid_of_diff_positive + 1
    if diff_np_center_line[index] > mid_of_diff_positive and diff_np_center_line[index + 1] < mid_of_diff_positive:
        number_of_mid_of_diff_positive = number_of_mid_of_diff_positive + 1

    if diff_np_center_line[index] > mid_of_diff_negative and diff_np_center_line[index + 1] < mid_of_diff_negative:
        number_of_mid_of_diff_negative = number_of_mid_of_diff_negative + 1
    if diff_np_center_line[index] < mid_of_diff_negative and diff_np_center_line[index + 1] > mid_of_diff_negative:
        indentation_end_index = index
        number_of_mid_of_diff_negative = number_of_mid_of_diff_negative + 1

indentation_start_Z = np_center_line[indentation_start_index]
indentation_end_Z = np_center_line[indentation_end_index]

indentation_start_point = np.asarray(
    (np_all_points_r_X[row, indentation_start_index],
    np_all_points_r_Y[row, indentation_start_index],
    np_all_points_r_Z[row, indentation_start_index]))
indentation_end_point = np.asarray(
    (np_all_points_r_X[row, indentation_end_index],
    np_all_points_r_Y[row, indentation_end_index],
    np_all_points_r_Z[row, indentation_end_index]))

if number_of_mid_of_diff_positive != 2 or number_of_mid_of_diff_negative != 2:
    print("기울기 설정을 다시 하세요.")

#압흔이 시작하고 끝나는 지점 가시화.
plt.plot(diff_np_center_line)
plt.axhline(mid_of_diff_positive, color='lightgray', linestyle='--')
plt.axhline(mid_of_diff_negative, color='gray', linestyle='solid')
plt.show()

plt.plot(np_center_line)
plt.axhline(indentation_start_Z, color='lightgray', linestyle='--')
plt.axhline(indentation_end_Z, color='gray', linestyle='solid')
plt.show()

plt.plot(np_center_line_smoothed)
plt.axhline(indentation_start_Z, color='lightgray', linestyle='--')
plt.axhline(indentation_end_Z, color='gray', linestyle='solid')
plt.show()

#압흔의 반지름 밑 센터 찾기(현재는 사용안함)
radius = np.linalg.norm(indentation_end_point - indentation_start_point)
center = (indentation_end_point + indentation_start_point)/2

#압흔을 포함하는 가로세로 여유 사이즈 결정
margin_X = 50.0
margin_Y = 50.0

margin_index_X = int(margin_X/space_X)
margin_index_Y = int((radius+margin_Y) / space_Y)

#가로세로 여유 사이즈를 포함하는 분석범위(인덱스) 계산
analysis_region_X = (indentation_start_index - margin_index_X, indentation_end_index + margin_index_X)
analysis_region_Y = (row - margin_index_Y , row + margin_index_Y )

#분석범위만 추출
np_analysis_region_X = np_all_points_r_X[analysis_region_Y[0]:analysis_region_Y[1], analysis_region_X[0]:analysis_region_X[1]]
np_analysis_region_Y = np_all_points_r_Y[analysis_region_Y[0]:analysis_region_Y[1], analysis_region_X[0]:analysis_region_X[1]]
np_analysis_region_Z = np_all_points_r_Z[analysis_region_Y[0]:analysis_region_Y[1], analysis_region_X[0]:analysis_region_X[1]]

np_analysis_region_X = np_analysis_region_X.reshape((analysis_region_X[1]-analysis_region_X[0])*(analysis_region_Y[1]-analysis_region_Y[0]), 1)
np_analysis_region_Y = np_analysis_region_Y.reshape((analysis_region_X[1]-analysis_region_X[0])*(analysis_region_Y[1]-analysis_region_Y[0]), 1)
np_analysis_region_Z = np_analysis_region_Z.reshape((analysis_region_X[1]-analysis_region_X[0])*(analysis_region_Y[1]-analysis_region_Y[0]), 1)
np_analysis_region_points = np.concatenate([np_analysis_region_X, np_analysis_region_Y, np_analysis_region_Z], axis=1)

pcd_analysis_region_points = o3d.geometry.PointCloud()
pcd_analysis_region_points.points = o3d.utility.Vector3dVector(np_analysis_region_points)
o3d.visualization.draw_geometries([pcd_analysis_region_points])

plt.scatter(np_analysis_region_X, np_analysis_region_Y, s=0.1, c = "black")
plt.show()


#압흔이 시작된 점


a = 0






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