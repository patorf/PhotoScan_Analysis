__author__ = 'philipp.atorf'

import math
import PhotoScan
from collections import defaultdict
from math import sqrt
import warnings

class MyPhoto():

    def __init__(self, label=None):

        self.label = label
        self.points = []

    def addPoint(self, newPoint=None):
        if newPoint == None:
            newPoint = MyPoint()
        self.points.append(newPoint)
        return self.points[-1]

    def calc_sigma(self):
        error_quad_sum = 0
        count = 0
        for point in self.points:
            error_quad_sum += point.error_I.norm ** 2
            count += 1

        return (math.sqrt(error_quad_sum / count), error_quad_sum, count)


class MyPoint():
    def __init__(self, projection_I=None,
                 measurement_I=None,
                 track_id=None,
                 coord_W=None,
                 coord_C=None,
                 error_W=None,
                 ratio_I_2_W=None):

        self.projection_I = projection_I
        self.measurement_I = measurement_I
        self.track_id = track_id
        self.coord_W = coord_W
        self.coord_C = coord_C
        self.error_W = error_W
        self.ratio_I_2_W = ratio_I_2_W

class MyProject():
    def __init__(self):
        self.photos = []


    def calcGlobalSigma(self, photos=None):
        if not photos:
            photos=self.photos

        error_quad_sum = 0
        count = 0
        for photo in photos:
            sigma_photo, error_quad_sum_photo, count_photo = photo.calc_sigma()
            error_quad_sum += error_quad_sum_photo
            count += count_photo
        return (math.sqrt(error_quad_sum / count), count)


def calc_reprojection(chunk):
    allPhotos = []
    point_cloud = chunk.point_cloud

    points = point_cloud.points
    npoints = len(points)
    projections = chunk.point_cloud.projections

    err_sum = 0
    num = 0

    photo_avg = {}

    for camera in chunk.cameras:
        if not camera.transform:
            continue

        thisPhoto = MyPhoto(camera.label)
        allPhotos.append(thisPhoto)

        T = camera.transform.inv()
        calib = camera.sensor.calibration

        point_index = 0

        photo_num = 0
        photo_err = 0

        for proj in projections[camera]:
            track_id = proj.track_id
            while point_index < npoints and points[point_index].track_id < track_id:
                point_index += 1
            if point_index < npoints and points[point_index].track_id == track_id:
                if not points[point_index].valid:
                    continue

                # print (calib.project(T.mulp(points[point_index].coord)))#Pixel Coordinates
                # print (proj.coord) #Position of Sift Operator in Pixel
                point_C = T.mulp(points[point_index].coord)
                # print (points[point_index].coord)
                # pointonImage = calib.project(pointVec)

                error_I = calib.error(
                    T.mulp(points[point_index].coord), proj.coord)

                # Berechnung des Verbesserungsvektors im Kamerasystem
                # point_C ist der 2D Verbesserungsvektor relativ zum Mittelpunkt
                # im 3D Camerasystem (in Entfernung des Punktes)
                # center_C ist der Mittelpunkt des Camerasystems in Entfernung des Punkte
                # der Verbesserungsvektor V = point_C - center_C
                #
                point_C, center_C = trans_error_image_2_camera(
                    camera, error_I, point_C)

                point_W = camera.transform.mulp(point_C)
                center_W = camera.transform.mulp(center_C)
                error_W = point_W - center_W

                pointErrors_I[track_id].append(error_I)

                pointErrors_W[track_id].append(error_W)
                # test
                #
                # PointW = PhotoScan.Vector((20,30,40))
                # PointI = T.mulp(PointW)
                #
                # PointWback = camera.transform.mulp(PointI)
                # print (PointWback) # 20,40,40
                #
                #
                point = thisPhoto.addPoint()
                point.track_id = track_id
                point.projection_I = calib.project(
                    T.mulp(points[point_index].coord))
                point.measurement_I = proj.coord
                point.coord_C = point_C
                point.coord_W = point_W
                point.error_W = error_W
                point.ratio_I_2_W = error_I.norm() / error_W.norm()

                #
                #

                dist = error_I.norm() ** 2

                err_sum += dist
                num += 1

                photo_num += 1
                photo_err += dist

                photo_avg[camera.label] = (
                    math.sqrt(photo_err / photo_num), photo_num)

        sigma = math.sqrt(err_sum / num)

    rep_avg = sigma

    return (rep_avg, photo_avg, allPhotos)


def trans_error_image_2_camera(camera, point_pix, point_Camera):
    T = camera.transform
    calib = camera.sensor.calibration
    fx = calib.fx
    fy = calib.fy
    u = point_pix.x
    v = point_pix.y

    x = u / fx  # -calib.cx/fx # den hinteren term entfernen
    y = v / fy  # -calib.cy/fy

    center_C = PhotoScan.Vector((0, 0, 1)) * point_Camera.z
    point_C = PhotoScan.Vector((x, y, 1)) * point_Camera.z

    return point_C, center_C


def calc_Cov_4_Point(pointError):
    X_list = []
    summe = 0
    for error in pointError:
        X_list.append([error.x, error.y, error.z])

    X_matrix = PhotoScan.Matrix(X_list)

    C = X_matrix.t() * X_matrix
    C = C * (1 / (len(pointError) - 1))

    return C


def calc_Cov_4_allPoints(pointList):
    covs = {}  # Key = trackid ; value = 3x3 Matrix

    for track_id, error in pointList.items():
        if len(error) > 3:
            cov = calc_Cov_4_Point(error)
            covs[track_id] = cov
        else:
            pass

    return covs


def creatExportList(points, covs_Dict):
    exportPoints = []
    for point in points:
        if covs_Dict.get(point.track_id):
            exportPoints.append(
                [point.track_id, point.coord, covs_Dict[point.track_id]])
    return exportPoints


def export_No_xyz_cov(points, covs_Dict):
    exportPoints = creatExportList(points, covs_Dict)

    doc.path  # C:\User....\project.psz

    f = open(
        'C:\\Users\\philipp.atorf.INTERN\\Downloads\\building\\export.txt', 'w')

    print('output start for %i points' % len(exportPoints))

    for point in exportPoints:
        output = ''
        output += '%i;' % point[0]
        output += '%15.12e;%15.12e;%15.12e\n' % (
            point[1].x, point[1].y, point[1].z)
        output += '%15.12e;%15.12e;%15.12e\n' % (
            point[2].row(0).x, point[2].row(0).y, point[2].row(0).z)
        output += '%15.12e;%15.12e;%15.12e\n' % (
            point[2].row(1).x, point[2].row(1).y, point[2].row(1).z)
        output += '%15.12e;%15.12e;%15.12e' % (
            point[2].row(2).x, point[2].row(2).y, point[2].row(2).z)

        if point != exportPoints[-1]:
            output += '\n'
        f.write(output)

    f.close()
    print('output finish')


def export_No_xyz_std(points, covs_Dict):
    exportPoints = creatExportList(points, covs_Dict)
    f = open(
        'C:\\Users\\philipp.atorf.INTERN\\Downloads\\building\\export_xyz.txt', 'w')

    print('output xyz sx sy sz start for %i points' % len(exportPoints))

    for point in exportPoints:
        output = ''
        output += '%i;' % point[0]
        output += '%15.12e;%15.12e;%15.12e;%15.12e' % (
            point[1].x,
            point[1].y,
            point[1].z,
            sqrt(point[2].row(0).x + point[2].row(1).y + point[2].row(2).z))

        if point != exportPoints[-1]:
            output += '\n'
        f.write(output)

    f.close()
    print('output finish')


if __name__ == '__main__':
    testPointError = [PhotoScan.Vector((1, 2, 1.4)), PhotoScan.Vector(
        (-1.2, 1, 2.3)), PhotoScan.Vector((-1.4, 2, 3))]
    # print (calc_Cov_4_Point(testPointError))


    ### Programm Start ###

    pointErrors_W = defaultdict(list)
    pointErrors_I = defaultdict(list)

    doc = PhotoScan.app.document
    chunk = doc.chunk

    total_error, ind_error, allPhotos = calc_reprojection(chunk)
    print(total_error)
    print(ind_error)
    print(vars(allPhotos[0].points[1]))

    #covs_Dict = calc_Cov_4_allPoints(pointErrors_W)

    #point_cloud = chunk.point_cloud
    #points = point_cloud.points

    #export_No_xyz_cov(points, covs_Dict)
