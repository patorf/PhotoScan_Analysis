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
        """

        :rtype : MyPoint
        """
        if newPoint == None:
            newPoint = MyPoint()
        self.points.append(newPoint)
        return self.points[-1]

    def calc_sigma(self):
        # 'xy' -> Point
        # 'x,y' -> Sigma for x and y
        error_quad_sum = None
        count = 0
        error_quad_sum = PhotoScan.Vector([0, 0])
        errorMatrix = self.getErrorMatrix()

        # error_quad_sum.x += point.error_I.x ** 2
        #    error_quad_sum.y += point.error_I.y ** 2

        # count += 1

        # sigma_x = math.sqrt(error_quad_sum.x / count)
        #sigma_y = math.sqrt(error_quad_sum.y / count)

        cov = calc_Cov_from_ErrorMatrix(errorMatrix)
        print(cov)
        sigma_x = math.sqrt(cov[0, 0])
        sigma_y = math.sqrt(cov[1, 1])
        print(math.sqrt(sigma_y ** 2 + sigma_x ** 2))  #RMS in Pixel per Image
        #return (PhotoScan.Vector([sigma_x, sigma_y]), error_quad_sum, count)
        return (PhotoScan.Vector([sigma_x, sigma_y]))

    def getMax(self):
        errorMatrix = self.getErrorMatrix()

        maxError = PhotoScan.Vector((0, 0))

        maxError.x = max(abs(l[0]) for l in errorMatrix)
        maxError.y = max(abs(l[1]) for l in errorMatrix)

        return maxError

    def getErrorMatrix(self):
        errorMatrix = []
        for point in self.points:
            errorMatrix.append([point.error_I.x, point.error_I.y])
        return errorMatrix


    @classmethod
    def printReportHeader(cls):


        str = '{0:>12s}{1:>14s}{2:>9s}{3:>9s}{4:>9s}{5:>9s}{6:>9s}\n'.format('Cam #',
                                                                             'Projections',
                                                                             'SIG x',
                                                                             'SIG y',
                                                                             'SIG P',
                                                                             'MAX x',
                                                                             'MAX y'
                                                                             )

        return str

    def printReportLine(self, header=False):

        str = ''
        sigma = self.calc_sigma()
        maxError = self.getMax()
        str += '{:>12s}{:14d}{:9.5f}{:9.5f}{:9.5f}{:9.5f}{:9.5f}'.format(self.label,
                                                                         len(self.points),
                                                                         sigma.x,
                                                                         sigma.y,
                                                                         sigma.norm(),
                                                                         maxError.x,
                                                                         maxError.y)
        str += '\n'

        return str







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
        self.sigma_I = None


    def projectSigma_2_W(self, sigma_I=None):
        if not sigma_I:
            sigma_I = self.sigma_I
        # sigma_W is equal to the length of the error_W vector
        sigma_W = self.ratio_W_2_I * sigma_I

        trimFaktor = sigma_W / self.error_W.norm()
        return self.error_W * trimFaktor


    @property
    def error_I(self):
        return self.projection_I - self.measurement_I

    @property
    def ratio_W_2_I(self):
        return self.error_W.norm() / self.error_I.norm()


class MyGlobalPoint():
    def __init__(self):
        self.points = []
        self.cov_W = None
        self.sigma_W = None

        # def calcCov_W_from_Std(self):
        # if len(self.points) <= 2:
        #         return None
        #
        #     X_list = []
        #     summe1 = 0
        #     summe2 = 0
        #
        #     for point in self.points:
        #         assert isinstance(point, MyPoint)
        #         std_error_W = point.projectSigma_2_W()
        #
        #         X_list.append([std_error_W.x, std_error_W.y, std_error_W.z])
        #
        #     print('x_list', X_list)
        #     X_matrix = PhotoScan.Matrix(X_list)
        #
        #     C = X_matrix.t() * X_matrix
        #     C = C * (1 / (len(self.points) - 1))
        #
        #     self.cov_W = C




class MyProject():
    def __init__(self):
        self.photos = []
        self.points = defaultdict(MyGlobalPoint)


    def buildGlobalPointError(self):
        maxP = PhotoScan.Vector([0, 0, 0])
        minP = PhotoScan.Vector([0, 0, 0])
        for photo in self.photos:
            sigma_photo = photo.calc_sigma()
            assert isinstance(photo, MyPhoto)
            for point in photo.points:
                assert isinstance(point, MyPoint)
                maxP.x = max(maxP.x, point.coord_W.x)
                maxP.y = max(maxP.y, point.coord_W.y)
                maxP.z = max(maxP.z, point.coord_W.z)

                minP.x = min(minP.x, point.coord_W.x)
                minP.y = min(minP.y, point.coord_W.y)
                minP.z = min(minP.z, point.coord_W.z)

                point.sigma_I = sigma_photo

                self.points[point.track_id].points.append(point)

        print(minP, maxP)

    def calc_cov_for_all_points(self):
        pass
        # for trackid, point in self.points.items():
        #    point.calcCov_W_from_Std()

        # for point in list(self.points.values())[99].points:
        #pass

    # not needet by this point
    def calcGlobalSigma(self, photos=None):
        if not photos:
            photos = self.photos

        error_quad_sum = 0
        count = 0
        for photo in photos:
            sigma_photo, error_quad_sum_photo, count_photo = photo.calc_sigma()
            error_quad_sum += error_quad_sum_photo
            count += count_photo
        return (math.sqrt(error_quad_sum / count), count)


    def calc_reprojection(self, chunk):
        allPhotos = self.photos
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
            print(camera)
            for proj in projections[camera]:
                track_id = proj.track_id
                while point_index < npoints and points[point_index].track_id < track_id:
                    point_index += 1
                if point_index < npoints and points[point_index].track_id == track_id:
                    if not points[point_index].valid:
                        continue

                    point_W = points[point_index].coord
                    point_C = T.mulp(point_W)
                    point_I = calib.project(point_C)

                    measurement_I = proj.coord
                    measurement_C = calib.unproject(measurement_I)
                    error_I = calib.error(point_C, measurement_I)  # error = projection - measurement
                    # error_I_length = error_I.norm()

                    error_C = point_C - measurement_C * point_C.z
                    # error_C_length = error_C.norm()

                    measurement_W = camera.transform.mulp(measurement_C * point_C.z)
                    error_W = point_W - measurement_W
                    # error_W_length = error_W.norm()

                    # save Point in curren Photo
                    if point_I:
                        point = thisPhoto.addPoint()

                        point.track_id = track_id
                        point.projection_I = point_I
                        point.measurement_I = measurement_I
                        point.coord_C = point_C
                        point.coord_W = point_W
                        point.error_W = error_W
                        # print('ratio',point.ratio_W_2_I)
                        # print('disttocenter',point_C.norm())
                        # print('error_W', point.error_W)
                        #  print('error_I', point.error_I)
                        #  print('--------------W', point.coord_C)
                    # [-0.25211071968078613, -0.04763663187623024, 5.12844181060791])
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

    def printReport(self):
        str = ""
        str += MyPhoto.printReportHeader()
        for phots in self.photos:
            assert isinstance(phots, MyPhoto)
            str += phots.printReportLine()
        print(str)





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


def calc_Cov_from_ErrorMatrix(errorMatrix):
    # X_list = []
    #for error in pointError:
    #    X_list.append([error.x, error.y, error.z])

    X_matrix = PhotoScan.Matrix(errorMatrix)

    C = X_matrix.t() * X_matrix
    C = C * (1 / (len(errorMatrix) ))

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

    project = MyProject()
    total_error, ind_error, allPhotos = project.calc_reprojection(chunk)
    project.buildGlobalPointError()
    project.calc_cov_for_all_points()
    project.printReport()

    # print(total_error)
    # print(ind_error)
    # print(vars(allPhotos[0].points[1]))

    #covs_Dict = calc_Cov_4_allPoints(pointErrors_W)

    #point_cloud = chunk.point_cloud
    #points = point_cloud.points

    #export_No_xyz_cov(points, covs_Dict)
