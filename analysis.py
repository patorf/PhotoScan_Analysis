import copy
import os
import re
import sys
from STL_Writer import Binary_STL_Writer

__author__ = 'philipp.atorf'

import math
from collections import defaultdict
from math import sqrt

import PhotoScan
import svd
from pysvg.builders import *
import pysvg
import imp
from csg import *


imp.reload(pysvg)
from pysvg.builders import *


class I3_Photo(object):
    def __init__(self, label=None):

        self.label = label
        self.points = []
        """:type : list of I3_Point"""
        self.photoscanCamera = None
        self.sigma = None

    def add_point(self, new_point=None):
        """

        :rtype : I3_Point
        """
        if new_point is None:
            new_point = I3_Point()
        self.points.append(new_point)
        return self.points[-1]

    def calc_sigma(self):
        if self.sigma is None:
            # 'xy' -> Point
            # 'x,y' -> Sigma for x and y
            error_quad_sum = None
            count = 0
            error_quad_sum = PhotoScan.Vector([0, 0])
            error_matrix = self.get_error_matrix()

            # error_quad_sum.x += point.error_I.x ** 2
            # error_quad_sum.y += point.error_I.y ** 2

            # count += 1

            # sigma_x = math.sqrt(error_quad_sum.x / count)
            # sigma_y = math.sqrt(error_quad_sum.y / count)

            cov = calc_Cov_from_ErrorMatrix(error_matrix)
            sigma_x = math.sqrt(cov[0, 0])
            sigma_y = math.sqrt(cov[1, 1])
            # return (PhotoScan.Vector([sigma_x, sigma_y]), error_quad_sum, count)
            self.sigma = PhotoScan.Vector([sigma_x, sigma_y])
        return self.sigma


    def get_max(self):
        error_matrix = self.get_error_matrix()

        max_error = PhotoScan.Vector((0, 0))

        max_error.x = max(abs(l[0]) for l in error_matrix)
        max_error.y = max(abs(l[1]) for l in error_matrix)

        return max_error

    def get_error_matrix(self):
        error_matrix = []
        for point in self.points:
            error_matrix.append([point.error_I.x, point.error_I.y])
        return error_matrix


    @classmethod
    def print_report_header(cls):


        r_str = '{0:>12s}{1:>14s}{2:>9s}{3:>9s}{4:>9s}{5:>9s}{6:>9s}\n'.format('Cam #',
                                                                               'Projections',
                                                                               'SIG x',
                                                                               'SIG y',
                                                                               'SIG P',
                                                                               'MAX x',
                                                                               'MAX y'
                                                                               )

        return r_str

    def print_report_line(self):

        r_str = ''
        sigma = self.calc_sigma()
        max_error = self.get_max()
        r_str += '{:>12s}{:14d}{:9.5f}{:9.5f}{:9.5f}{:9.5f}{:9.5f}\n'.format(self.label,
                                                                             len(self.points),
                                                                             sigma.x,
                                                                             sigma.y,
                                                                             sigma.norm(),
                                                                             max_error.x,
                                                                             max_error.y)

        return r_str


class I3_Point():
    def __init__(self,
                 projection_I=None,
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
        self.measurement_C = None
        self.sigma_I = None


    def project_sigma_2_W(self, sigma_I=None):
        if not sigma_I:
            sigma_I = self.sigma_I
        # sigma_W is equal to the length of the error_W vector
        sigma_W = self.ratio_W_2_I * sigma_I

        trim_faktor = sigma_W / self.error_W.norm()
        return self.error_W * trim_faktor


    @property
    def error_I(self):
        return self.projection_I - self.measurement_I

    @property
    def ratio_W_2_I(self):
        return self.error_W.norm() / self.error_I.norm()


class I3_GlobalPoint():
    def __init__(self):
        self.points = []
        self.cov_W = None
        self.sigma_W = None

        # def calcCov_W_from_Std(self):
        # if len(self.points) <= 2:
        # return None
        #
        # X_list = []
        # summe1 = 0
        # summe2 = 0
        #
        # for point in self.points:
        # assert isinstance(point, I3_Point)
        # std_error_W = point.projectSigma_2_W()
        #
        # X_list.append([std_error_W.x, std_error_W.y, std_error_W.z])
        #
        # print('x_list', X_list)
        # X_matrix = PhotoScan.Matrix(X_list)
        #
        # C = X_matrix.t() * X_matrix
        # C = C * (1 / (len(self.points) - 1))
        #
        # self.cov_W = C


class I3_Project():
    def __init__(self):
        self.photos = []
        """:type: list[I3_Photo]"""

        # self.points = defaultdict(I3_GlobalPoint)
        self.point_photo_reference = {}
        """:type: dict[int, list[I3_Photo]]"""

        self.path = PhotoScan.app.document.path
        self.directory = "\\".join(self.path.split('\\')[:-1])

    def get_point_photos_reference(self):

        if not self.point_photo_reference:
            points_photo_dict = self.point_photo_reference
            for photo in self.photos:
                for point in photo.points:
                    if point.track_id in points_photo_dict:
                        points_photo_dict[point.track_id].append(photo)
                    else:
                        points_photo_dict[point.track_id] = []
                        points_photo_dict[point.track_id].append(photo)

        return self.point_photo_reference

    def export_for_OpenScad(self, filename='openScad'):
        filename += ".scad"
        adjustment = peseudo_3D_intersection_adjustment(self.get_point_photos_reference())

        ellipsoid_parameter_list = adjustment.get_all_for_ellipsoid()
        output_str = "factor = 0.051;\n"
        for ellipsoid_parameter in ellipsoid_parameter_list:
            eig_val, eig_vec, pos = ellipsoid_parameter
            output_str += Py_2_OpenScad.errorEllipse_from_eig(eig_vec, eig_val, pos)

        f = open(self.directory + '\\' + filename, 'w')
        f.write(output_str)
        f.close()
        print('save file ', filename, ' to: ', self.directory)

    def export_STL(self, filename='stl_export.stl', binary=True, factor=0.1):
        print('start output STL-File with factor {:4.2f}: '.format(factor) + filename)

        adjustment = peseudo_3D_intersection_adjustment(self.get_point_photos_reference())

        ellipsoid_parameter_list = adjustment.get_all_for_ellipsoid()
        output_str = "solid Ellipsoids\n"

        stl_handler = STL_Handler()

        if not binary:

            for ellipsoid_parameter in ellipsoid_parameter_list:
                eig_val, eig_vec, pos = ellipsoid_parameter
                output_str += stl_handler.create_ellipsoid_stl(eig_vec, eig_val, pos, factor, False)
                ## output_str += Py_2_OpenScad.errorEllipse_from_eig(eig_vec, eig_val, pos)
            output_str += "endsolid Ellipsoids"
            f = open(self.directory + '\\' + filename, 'w')
            f.write(output_str)
            f.close()
            print('save file ', filename, ' to: ', self.directory)

        else:
            data = []
            for ellipsoid_parameter in ellipsoid_parameter_list:
                eig_val, eig_vec, pos = ellipsoid_parameter
                data.extend(stl_handler.create_ellipsoid_stl(eig_vec, eig_val, pos, factor, True))
            with open(self.directory + '\\' + filename, 'wb') as fp:
                writer = Binary_STL_Writer(fp)
                writer.add_faces(data)
                writer.close()
                print('save bin file ', filename, ' to: ', self.directory)

    def build_global_point_error(self):

        max_P = PhotoScan.Vector([0, 0, 0])
        min_P = PhotoScan.Vector([0, 0, 0])
        for photo in self.photos:
            sigma_photo = photo.calc_sigma()
            assert isinstance(photo, I3_Photo)
            for point in photo.points:
                assert isinstance(point, I3_Point)
                max_P.x = max(max_P.x, point.coord_W.x)
                max_P.y = max(max_P.y, point.coord_W.y)
                max_P.z = max(max_P.z, point.coord_W.z)

                min_P.x = min(min_P.x, point.coord_W.x)
                min_P.y = min(min_P.y, point.coord_W.y)
                min_P.z = min(min_P.z, point.coord_W.z)

                point.sigma_I = sigma_photo

                self.points[point.track_id].points.append(point)


    def calc_cov_for_all_points(self):
        pass
        # for trackid, point in self.points.items():
        # point.calcCov_W_from_Std()

        # for point in list(self.points.values())[99].points:
        # pass


    # not needet by this point
    def get_RMS_4_all_photos(self, photos=None):
        if not photos:
            photos = self.photos

        var_x_sum = 0
        var_y_sum = 0
        for photo in photos:
            sigma_photo = photo.calc_sigma()
            var_x_sum += sigma_photo.x ** 2
            var_y_sum += sigma_photo.y ** 2

        rms_x = math.sqrt(var_x_sum / len(photos))
        rms_y = math.sqrt(var_y_sum / len(photos))

        return rms_x, rms_y


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

            this_photo = I3_Photo(camera.label)
            this_photo.photoscanCamera = camera
            allPhotos.append(this_photo)

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
                        point = this_photo.add_point()

                        point.track_id = track_id
                        point.projection_I = point_I
                        point.measurement_I = measurement_I
                        point.coord_C = point_C
                        point.coord_W = point_W
                        point.error_W = error_W
                        point.measurement_C = measurement_C
                        # print('ratio',point.ratio_W_2_I)
                        # print('disttocenter',point_C.norm())
                        # print('error_W', point.error_W)
                        # print('error_I', point.error_I)
                        # print('--------------W', point.coord_C)
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

        return rep_avg, photo_avg, allPhotos

    def print_report(self):
        filename = 'report.txt'

        r_str = ""
        r_str += I3_Photo.print_report_header()
        for phots in self.photos:
            assert isinstance(phots, I3_Photo)
            r_str += phots.print_report_line()

        r_str += '\n'
        rms_x, rms_y = self.get_RMS_4_all_photos()
        r_str += '{:>26s}{:9.5f}{:9.5f}'.format('RMS:', rms_x, rms_y)

        print(r_str)

        f = open(self.directory + '\\' + filename, 'w')
        f.write(r_str)
        f.close()
        print('save file ', filename, ' to: ', self.directory)


    def create_project_SVG(self, error_factor=40, cols=20):

        filename = 'imageMeasurements.svg'

        s = svg()

        summery_SVG = SVG_Photo_Representation(self.photos)
        summery_SVG.point_radius = 1

        summery, height = summery_SVG.get_raw_error_vector_svg(factor=error_factor)

        s.addElement(summery)
        summery_group = g()
        summery_error_raster, height = summery_SVG.get_raw_error_vector_svg(True, factor=error_factor, cols=cols)
        summery_count_raster = summery_SVG.get_raster_count_svg(cols)

        legend = summery_SVG.count_legend

        trans_legend = TransformBuilder()
        trans_legend.setTranslation(605, 20)
        legend.set_transform(trans_legend.getTransform())

        summery_group.addElement(summery_count_raster)
        summery_group.addElement(summery_error_raster)
        summery_group.addElement(legend)

        # Group Transformation
        trans_raster = TransformBuilder()
        trans_raster.setTranslation(700, 0)
        summery_group.set_transform(trans_raster.getTransform())

        s.addElement(summery_group)

        totol_height = height
        i = 1

        for photo in self.photos:
            svg_photo = SVG_Photo_Representation([photo])

            photoSVG_group, group_height = svg_photo.get_raw_error_vector_svg(factor=error_factor)

            # i= add_2_summery_photo(s,photoSVG_group,i)

            # Group Transformation
            trans = TransformBuilder()
            trans.setTranslation(0, group_height * i)
            photoSVG_group.set_transform(trans.getTransform())

            s.addElement(photoSVG_group)
            totol_height += group_height
            i += 1

        s.set_height(totol_height)

        s.save(self.directory + '\\' + filename)
        print('save file ', filename, ' to: ', self.directory)


class X_vector_element():
    paramerter_type_point = 'point'
    paramerter_type_cam = 'cam'
    value_type_X = 'X'
    value_type_Y = 'Y'
    value_type_Z = 'Z'
    value_type_R = 'R'

    def __init__(self, parameter_type, value_type, value, id):
        self.value_type = value_type
        self.parameter_type = parameter_type
        self.value = value
        self.id = id

    def __str__(self):

        if self.value_type == self.value_type_R:
            return "{:s} {:s} :{:s} id:{:s}".format(self.parameter_type,
                                                    self.value_type,
                                                    str(self.value),
                                                    str(self.id))
        else:
            return "{:s} {:s} :{:.9f} id:{:s}".format(self.parameter_type,
                                                      self.value_type,
                                                      self.value,
                                                      str(self.id))


class L_vector_element():
    value_type_x = 'x'
    value_type_y = 'y'

    def __init__(self, cam_id, track_id, value_type, value, sigma):
        self.cam_id = cam_id
        self.track_id = track_id
        self.value_type = value_type
        self.value = value
        self.sigma = sigma  # varianz

    def __str__(self):
        return "{:s} track_id:{:d} value_type:{:s} value:{:.9f} sigam:{:.9f} ".format(self.cam_id,
                                                                                      self.track_id,
                                                                                      self.value_type,
                                                                                      self.value,
                                                                                      self.sigma)


class peseudo_3D_intersection_adjustment():
    measurment_x = 'x'
    measurment_y = 'y'

    rotation = 'R'
    point_X = 'X'
    point_Y = 'Y'
    point_Z = 'Z'

    cam_X = 'X_0'
    cam_Y = 'Y_0'
    cam_Z = 'Z_0'

    def __init__(self, point_with_reference=None):

        """

        :type point_with_reference:  dict[int, list[I3_Photo]]
        """
        self.points = point_with_reference
        self.points_pos = {}

    @staticmethod
    def get_eigen_vel_vec(m):
        """
        return a tuble of eigenvalue and eigenvectro for a given Matrix using SVD
        the eigenvalue are not sorted!!!

        :param m: PhotoScan.Matrix
        :rtype : (list[float],list[list[float]])
        """
        rows, cols = m.size
        m_list = []

        for r in range(0, rows):
            new_row_for_list = []
            for col in list(m.row(r)):
                new_row_for_list.append(col)
            m_list.append(new_row_for_list)

        s, v, d = svd.svd(m_list)
        eigenvalues = v
        eigenvector = s  # PhotoScan.Matrix(s)

        # sorted_indeces =sorted(range(len(eigenvalues)), key=lambda k: v[k])

        return eigenvalues, eigenvector

    def get_all_for_ellipsoid(self, track_id=None):
        """
        returns a list of a tuple of Eigenvalue, Eigenvector and Position for
        a track_id. if no id is passed it returns a list of all points

        :type track_id: int
        :param track_id: track_id or None. i
        :rtype : (list[float],list[list[float]],list[float])
        """

        if track_id:
            list_of_track_ids = [track_id]
        else:
            list_of_track_ids = self.points.keys()

        return_list = []

        for track_id in list_of_track_ids:
            cov = self.get_cov_for_point(track_id)

            eig_val, eig_vec = self.get_eigen_vel_vec(cov)
            pos_vector = None
            for point in self.points[track_id][0].points:
                if point.track_id == track_id:
                    pos_vector = point.coord_W
            # pos_vector = self.points[track_id][0].points[track_id].coord_W
            pos = [pos_vector.x, pos_vector.y, pos_vector.z]
            return_list.append((eig_val, eig_vec, pos))
        return return_list


    def get_measurment_vector_4_track_id(self, point_photo_reference, track_id):
        """

        :type photos: list of I3_Photo
        """

        # for track_id,photos in point_photo_reference.items():
        photos = point_photo_reference[track_id]
        measurement_list = []
        for photo in photos:
            for point in photo.points:
                if point.track_id == track_id:
                    assert isinstance(photo, I3_Photo)
                    measurement_list.append((photo.label, point.measurement_I))
        return track_id, measurement_list

    def get_cov_for_point(self, track_id):
        """


        object
        :param track_id: id of a 3D Point
        :return: 3x3 Cov-Matrix
        :rtype : list[list[float]]
        """
        jacobian_matrix, X_vector, L_vector = self.get_jacobian(track_id)

        A = jacobian_matrix
        P = self.get_P_matrix(L_vector, 0.1)
        N = A.t() * P * A
        Qxx = N.inv()

        return Qxx

    def get_P_matrix(self, L_vector, sigma0=1):
        """

        :type L_vector: list of L_vector_element
        """
        K_ll_diag = []
        for L_element in L_vector:
            k_l = L_element.sigma ** 2
            K_ll_diag.append(k_l)
        K_ll = PhotoScan.Matrix.diag(K_ll_diag)
        Q_ll = 1 / sigma0 ** 2 * K_ll
        # Invers is only allowd for 4x4 Matrix. Invers of diag-matrix is 1/A[i,i]
        for i in range(0, Q_ll.size[0]):
            Q_ll[i, i] = 1 / Q_ll[i, i]
        P = Q_ll
        return P


    def get_jacobian(self, track_id, point_photo_reference=None):
        if point_photo_reference == None:
            point_photo_reference = self.points
        photos = point_photo_reference[track_id]
        X_vector = []
        L_vectro = []
        jacobian = []
        for photo in photos:
            """:type photo: I3_Photo"""
            assert isinstance(photo, I3_Photo)
            X_to_optimize = [self.point_X, self.point_Y, self.point_Z]

            X_vector_for_cam = []

            L_vector_for_cam = []

            paramerter_type = X_vector_element.paramerter_type_cam
            R_t = photo.photoscanCamera.transform

            R = PhotoScan.Matrix([[R_t[0, 0], R_t[0, 1], R_t[0, 2]],
                                  [R_t[1, 0], R_t[1, 1], R_t[1, 2]],
                                  [R_t[2, 0], R_t[2, 1], R_t[2, 2]]])
            cam_R = X_vector_element(paramerter_type,
                                     X_vector_element.value_type_R,
                                     R,
                                     photo.label)
            cam_X = X_vector_element(paramerter_type, X_vector_element.value_type_X, photo.photoscanCamera.center.x,
                                     photo.label)
            cam_Y = X_vector_element(paramerter_type, X_vector_element.value_type_Y, photo.photoscanCamera.center.y,
                                     photo.label)
            cam_Z = X_vector_element(paramerter_type, X_vector_element.value_type_Z, photo.photoscanCamera.center.z,
                                     photo.label)

            X_vector_for_cam.extend([cam_X, cam_Y, cam_Z, cam_R])

            # point = photo.points[0]


            for point in photo.points:
                if point.track_id == track_id:
                    self.points_pos[track_id] = point.coord_W
                    paramerter_type = X_vector_element.paramerter_type_point
                    point_X = X_vector_element(paramerter_type, X_vector_element.value_type_X, point.coord_W.x,
                                               track_id)
                    point_Y = X_vector_element(paramerter_type, X_vector_element.value_type_Y, point.coord_W.y,
                                               track_id)
                    point_Z = X_vector_element(paramerter_type, X_vector_element.value_type_Z, point.coord_W.z,
                                               track_id)
                    X_vector_for_cam.extend([point_X, point_Y, point_Z])

                    # todo: sigma ist noch fuer pixel ! das muss geaendert werden
                    L_x = L_vector_element(photo.label, track_id, L_vector_element.value_type_x, point.measurement_C.x,
                                           photo.calc_sigma().x)
                    L_y = L_vector_element(photo.label, track_id, L_vector_element.value_type_y, point.measurement_C.y,
                                           photo.calc_sigma().y)

                    L_vector_for_cam.extend([L_x, L_y])

            jacobian_row = self.get_jacobian_row_for_point(X_vector_for_cam,
                                                           L_vector_for_cam,
                                                           X_to_optimize)

            X_vector.extend(X_vector_for_cam)
            L_vectro.extend(L_vector_for_cam)
            jacobian.extend(jacobian_row)
        jacobian_matrix = PhotoScan.Matrix(jacobian)
        return jacobian_matrix, X_vector, L_vectro


    def get_jacobian_row_for_point(self, X_vector, L_vector, X_used):
        """
        get the row of the jacobian for a specific parameter - measurement combination

        :type X_vector: list of X_vector_element
        :type L_vector: list of L_vector_element
        :type X_used: list of str
        :param X_vector: list of X_vector_element
        :param L_vector: list of L_vector_element
        :param X_used: list of str
        :return:
        """

        z = 1  # because all unprojected points has z=1
        R = None
        X_0 = None
        Y_0 = None
        Z_0 = None
        X = None
        Y = None
        Z = None
        for X_element in X_vector:
            if X_element.parameter_type == X_element.paramerter_type_cam:
                if X_element.value_type == X_element.value_type_R:
                    R = X_element.value
                elif X_element.value_type == X_element.value_type_X:
                    X_0 = X_element.value
                elif X_element.value_type == X_element.value_type_Y:
                    Y_0 = X_element.value
                elif X_element.value_type == X_element.value_type_Z:
                    Z_0 = X_element.value
            elif X_element.parameter_type == X_element.paramerter_type_point:
                if X_element.value_type == X_element.value_type_X:
                    X = X_element.value
                elif X_element.value_type == X_element.value_type_Y:
                    Y = X_element.value
                elif X_element.value_type == X_element.value_type_Z:
                    Z = X_element.value

        k_x = R[0, 0] * (X - X_0) + R[1, 0] * (Y - Y_0) + R[2, 0] * (Z - Z_0)
        k_y = R[0, 1] * (X - X_0) + R[1, 1] * (Y - Y_0) + R[2, 1] * (Z - Z_0)
        N = R[0, 2] * (X - X_0) + R[1, 2] * (Y - Y_0) + R[2, 2] * (Z - Z_0)

        row_x = [None] * len(X_used)  # row for x image measurement
        row_y = [None] * len(X_used)  # row for y image maesurement

        for L in L_vector:
            if L.value_type == L_vector_element.value_type_x:
                for i, X in enumerate(X_used):


                    if X == self.point_X:
                        # df(x)/dX
                        row_x[i] = -(z / N ** 2) * (R[0, 2] * k_x - R[0, 0] * N)
                    if X == self.point_Y:
                        row_x[i] = -(z / N ** 2) * (R[1, 2] * k_x - R[1, 0] * N)
                    if X == self.point_Z:
                        row_x[i] = -(z / N ** 2) * (R[2, 2] * k_x - R[2, 0] * N)


            elif L.value_type == L_vector_element.value_type_y:
                for i, X in enumerate(X_used):
                    if X == self.point_X:
                        # df(x)/dX
                        row_y[i] = -(z / N ** 2) * (R[0, 2] * k_y - R[0, 1] * N)
                    if X == self.point_Y:
                        row_y[i] = -(z / N ** 2) * (R[1, 2] * k_y - R[1, 1] * N)
                    if X == self.point_Z:
                        row_y[i] = -(z / N ** 2) * (R[2, 2] * k_y - R[2, 1] * N)
        jacobian = [row_x, row_y]
        return jacobian

    def export_no_xyz_cov(self, filename='export.txt'):

        point_track_ids = list(self.points.keys())
        print('outputpoints ', len(point_track_ids))
        f = open(
            filename, 'w')

        print('output start for %i points' % len(point_track_ids))

        for track_id in point_track_ids:
            cov = self.get_cov_for_point(track_id)
            position = self.points_pos[track_id]

            # doc.path  # C:\User....\project.psz




            output = ''
            output += '%i;' % track_id
            output += '%15.12e;%15.12e;%15.12e\n' % (
                position.x, position.y, position.z)
            output += '%15.12e;%15.12e;%15.12e\n' % (
                cov.row(0).x, cov.row(0).y, cov.row(0).z)
            output += '%15.12e;%15.12e;%15.12e\n' % (
                cov.row(1).x, cov.row(1).y, cov.row(1).z)
            output += '%15.12e;%15.12e;%15.12e' % (
                cov.row(2).x, cov.row(2).y, cov.row(2).z)

            if track_id != point_track_ids[-1]:
                output += '\n'
            f.write(output)

        f.close()
        print('output finish')


class Py_2_OpenScad():
    @classmethod
    def errorEllipse_from_eig(self, eigvector, eigvalue, position, factor=1):

        """


        :rtype : str
        :return : scad_string
        :param eigvector: 3x3 list each column is a eigenvector
        :param eigvalue: 1x3 list of eigenvalue. each corrensponding to the column in eigenvector
        :param position: 1x3 list of x,y,z coordinates
        :param factor: the scale factor
        :type eigvector: list of float
        :type eigvalue: list of float
        :type position: list of float
        :type factor: float
        """
        sorted_indeces_descanding = sorted(range(len(eigvalue)), key=lambda k: eigvalue[k])[::-1]
        sorted_eigenvalue = []
        for sort_i in sorted_indeces_descanding:
            sorted_eigenvalue.append(eigvalue[sort_i])
        v1 = []
        v2 = []
        v3 = []

        for row in eigvector:
            v1.append(row[sorted_indeces_descanding[0]])
            v2.append(row[sorted_indeces_descanding[1]])
            v3.append(row[sorted_indeces_descanding[2]])

        roh = 180 / math.pi
        gamma = math.atan(v1[1] / v1[0]) * roh
        len_x_ = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        beta = math.atan(v1[2] / len_x_) * roh
        alpha = -math.atan(v3[1] / v3[2]) * roh

        scale = list(
            map(lambda x: x / factor,
                [sqrt(sorted_eigenvalue[0]), sqrt(sorted_eigenvalue[1]), sqrt(sorted_eigenvalue[2])]))

        scad_string = "render(){"
        scad_string += "translate([{:6.3f},{:6.3f},{:6.3f}])".format(position[0], position[1], position[2])
        scad_string += "rotate([{:6.3f},{:6.3f},{:6.3f}])".format(alpha, beta, gamma)
        scad_string += "scale([{:6.3f},{:6.3f},{:6.3f}]*factor)".format(scale[0], scale[1], scale[2])
        scad_string += "sphere(r = {:6.3f},$fn=15);}}\n".format(factor)
        # scad_string += "import(\"C:\\\\Users\\\\philipp.atorf.INTERN\\\\Downloads\\\\sphere.stl\");}}\n".format(factor)

        return scad_string


class STL_Handler():
    def __init__(self):
        self.triangle = []
        self.importSTL()
        self.farcet_count = 0


    def importSTL(self, fname="sp_exp.stl"):
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        with open(__location__ + '\\' + fname, 'r') as f:
            # content = f.readlines()
            content = f.read().splitlines()
            for line in content:
                if "outer loop" in line:
                    triple = []

                vetex_pattern = 'vertex\s(\S*)\s(\S*)\s(\S*)'

                match = re.search(vetex_pattern, line)
                if match:
                    vertex_x = float(match.group(1))
                    vertex_y = float(match.group(2))
                    vertex_z = float(match.group(3))
                    vertex_photoscan_vector = PhotoScan.Vector((vertex_x, vertex_y, vertex_z))
                    triple.append(vertex_photoscan_vector)
                if "endloop" in line:
                    self.triangle.append(triple)

    def create_ellipsoid_stl(self, eigvector, eigvalue, position, factor=1, binary=True):
        sorted_indeces_descanding = sorted(range(len(eigvalue)), key=lambda k: eigvalue[k])[::-1]
        sorted_eigenvalue = []
        for sort_i in sorted_indeces_descanding:
            sorted_eigenvalue.append(eigvalue[sort_i])
        v1 = []
        v2 = []
        v3 = []

        for row in eigvector:
            v1.append(row[sorted_indeces_descanding[0]])
            v2.append(row[sorted_indeces_descanding[1]])
            v3.append(row[sorted_indeces_descanding[2]])

        roh = 1  # 180 / math.pi
        gamma = math.atan(v1[1] / v1[0]) * roh
        len_x_ = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        beta = math.atan(v1[2] / len_x_) * roh
        alpha = -math.atan(v3[1] / v3[2]) * roh

        scale = list(
            map(lambda x: x * factor,
                [sqrt(sorted_eigenvalue[0]), sqrt(sorted_eigenvalue[1]), sqrt(sorted_eigenvalue[2])]))

        scale_matrix = PhotoScan.Matrix.diag(scale
                                             )
        rot_x = PhotoScan.Matrix([[1, 0, 0],
                                  [0, math.cos(alpha), -math.sin(alpha)],
                                  [0, math.sin(alpha), math.cos(alpha)]])
        rot_y = PhotoScan.Matrix([[math.cos(beta), 0, math.sin(beta)],
                                  [0, 1, 0],
                                  [-math.sin(beta), 0, math.cos(beta)]])

        rot_z = PhotoScan.Matrix([[math.cos(gamma), -math.sin(gamma), 0],
                                  [math.sin(gamma), math.cos(gamma), 0],
                                  [0, 0, 1]])

        rot_all = rot_z * rot_y * rot_x

        rot_and_scale_matrix = rot_all * scale_matrix
        translation = PhotoScan.Vector(position)

        ellisoid_data = None
        if not binary:
            ellisoid_data = ""
        else:
            ellisoid_data = []

        for triangle in self.triangle:
            transformed_triple = []
            if binary:
                ellisoid_data.append([])
                for vertex in triangle:
                    newvertex = translation + rot_and_scale_matrix * vertex
                    transformed_triple.append(newvertex)
                    ellisoid_data[-1].append((newvertex.x, newvertex.y, newvertex.z))
            else:
                for vertex in triangle:
                    newvertex = translation + rot_and_scale_matrix * vertex
                    transformed_triple.append(newvertex)
                    ellisoid_data += self.create_vertex_string(transformed_triple)
                # else:
                # ellisoid_data.append([transformed_triple.x,transformed_triple.y,transformed_triple.z])

        return ellisoid_data


    @staticmethod
    def create_vertex_string(vertex_triple):
        vertex_string = "facet normal 0 0 0\n" + \
                        "outer loop\n"
        for vertex in vertex_triple:
            vertex_string += "vertex {:6.3f} {:6.3f} {:6.3f}\n".format(vertex.x, vertex.y, vertex.z)
        vertex_string += "endloop\nendfacet\n"
        return vertex_string

    def create_vertex_binary(self, vertex_triple):
        self.farcet_count += 1
        data = [0, 0, 0]
        for vertex in vertex_triple:
            data.append(vertex.x, vertex.y, vertex.z)
        data.append(0)
        return data


class SVG_Photo_Representation():
    colormap = ['rgb(254,240,217)', 'rgb(253,204,138)', 'rgb(252,141,89)', 'rgb(215,48,31)']
    colormap_green_2_red = ['rgb(141, 236, 14)', 'rgb( 222,  239, 13)', 'rgb(243, 149, 11)', 'rgb(244, 10, 38)']

    def __init__(self, photo, svg_width=600):

        """

        :type photo: list of I3_Photo
        """

        self.i3Photo = photo
        self.width = photo[0].photoscanCamera.sensor.width
        self.height = photo[0].photoscanCamera.sensor.height
        self.svg_witdh = svg_width
        self.svg_height = self.svg_witdh / (self.width / self.height)
        self.labelpos = (10, 16)
        self.imagepos = (0, 20)
        self.point_radius = 2
        self.circle_stroke = 1
        self.p_sigma = None  # todo: p_sigma bestimmen
        self.count_legend = None

    @property
    def points(self):
        points = []
        for photo in self.i3Photo:
            points.extend(photo.points)
        return points


    def set_count_legend(self, colormap, min_max):
        group = g()
        height = 0
        cat_borders, cat_size = self.__class__.get_categroy_ranges(min_max, colormap)
        shape_builder = ShapeBuilder()

        title = text("Point Count per Cell", 0, -4)
        group.addElement(title)

        color_rec = shape_builder.createRect(0, height, 20, 20, strokewidth=1, fill='white')
        label = text("&lt; {:9.2f}".format(1), 30, 16)
        height = 20

        group.addElement(label)
        group.addElement(color_rec)
        for i, border in enumerate(cat_borders):
            # draw rect
            color_rec = shape_builder.createRect(0, 20 * i + height, 20, 20, strokewidth=1, fill=colormap[i])
            label = text("&lt; {:9.2f}".format(border), 30, 20 * (i + 1) - 4 + height)

            group.addElement(label)
            group.addElement(color_rec)

        self.count_legend = group

        # draw label

        return group


    def get_raster_legend(self):
        return self.count_legend

    def get_lable(self, ):

        # Add Label
        label = text("All Photos Error", *self.labelpos)
        if (len(self.i3Photo) == 1):
            label = text(self.i3Photo[0].print_report_line(), *self.labelpos)
        textStyle = StyleBuilder()
        textStyle.setFontSize('16')
        label.set_style(textStyle.getStyle())

        return label

    def get_raw_error_vector_svg(self, as_raster=False, factor=40, cols=22):


        shape_builder = ShapeBuilder()
        photo_group = g()

        label = self.get_lable()
        photo_group.addElement(label)

        image_group = g()
        image_frame = shape_builder.createRect(0, 0, self.svg_witdh, self.svg_height, 0, 0, strokewidth=1,
                                               stroke='navy')
        image_group.addElement(image_frame)

        points = self.points

        if as_raster:
            points = self.get_points_in_raster(cols)[0]

        for point in points:
            point_x, point_y = self.transform_2_SVG(point.measurement_I.x,
                                                    point.measurement_I.y)
            point_pos = shape_builder.createCircle(point_x, point_y, self.point_radius,
                                                   self.circle_stroke)  # ,fill='rgba(0,0,0,1)')
            image_group.addElement(point_pos)
            image_group.addElement(self.drawErrorVector(point, factor))

        # Image Group Translation
        transImage = TransformBuilder()
        transImage.setTranslation(*self.imagepos)
        image_group.set_transform(transImage.getTransform())

        photo_group.addElement(image_group)

        total_height = self.imagepos[1] + self.svg_height
        return photo_group, total_height


    @classmethod
    def get_color_4_value(cls, min_max, val, colormap):
        min_val = min_max[0]
        cat_size = cls.get_categroy_ranges(min_max, colormap)[1]
        cat_value = int((val - min_val) / cat_size)

        return colormap[cat_value]

    @classmethod
    def get_categroy_ranges(cls, min_max, colormap):
        min_val = min_max[0] - 0.00000001
        max_val = min_max[1] + 0.00000001
        val_range = max_val - min_val
        cat_count = len(colormap)
        cat_size = val_range / cat_count
        cat_border = []
        for i, color in enumerate(colormap):
            cat_border.append((i + 1) * cat_size)
        return cat_border, cat_size


    def get_raster_count_svg(self, cols):
        coutn_raster, size = self.getRaster(cols)
        min_max_list = []
        shape_builder = ShapeBuilder()
        group = g()
        min_max = []

        for i, col in enumerate(coutn_raster):
            for j, row in enumerate(col):
                min_max_list.append(len(row))

        max_count = min(min_max_list)
        min_count = max(min_max_list)

        min_max.extend((max_count, min_count))
        self.set_count_legend(self.colormap, min_max)

        for i, col in enumerate(coutn_raster):
            for j, row in enumerate(col):
                coutn_raster[i][j] = len(row)

                pos_x, pos_y = self.transform_2_SVG(j * size, i * size)
                size_svg = self.transform_2_SVG(size, size)[0]

                color = SVG_Photo_Representation.get_color_4_value(min_max, len(row), self.colormap)
                if len(row) <= 1:
                    color = 'white'
                count_rect = shape_builder.createRect(pos_x,
                                                      pos_y,
                                                      size_svg,
                                                      size_svg,
                                                      strokewidth=0,
                                                      fill=color)
                group.addElement(count_rect)





        # Image Group Translation
        transImage = TransformBuilder()
        transImage.setTranslation(*self.imagepos)
        group.set_transform(transImage.getTransform())

        return group


    def drawErrorVector(self, point, factor=30, ):
        """
        :type factor: int
        :type point: I3_Point
        """

        error_vector = point.error_I * factor
        endpoint = point.measurement_I + error_vector
        x0, y0 = self.transform_2_SVG(point.measurement_I.x, point.measurement_I.y, )
        x1, y1 = self.transform_2_SVG(endpoint.x, endpoint.y)

        sha = ShapeBuilder()

        color = 'black'
        if self.p_sigma:
            error_length = error_vector.norm()
            color = self.colormap_green_2_red[3]
            for i in range(1, 4):
                if i * error_length <= error_length:
                    color = self.colormap_green_2_red[i - 1]

        error_line = sha.createLine(x0, y0, x1, y1, 1, stroke=color)

        return error_line

    def transform_2_SVG(self, x_image, y_image):


        x_svg = x_image * self.svg_witdh / self.width
        y_svg = y_image * self.svg_witdh / self.width

        return int(x_svg + 0.5), int(y_svg + 0.5)  # correct int round

    def getRaster(self, cols=22):

        width_I = self.width
        height_I = self.height

        size = width_I / cols
        rows = int(height_I / size + 0.5)
        # cols += 1 #fall nicht kann das array zu kurz sein falls ein punkt genau am bildrand liegt
        # errorRaster=[]
        # for row in range(rows): errorRaster += [[PhotoScan.Vector((0,0))]*cols]
        error_raster = [[[] for x in range(cols)] for x in range(rows)]

        for point in self.points:
            i = int(point.measurement_I.y * (rows ) / height_I)
            j = int(point.measurement_I.x * (cols ) / width_I)
            # print('floatcols', point.measurement_I.x * (cols - 1) / width_I)
            # print('floatrows', point.measurement_I.y * (rows - 1) / height_I)

            # print('rows', len(errorRaster))
            # print('cols', len(errorRaster[i]))
            # print(len(errorRaster[i][j]))
            # errorRaster[i][j] += point.error_I
            error_raster[i][j].append(point)
            # errorRaster[i][j][1] += 1
        return error_raster, size

    def get_points_in_raster(self, cols=22):
        """

        :rtype : (list(I3_Point),int)
        """
        error_raster, size = self.getRaster(cols)
        new_points = []
        for i, col in enumerate(error_raster):
            for j, row in enumerate(col):
                error_vector = PhotoScan.Vector((0, 0))
                for point in row:
                    error_vector += point.error_I

                error_mean = error_vector

                if len(row):  # if empty  avoid div by 0
                    error_mean = (error_vector / len(row))

                    pos_center = PhotoScan.Vector((j * size + (size / 2), (i * size + (size / 2))))
                    pseuso_projection = pos_center + error_mean

                    new_point_at_cell_center = I3_Point(measurement_I=pos_center, projection_I=pseuso_projection)

                    new_points.append(new_point_at_cell_center)
                    # error_raster[i][j] = new_point_at_cell_center

        return new_points, size


def trans_error_image_2_camera(camera, point_pix, point_Camera):
    t = camera.transform
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


def calc_Cov_from_ErrorMatrix(error_matrix):
    # X_list = []
    # for error in pointError:
    # X_list.append([error.x, error.y, error.z])

    X_matrix = PhotoScan.Matrix(error_matrix)

    C = X_matrix.t() * X_matrix
    C = C * (1 / (len(error_matrix) ))

    return C


def calc_Cov_4_allPoints(point_list):
    covs = {}  # Key = trackid ; value = 3x3 Matrix

    for track_id, error in point_list.items():
        if len(error) > 3:
            pass
            # cov = calc_Cov_4_Point(error)
            # covs[track_id] = cov
        else:
            pass

    return  # covs


def creatExportList(points, covs_dict):
    export_points = []
    for point in points:
        if covs_dict.get(point.track_id):
            export_points.append(
                [point.track_id, point.coord, covs_dict[point.track_id]])
    return export_points


def export_no_xyz_std(points, covs_Dict):
    export_points = creatExportList(points, covs_Dict)
    f = open(
        'C:\\Users\\philipp.atorf.INTERN\\Downloads\\building\\export_xyz.txt', 'w')

    print('output xyz sx sy sz start for %i points' % len(export_points))

    for point in export_points:
        output = ''
        output += '%i;' % point[0]
        output += '%15.12e;%15.12e;%15.12e;%15.12e' % (
            point[1].x,
            point[1].y,
            point[1].z,
            sqrt(point[2].row(0).x + point[2].row(1).y + point[2].row(2).z))

        if point != export_points[-1]:
            output += '\n'
        f.write(output)

    f.close()
    print('output finish')


if __name__ == '__main__':

    doc = PhotoScan.app.document
    chunk = doc.chunk
    project = I3_Project()
    total_error, ind_error, allPhotos = project.calc_reprojection(chunk)
    # project.build_global_point_error()
    project.calc_cov_for_all_points()
    project.print_report()
    project.create_project_SVG()
    project.export_STL(binary=True, factor=(0.05 / 10))
    for arg in sys.argv:

        if arg is not "no3d":
            pass





    testPointError = [PhotoScan.Vector((1, 2, 1.4)), PhotoScan.Vector(
        (-1.2, 1, 2.3)), PhotoScan.Vector((-1.4, 2, 3))]
    # print (calc_Cov_4_Point(testPointError))


    ### Programm Start ###

    pointErrors_W = defaultdict(list)
    pointErrors_I = defaultdict(list)



    project = I3_Project()
    total_error, ind_error, allPhotos = project.calc_reprojection(chunk)
    # project.build_global_point_error()
    project.calc_cov_for_all_points()
    project.print_report()
    # project.create_project_SVG()
    # project.export_for_OpenScad()
    # project.export_STL(binary=True, factor=(0.05/10))
    # points_reference = project.get_point_photos_reference()
    # for key,value in points_reference.items():
    # if len(value)<3:
    # print(key)
    # print(points_reference)
    # adjustment = peseudo_3D_intersection_adjustment(points_reference)
    # adjustment.get_jacobian(points_reference, 19101)
    # adjustment.get_jacobian( list(points_reference.keys())[1000])
    # Qxx = adjustment.get_cov_for_point(list(points_reference.keys())[200])
    # print(Qxx)
    # project.create_project_SVG()
    # print(total_error)
    # print(ind_error)
    # print(vars(allPhotos[0].points[1]))


    # covs_Dict = calc_Cov_4_allPoints(pointErrors_W)

    # point_cloud = chunk.point_cloud
    # points = point_cloud.points

    # export_No_xyz_cov(points, covs_Dict)
