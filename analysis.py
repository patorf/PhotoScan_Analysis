"""
PhotoScan Analyse 0.2
"""

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

imp.reload(pysvg)
from pysvg.builders import *


class I3_Photo():
    """
    I3 Photo can store photo information.

    :ivar label: image label
    :ivar points: list of visible points
    :ivar photoScan_camera: PhotoScan.Camera representation of this Photo
    :ivar sigma: standard deviation of the image measurement. calculatet from the covarianzmatrix of all measurements

    """

    def __init__(self, label=None):
        """

        """
        self.label = label

        self.points = []
        """:type : list of I3_Point"""
        self.photoScan_camera = None
        """:type : PhotoScan.Camera"""
        self.__sigma_I = None
        """:type : PhotoScan.Vector"""
        self.__sigma_C = None
        """:type : PhotoScan.Vector"""
        self.thumbnail_path = None

    def add_point(self, new_point=None):
        """

        :rtype : I3_Point
        """
        if new_point is None:
            new_point = I3_Point()
        self.points.append(new_point)
        return self.points[-1]

    def __calc_sigma(self, where='I'):
        """
        returns the standard deviation of the x and y image measurements.
        the calculation used the covarianzmatrix (see. http://www.chemgapedia.de/vsengine/vlu/vsc/de/ch/13/vlu/daten/multivariate_datenanalyse_allg/multivar_datenanalyse_allg.vlu/Page/vsc/de/ch/13/anc/daten/multivar_datenanalyse_allg/varianz_kovarianzmatrix.vscml.html)
        :rtype : PhotoScan.Vector
        """

        error_matrix = self.get_error_matrix(where)
        cov = self.calc_cov_from_error_matrix(error_matrix)

        sigma_x = math.sqrt(cov[0, 0])
        sigma_y = math.sqrt(cov[1, 1])

        return PhotoScan.Vector([sigma_x, sigma_y])

    def __calc_sigma_I(self):
        if self.__sigma_I is None:
            self.__sigma_I = self.__calc_sigma('I')
        return self.__sigma_I

    def __call_sigma_C(self):
        if self.__sigma_C is None:
            self.__sigma_C = self.__calc_sigma('C')
        return self.__sigma_C

    def __set_sigma_I(self, sigma):
        self.__sigma_I = sigma

    def __set_sigma_C(self, sigma):
        self.__sigma_C = sigma

    def get_max_error(self):
        error_matrix = self.get_error_matrix()

        max_error = PhotoScan.Vector((0, 0))

        max_error.x = max(abs(l[0]) for l in error_matrix)
        max_error.y = max(abs(l[1]) for l in error_matrix)

        return max_error

    @staticmethod
    def calc_cov_from_error_matrix(error_matrix):
        # X_list = []
        # for error in pointError:
        # X_list.append([error.x, error.y, error.z])

        X_matrix = PhotoScan.Matrix(error_matrix)

        C = X_matrix.t() * X_matrix
        C = C * (1 / (len(error_matrix)))

        return C

    def get_error_matrix(self, where='I'):
        """
        the value of each row of error matrix shows the image error in x and y.
        normalie this matrix is calculated by subtraction the measurement by the
        mean of all measurements (coloumn mean) [x_11 - x_mean, y_11-y_mean;...]
        """
        error_matrix = []

        for point in self.points:
            if where == 'I':
                error_matrix.append([point.error_I.x, point.error_I.y])
            elif where == 'C':
                error_matrix.append([point.error_C.x, point.error_C.y])
        return error_matrix

    @staticmethod
    def print_report_header():
        r_str = 'Photoscan Analysis Image-Measurement-Report (Unit = Pixel) \n'
        r_str += '{0:>12s}{1:>14s}{2:>9s}{3:>9s}{4:>9s}{5:>9s}{6:>9s}\n' \
            .format('Camera Name',
                    'Projections',
                    'SIGMA x',
                    'SIGMA y',
                    'SIGMA P',
                    'MAX x',
                    'MAX y')

        return r_str

    def print_report_line(self):

        r_str = ''
        sigma = self.sigma_I
        max_error = self.get_max_error()
        r_str += '{:>12s}{:14d}{:9.5f}{:9.5f}{:9.5f}{:9.5f}{:9.5f}\n'. \
            format(self.label,
                   len(self.points),
                   sigma.x,
                   sigma.y,
                   sigma.norm(),
                   max_error.x,
                   max_error.y)

        return r_str

    sigma_I = property(__calc_sigma_I, __set_sigma_I)
    sigma_C = property(__call_sigma_C, __set_sigma_C)


class I3_Point():
    """
    Representation of a feature point in a Image and the world point.
    :ivar projection_I: reprojection of the world point into the image plane in meter
    :ivar measurement_I: measurement of the feature in pixel
    :ivar track_id: global id of the point in meter
    :ivar coord_W: world coordinates of the point in meter
    :ivar coord_C: camera coordinates of the point in meter
    """

    def __init__(self,
                 projection_I=None,
                 measurement_I=None,
                 measurement_C=None,
                 track_id=None,
                 coord_W=None,
                 coord_C=None,
                 ):
        self.projection_I = projection_I
        self.measurement_I = measurement_I
        self.measurement_C = measurement_C
        self.track_id = track_id
        self.coord_W = coord_W
        self.coord_C = coord_C

    @property
    def error_I(self):
        """
        reprojection error in pixel

        :return:
        """
        return self.projection_I - self.measurement_I

    @property
    def error_C(self):
        """
        reprojection error in meter on the image plance with a distance of 1m
        """
        point_on_image_plane = self.coord_C / self.coord_C.z
        return point_on_image_plane - self.measurement_C


class I3_Project():
    """
    I3_Project is the  main class of the module.
    it is used to controll the workflow of the analysis.
    Beside some helper functions, the main workflow functions are:
    print_report()
    create_project_SVG()
    export_STL(binary=True, factor=factor)

    :ivar photos: list of all photos in this project
    :ivar point_photo_reference: dictonary with track_id as key
    and a list of all photos, in which the points are visible, as value
    """

    def __init__(self, chunk=None):
        self.photos = []
        """:type: list[I3_Photo]"""
        self.point_photo_reference = {}
        """:type: dict[int, list[I3_Photo]]"""

        self.path = PhotoScan.app.document.path
        self.adjustment = None

        project_directory = "\\".join(self.path.split('\\')[:-1])
        analyse_dir = PhotoScan.app.getExistingDirectory(
            'Please create or select a folder where the analys files will be saved')
        if analyse_dir:
            self.directory = analyse_dir
        else:

            self.directory = project_directory

        if chunk:
            self.__fill_photos_with_points(chunk)

    def __get_point_photos_reference(self):

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
        adjustment = Peseudo_3D_intersection_adjustment(self.__get_point_photos_reference())

        ellipsoid_parameter_list = adjustment._get_eigvalues_eigvectors_pos_cov_for_track_id()
        output_str = "factor = 0.051;\n"
        for ellipsoid_parameter in ellipsoid_parameter_list:
            eig_val, eig_vec, pos = ellipsoid_parameter
            output_str += Py_2_OpenScad.errorEllipse_from_eig(eig_vec, eig_val, pos)

        f = open(self.directory + '\\' + filename, 'w')
        f.write(output_str)
        f.close()
        print('save file ', filename, ' to: ', self.directory)

    def export_STL(self, filename=None, binary=None, factor=None):
        if filename is None:
            filename = 'stl_export'
        filename += '.stl'
        if binary is None:
            binary = True
        if factor is None:
            factor = 100

        print('start output STL-File with factor {:8.6f}: '.format(factor) + filename)
        PhotoScan.app.update()
        if self.adjustment is None:
            self.adjustment = Peseudo_3D_intersection_adjustment(self.__get_point_photos_reference())

        ellipsoid_parameter_list = self.adjustment._get_eigvalues_eigvectors_pos_cov_for_track_id()
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
                eig_val, eig_vec, pos, cov = ellipsoid_parameter
                data.extend(stl_handler.create_ellipsoid_stl(eig_vec, eig_val, pos, factor, True))
            with open(self.directory + '\\' + filename, 'wb') as fp:
                writer = Binary_STL_Writer(fp)
                writer.add_faces(data)
                writer.close()
                print('save bin file ', filename, ' to: ', self.directory)

    def exportEllipsoids(self, filename=None):
        if filename is None:
            filename = 'ellipsoid_export'
        filename += '.ell'

        if self.adjustment is None:
            self.adjustment = Peseudo_3D_intersection_adjustment(self.__get_point_photos_reference())

        ellipsoid_list = self.adjustment._get_eigvalues_eigvectors_pos_cov_for_track_id()

        export_str = "xc yc zc\n" \
                     "xr.x xr.y xr.z\n" \
                     "yr.x yr.y yr.z\n" \
                     "zr.x zr.y zr.z\n" \
                     "C11 C12 C13\n" \
                     "C21 C22 C23\n" \
                     "C31 C32 C33\n"

        for ellipsoid in ellipsoid_list:
            eigVal = PhotoScan.Vector(ellipsoid[0])
            eigVecs = PhotoScan.Matrix(ellipsoid[1])
            pos = PhotoScan.Vector(ellipsoid[2])
            cov = ellipsoid[3]

            xr = list(eigVecs.col(0) * eigVal[0])
            yr = list(eigVecs.col(1) * eigVal[1])
            zr = list(eigVecs.col(2) * eigVal[2])
            xc = pos[0]
            yc = pos[1]
            zc = pos[2]
            export_str += "{:.2e} {:.2e} {:.2e}\n".format(xc, yc, zc)

            export_str += "{:.2e} {:.2e} {:.2e}\n".format(xr[0], xr[1], xr[2])
            export_str += "{:.2e} {:.2e} {:.2e}\n".format(yr[0], yr[1], yr[2])
            export_str += "{:.2e} {:.2e} {:.2e}\n".format(zr[0], zr[1], zr[2])

            export_str += "{:.4e} {:.4e} {:.4e}\n".format(cov[0, 0], cov[0, 1], cov[0, 2])
            export_str += "{:.4e} {:.4e} {:.4e}\n".format(cov[1, 0], cov[1, 1], cov[1, 2])
            export_str += "{:.4e} {:.4e} {:.4e}\n".format(cov[2, 0], cov[2, 1], cov[2, 2])
        f = open(self.directory + '\\' + filename, 'w')
        f.write(export_str)
        f.close()
        print('save file ', filename, ' to: ', self.directory)

    def _get_RMS_4_all_photos(self, photos=None):
        """
        returns the root mean square for all photos in this project
        :param photos:
        :return:
        """
        if not photos:
            photos = self.photos

        var_x_sum = 0
        var_y_sum = 0
        for photo in photos:
            sigma_photo = photo.sigma_I
            var_x_sum += sigma_photo.x ** 2
            var_y_sum += sigma_photo.y ** 2

        rms_x = math.sqrt(var_x_sum / len(photos))
        rms_y = math.sqrt(var_y_sum / len(photos))

        return rms_x, rms_y

    def __save_thumbnails(self):
        for photo in self.photos:
            thumbnail_path = self.directory + '/' + photo.label
            success = photo.photoScan_camera.thumbnail.image().save(thumbnail_path)

            if success:
                photo.thumbnail_path = thumbnail_path
            else:
                print('can not save thumbnail')

    def __fill_photos_with_points(self, chunk):
        """
        saves all photos (with points) in the chunck to slef.photos
        :param chunk:
        :return:
        """
        all_photos = self.photos
        point_cloud = chunk.point_cloud

        points = point_cloud.points
        npoints = len(points)
        projections = chunk.point_cloud.projections
        for camera in chunk.cameras:

            if not camera.transform:
                continue
            # create new photo
            this_photo = I3_Photo(camera.label)
            this_photo.photoScan_camera = camera
            all_photos.append(this_photo)

            T = camera.transform.inv()
            calib = camera.sensor.calibration

            point_index = 0

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
                    # print("-------------",track_id)
                    # print("center",calib.project(PhotoScan.Vector([0,0,1])))

                    # print("PointW",point_W)
                    # print("PointC ",point_C)
                    # print("point_I proj", point_I )
                    measurement_I = proj.coord
                    measurement_C = calib.unproject(measurement_I)
                    # print("measI",measurement_I)
                    # print("mearC",measurement_C)
                    # error_I = calib.error(point_C, measurement_I)

                    # error_C = point_C - measurement_C * point_C.z

                    # print(point_C)
                    # print(measurement_C)
                    # save Point in curren Photo
                    if point_I:
                        point = this_photo.add_point()

                        point.track_id = track_id
                        point.projection_I = point_I
                        point.measurement_I = measurement_I
                        point.coord_C = point_C
                        point.coord_W = point_W
                        point.measurement_C = measurement_C
                        # point.projection_C = error_C

    def save_and_print_report(self, filename=None):
        """
        prints and saves the report. the report contains information
        about each photo (count of measurements, standard deviation and max_error)
        in this project
        :param filename: filename to save the file
        :return:
        """
        if filename is None:
            filename = 'report'
        filename += '.txt'

        r_str = ""
        r_str += I3_Photo.print_report_header()
        for phots in self.photos:
            assert isinstance(phots, I3_Photo)
            r_str += phots.print_report_line()

        r_str += '\n'
        rms_x, rms_y = self._get_RMS_4_all_photos()
        r_str += '{:>26s}{:9.5f}{:9.5f}'.format('RMS:', rms_x, rms_y)

        print(r_str)

        f = open(self.directory + '\\' + filename, 'w')
        f.write(r_str)
        f.close()
        print('save file ', filename, ' to: ', self.directory)

    def create_project_SVG(self, filename=None, error_factor=None, cols=None):
        """
        this methode creates a svg file. The file contains a overview of
        all images with its feature-points and error vectors. additionally a
        overview shows the number of measurements in one raster cell.

        :param error_factor: magnification factor of the error-vector
        :param cols: the number of columns used to generate the overview image
        :return:
        """
        self.__save_thumbnails()
        if filename is None:
            filename = 'image_measurements'
        if error_factor is None:
            error_factor = 40
        if cols is None:
            cols = 20

        filename += '.svg'

        s = svg()

        summery_SVG = SVG_Photo_Representation(self.photos)
        summery_SVG.point_radius = 1

        summery, height = summery_SVG.get_raw_error_vector_svg(factor=error_factor)

        s.addElement(summery)
        summery_group = g()
        summery_error_raster, height = summery_SVG.get_raw_error_vector_svg(as_raster=True,
                                                                            factor=error_factor, cols=cols)
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
    """
    container class for X-Vector (Parameter Vector) elements used in the adjustment
    """
    paramerter_type_point = 'point'
    paramerter_type_cam = 'cam'
    value_type_X = 'X'
    value_type_Y = 'Y'
    value_type_Z = 'Z'
    value_type_R = 'R'

    def __init__(self, parameter_type, value_type, value, id_x):
        self.value_type = value_type
        self.parameter_type = parameter_type
        self.value = value
        self.id_x = id_x

    def __str__(self):

        if self.value_type == self.value_type_R:
            return "{:s} {:s} :{:s} id:{:s}".format(self.parameter_type,
                                                    self.value_type,
                                                    str(self.value),
                                                    str(self.id_x))
        else:
            return "{:s} {:s} :{:.9f} id:{:s}".format(self.parameter_type,
                                                      self.value_type,
                                                      self.value,
                                                      str(self.id_x))


class L_vector_element():
    """
    container class for L-Vector (measurement vector) elements used in the adjustment

    """
    value_type_x = 'x'
    value_type_y = 'y'

    def __init__(self, cam_id, track_id, value_type, value, sigma):
        self.cam_id = cam_id
        self.track_id = track_id
        self.value_type = value_type
        self.value = value
        self.sigma = sigma  # standard deviation

    def __str__(self):
        return "{:s} track_id:{:d} value_type:{:s} value:{:.9f} sigam:{:.9f} ".format(self.cam_id,
                                                                                      self.track_id,
                                                                                      self.value_type,
                                                                                      self.value,
                                                                                      self.sigma)


class Peseudo_3D_intersection_adjustment():
    """
    this class can calculate the jakobian matrix and the weight matrix and
    the covarianzmatrix of the 3D Points
    """
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
    def _get_eigen_vel_vec(m):
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

    def _get_eigvalues_eigvectors_pos_cov_for_track_id(self, track_id=None):
        """
        returns a list of a tuple of Eigenvalue, Eigenvector and Position for
        a track_id. if no id is passed it returns a list of all points

        :type track_id: int
        :param track_id: track_id or None. i
        :rtype : (list[float],list[list[float]],list[float],list[list[float]])
        """

        if track_id:
            list_of_track_ids = [track_id]
        else:
            list_of_track_ids = self.points.keys()

        return_list = []

        for track_id in list_of_track_ids:
            cov = self.get_cov_for_point(track_id)

            eig_val, eig_vec = self._get_eigen_vel_vec(cov)
            pos_vector = None
            for point in self.points[track_id][0].points:
                if point.track_id == track_id:
                    pos_vector = point.coord_W
            # pos_vector = self.points[track_id][0].points[track_id].coord_W
            pos = [pos_vector.x, pos_vector.y, pos_vector.z]
            return_list.append((eig_val, eig_vec, pos, cov))
        return return_list

    def get_cov_for_point(self, track_id):
        """
        calculate the covarianze matrix for one point
        :param track_id: id of a 3D Point
        :return: 3x3 Cov-Matrix
        :rtype : PhotoScan.Matrix
        """
        jacobian_matrix, X_vector, L_vector = self.get_jacobian(track_id)

        A = jacobian_matrix
        P = self.__get_P_matrix(L_vector, 1)
        N = A.t() * P * A
        Qxx = N.inv()

        return Qxx

    @staticmethod
    def __get_P_matrix(L_vector, sigma0=1):
        """
        calculate the weight matrix for a given measurement vector and a sigma0
        :type L_vector: list of L_vector_element
        :param L_vector:
        :param sigma0:
        :return:
        """
        # todo: wie kann ich sigma0 richtig bestimmen
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
        """
        returns the jacobian matrix for one point
        :param track_id:
        :param point_photo_reference:
        :return:
        """
        if point_photo_reference is None:
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
            R_t = photo.photoScan_camera.transform

            R = PhotoScan.Matrix([[R_t[0, 0], R_t[0, 1], R_t[0, 2]],
                                  [R_t[1, 0], R_t[1, 1], R_t[1, 2]],
                                  [R_t[2, 0], R_t[2, 1], R_t[2, 2]]])
            cam_R = X_vector_element(paramerter_type,
                                     X_vector_element.value_type_R,
                                     R,
                                     photo.label)
            cam_X = X_vector_element(paramerter_type, X_vector_element.value_type_X, photo.photoScan_camera.center.x,
                                     photo.label)
            cam_Y = X_vector_element(paramerter_type, X_vector_element.value_type_Y, photo.photoScan_camera.center.y,
                                     photo.label)
            cam_Z = X_vector_element(paramerter_type, X_vector_element.value_type_Z, photo.photoScan_camera.center.z,
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

                    L_x = L_vector_element(photo.label, track_id, L_vector_element.value_type_x, point.measurement_C.x,
                                           photo.sigma_C.x)
                    L_y = L_vector_element(photo.label, track_id, L_vector_element.value_type_y, point.measurement_C.y,
                                           photo.sigma_C.y)

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
        see Luhmann page 244,308

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


class Py_2_OpenScad():
    """
    class with one method to draw a allipsoid in OpenScad
    not used by now
    """

    @classmethod
    def errorEllipse_from_eig(cls, eigvector, eigvalue, position, factor=1):

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
    """
    class can import a STL file which represent a sphere.
    this sphere can transform to an ellipsoid
    """

    def __init__(self):

        # a triple of points
        self.triangle = []
        self.importSTL()
        self.farcet_count = 0

    def importSTL(self, fname="sphere_aus_meshlab.stl"):
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        self.triangle = []
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

    def create_ellipsoid_stl(self, eigvector, eigvalue, position, factor=1.0, binary=True):
        """
        returns a transformed sphere. list of triangles which is a list of vertexes
        :param eigvector:
        :param eigvalue:
        :param position:
        :param factor:
        :param binary:
        :return: list of triangles which is a list of vertexes
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

        roh = 1  # 180 / math.pi
        # gamm = rotation about the Z-Axis
        gamma = math.atan(v1[1] / v1[0]) * roh
        len_x_ = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        # beta = rotation about the Y-Axis
        beta = math.atan(v1[2] / len_x_) * roh
        # alpha = rotation about the X-Axis
        alpha = -math.atan(v3[1] / v3[2]) * roh

        scale = list(
            map(lambda x: x * factor,
                [sqrt(sorted_eigenvalue[0]), sqrt(sorted_eigenvalue[1]), sqrt(sorted_eigenvalue[2])]))

        scale_matrix = PhotoScan.Matrix.diag(scale)

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
        """
        returns ASCII stl facet
        :param vertex_triple:
        :return:
        """
        vertex_string = "facet normal 0 0 0\n" + \
                        "outer loop\n"
        for vertex in vertex_triple:
            vertex_string += "vertex {:6.3f} {:6.3f} {:6.3f}\n".format(vertex.x, vertex.y, vertex.z)
        vertex_string += "endloop\nendfacet\n"
        return vertex_string


class SVG_Photo_Representation():
    """
    class is the svg representation of a list of photos. if multiple photos are given,
    the represenatation is a overview or summery.
    """
    colormap = ['rgb(254,240,217)', 'rgb(253,204,138)', 'rgb(252,141,89)', 'rgb(215,48,31)']
    colormap_green_2_red = ['rgb(141, 236, 14)', 'rgb( 222,  239, 13)', 'rgb(243, 149, 11)', 'rgb(244, 10, 38)']

    def __init__(self, photos, svg_width=600):

        """

        :type photos: list of I3_Photo
        """

        self.i3Photos = photos
        self.width = photos[0].photoScan_camera.sensor.width
        self.height = photos[0].photoScan_camera.sensor.height
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
        for photo in self.i3Photos:
            points.extend(photo.points)
        return points

    def set_count_legend(self, colormap, min_max):
        """
        generate the legend of the raster image.
        :param colormap:
        :param min_max:
        :return:
        """
        group = g()
        height = 0
        cat_borders, cat_size = self.__class__.get_categroy_ranges(min_max, colormap)
        shape_builder = ShapeBuilder()

        title = text("point count per cell", 0, -4)
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

    def get_lable(self, as_raster=False):
        """
        returns tha lable of the photo.
        :return:
        """

        # Add Label
        # if overview photo
        label = None
        if as_raster:
            label = text("Images Error Raster Summery", *self.labelpos)
        else:
            label = text("Images Error Summery", *self.labelpos)
        # if normal photo
        if len(self.i3Photos) == 1:
            label = text(self.i3Photos[0].print_report_line(), *self.labelpos)
        text_style = StyleBuilder()
        text_style.setFontSize('16')
        label.set_style(text_style.getStyle())

        return label

    def get_raw_error_vector_svg(self, as_raster=False, factor=40, cols=22):
        """
        get a photo svg with error vectors. if as_raster is true then the errors are summarized
        :param as_raster:
        :param factor:
        :param cols:
        :return:
        """
        shape_builder = ShapeBuilder()
        photo_group = g()

        label = self.get_lable(as_raster)
        photo_group.addElement(label)

        image_group = g()
        image_frame = shape_builder.createRect(0, 0, self.svg_witdh, self.svg_height, 0, 0, strokewidth=1,
                                               stroke='navy')
        image_group.addElement(image_frame)

        if len(self.i3Photos) == 1:
            thumbnail = image(width=self.svg_witdh, height=self.svg_height)
            thumbnail.set_xlink_href(self.i3Photos[0].thumbnail_path)
            thumbnail.set_opacity(0.5)
            image_group.addElement(thumbnail)

        points = self.points

        if as_raster:
            points = self.get_points_in_raster(cols)[0]

        for point in points:
            point_x, point_y = self.transform_2_SVG(point.measurement_I.x,
                                                    point.measurement_I.y)
            point_pos = shape_builder.createCircle(point_x, point_y, self.point_radius,
                                                   self.circle_stroke)  # ,fill='rgba(0,0,0,1)')
            image_group.addElement(point_pos)
            image_group.addElement(self.draw_error_vector(point, factor))





        # Image Group Translation
        trans_image = TransformBuilder()
        trans_image.setTranslation(*self.imagepos)
        image_group.set_transform(trans_image.getTransform())

        photo_group.addElement(image_group)

        total_height = self.imagepos[1] + self.svg_height
        return photo_group, total_height

    @classmethod
    def get_color_4_value(cls, min_max, val, colormap):
        """
        get the color of a value by given the value, the min-max range and
        a color map

        :param min_max:
        :param val:
        :param colormap:
        :return:
        """
        min_val = min_max[0]
        cat_size = cls.get_categroy_ranges(min_max, colormap)[1]
        cat_value = int((val - min_val) / cat_size)

        return colormap[cat_value]

    @classmethod
    def get_categroy_ranges(cls, min_max, colormap):
        """
        get the category borders and the category size for a min-max range and a colormap

        :param min_max:
        :param colormap:
        :return:
        """
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
        """
        get the raster image
        :param cols:
        :return:
        """
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
        trans_image = TransformBuilder()
        trans_image.setTranslation(*self.imagepos)
        group.set_transform(trans_image.getTransform())

        return group

    def draw_error_vector(self, point, factor=30, ):
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
        # preparation for a vector in different color which used the3-sigma rule
        if self.p_sigma:
            error_length = error_vector.norm()
            color = self.colormap_green_2_red[3]
            for i in range(1, 4):
                if i * error_length <= error_length:
                    color = self.colormap_green_2_red[i - 1]

        error_line = sha.createLine(x0, y0, x1, y1, 1, stroke=color)

        return error_line

    def transform_2_SVG(self, x_image, y_image):
        """
        transform a x and y point (Image coordinates) to svg coordinates
        :param x_image:
        :param y_image:
        :return:
        """

        x_svg = x_image * self.svg_witdh / self.width
        y_svg = y_image * self.svg_witdh / self.width

        return int(x_svg + 0.5), int(y_svg + 0.5)  # correct int round

    def getRaster(self, cols=22):
        """
        returns a 2d list with points.
        :param cols:
        :return:
        """
        width_I = self.width
        height_I = self.height

        size = width_I / cols
        rows = int(height_I / size + 0.5)
        # cols += 1 #fall nicht kann das array zu kurz sein falls ein punkt genau am bildrand liegt
        # errorRaster=[]
        # for row in range(rows): errorRaster += [[PhotoScan.Vector((0,0))]*cols]
        error_raster = [[[] for x in range(cols)] for x in range(rows)]

        for point in self.points:
            i = int(point.measurement_I.y * rows / height_I)
            j = int(point.measurement_I.x * cols / width_I)
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
        returns a list of points which lie in the center of one raster cell.
        the point also has a projection coordinate which means that the error can be calculated

        :rtype : (list(I3_Point),int)
        """
        error_raster, size = self.getRaster(cols)
        new_points = []
        for i, col in enumerate(error_raster):
            for j, row in enumerate(col):
                error_vector = PhotoScan.Vector((0, 0))
                for point in row:
                    error_vector += point.error_I

                if len(row):  # if empty  avoid div by 0
                    error_mean = (error_vector / len(row))

                    pos_center = PhotoScan.Vector((j * size + (size / 2), (i * size + (size / 2))))
                    pseuso_projection = pos_center + error_mean

                    new_point_at_cell_center = I3_Point(measurement_I=pos_center, projection_I=pseuso_projection)

                    new_points.append(new_point_at_cell_center)
                    # error_raster[i][j] = new_point_at_cell_center

        return new_points, size


def creat_export_list(points, covs_dict):
    export_points = []
    for point in points:
        if covs_dict.get(point.track_id):
            export_points.append(
                [point.track_id, point.coord, covs_dict[point.track_id]])
    return export_points


def export_no_xyz_std(points, covs_Dict):
    export_points = creat_export_list(points, covs_Dict)
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

    make_report = False
    report_filename = None
    make_svg = False
    svg_filename = None
    svg_factor = None
    svg_cols = None
    make_stl = False
    stl_filename = None
    stl_factor = None
    export_ellipsoid = None

    def check_next_argument(current_argument_index):

        if len(sys.argv) - 1 > current_argument_index:
            next_argument = sys.argv[current_argument_index + 1]
            if next_argument[0] != '-' and next_argument != ' ':
                return next_argument
            else:
                return None

    doc = PhotoScan.app.document
    chunk = None
    if doc.chunk:
        chunk = doc.chunk

        if len(sys.argv) == 1:
            sys.argv.append('help')
        print('PhotoScan Analysis v0.1')

        for i, arg in enumerate(sys.argv):

            if arg == ' ':
                continue
            if arg == 'help':
                howto = 'HowTo:\n'
                howto += 'Command Line Arguments:\n'
                howto += '-rout [filename]\t\tCreates a report file. Options: filename (default: report)\n'
                howto += '-svgout [filename]\t\tCreates a SVG-Image with image-measurements Option: filename (default: image_measurements\n'
                howto += '-svgfactor [factor]\t\tMagnification factor of the error-vector for the SVG-File (default: 40)\n'
                howto += '-svgcols [columns]\t\tThe number of columns used to generate the overview image (default: 20)\n'
                howto += '-stlout [filename]\t\tCreate a STL-Mesh with Point-Error-Ellipsoids. Option: filename (default: stl_export)\n'
                howto += '-stlfactor [factor]\t\tMagnification factor of the ellipsoid-axis (default: 100)'
                howto += '-export_ellipsoid \t\t Export a ellipsoid file'

                howto += '\n\nSample:\n'
                howto += '-rout reportname -svgout svgname -svgfactor 12 -svgcols 10 -stlout stlname -stlfactor 12'
                howto += '\n\nGUI\n'
                howto += 'You can also use the GUI by choosing the argument \'-useGUI\''
                print(howto)
                break
            if arg == '-rout':
                report_filename = check_next_argument(i)
                make_report = True

            elif arg == '-svgout':
                svg_filename = check_next_argument(i)
                make_svg = True

            elif arg == '-svgfactor':
                svg_factor = float(check_next_argument(i))

            elif arg == '-svgcols':
                svg_cols = int(check_next_argument(i))

            elif arg == '-stlout':
                stl_filename = check_next_argument(i)
                make_stl = True

            elif arg == '-stlfactor':
                stl_factor = float(check_next_argument(i))

            elif arg == '-useGUI':

                answer_yes = 'Yes'
                make_report_answer = PhotoScan.app.getString('Do you want to create a report file?', answer_yes)
                if make_report_answer == answer_yes:
                    make_report = True
                    report_filename = PhotoScan.app.getString('Choose a file name for the report', 'report_file_name')

                make_svg_answer = PhotoScan.app.getString(
                    'Do you want to create a a SVG-Image with image-measurements?',
                    answer_yes)
                if make_svg_answer == answer_yes:
                    make_svg = True
                    svg_filename = PhotoScan.app.getString('Choose a file name for the SVG-Image',
                                                           'svg_image_file_name')
                    svg_factor = PhotoScan.app.getInt(
                        'Select a magnification factor of the error-vector for the SVG-File',
                        40)
                    svg_cols = PhotoScan.app.getInt('Select the number of columns used to generate the overview image',
                                                    20)

                make_stl_answer = PhotoScan.app.getString(
                    'Do you want to create a STL-Mesh with Point-Error-Ellipsoids?', answer_yes)
                if make_stl_answer == answer_yes:
                    make_stl = True
                    stl_filename = PhotoScan.app.getString('Choose a file name for the STL-File', 'stl_file_name')
                    stl_factor = PhotoScan.app.getInt('Choose a magnification factor of the ellipsoid-axis', 100)

                export_ellipsoid_answer = PhotoScan.app.getString('Do you want to export a ellipsoid file?', answer_yes)
                if export_ellipsoid_answer == answer_yes:
                    export_ellipsoid = True

            elif arg == '-export_ellipsoids':
                export_ellipsoid = True

        project = I3_Project(chunk)

        if make_report:
            project.save_and_print_report(report_filename)
            PhotoScan.app.update()

        if make_svg:
            project.create_project_SVG(svg_filename, svg_factor, svg_cols)
            PhotoScan.app.update()

        if make_stl:
            project.export_STL(stl_filename, factor=stl_factor)
            PhotoScan.app.update()
        if export_ellipsoid:
            project.exportEllipsoids()
            PhotoScan.app.update()


    else:
        print("Please open a Project with completed photo alignment")
