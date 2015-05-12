import os

__author__ = 'philipp.atorf'

from analysis import I3_Point
from analysis import I3_Photo
from analysis import I3_Project
from analysis import SVG_Photo_Representation
from analysis import Peseudo_3D_intersection_adjustment
from analysis import Py_2_OpenScad
from analysis import STL_Handler
import analysis
import imp

imp.reload(analysis)
from analysis import I3_Photo
from analysis import I3_Point
from analysis import I3_Project
from analysis import SVG_Photo_Representation
from analysis import Peseudo_3D_intersection_adjustment
from analysis import Py_2_OpenScad
from analysis import STL_Handler

import sys
import unittest
import PhotoScan


# analysis.__dict__.clear()


p1 = I3_Point(coord_W=PhotoScan.Vector([0.5794213274747677, 1.0299438009060218, 7.661802396272032]),
              projection_I=PhotoScan.Vector([672.8680381754943, 1508.2357157087156]),
              coord_C=PhotoScan.Vector([0.0009213739394838983, 3.56654678465967e-05, 4.941896795797742]),
              measurement_I=PhotoScan.Vector([672.3115234375, 1508.2142333984375]),
              )
p2 = I3_Point(coord_W=PhotoScan.Vector([0.5885454295971861, 1.0351837514683544, 7.67401271573676]),
              projection_I=PhotoScan.Vector([698.624257342221, 1582.757549644876]),
              coord_C=PhotoScan.Vector([-0.00010325672377653524, 0.000130867965612362, 4.925811420721138]),
              measurement_I=PhotoScan.Vector([698.6868286132812, 1582.678466796875]),
              )


class TestMyPhoto(unittest.TestCase):
    photo = I3_Photo()
    photo.points.append(p1)
    photo.points.append(p2)


    def test_addPoint(self):
        photo = I3_Photo()
        photo.add_point(p1)

        self.assertEqual(photo.points[-1], p1)
        self.assertIs(len(photo.points), 1)

        photo.add_point()
        self.assertIsInstance(photo.points[-1], I3_Point)


    def test_calc_sigma(self):
        sigma = self.photo.sigma_I
        self.assertAlmostEqual(sigma.x, 0.395994834, 6)
        self.assertAlmostEqual(sigma.y, 0.057946469, 6)

        # self.assertAlmostEqual(error_quad.x, 0.313623818, 6)
        #self.assertAlmostEqual(error_quad.y, 0.006715587, 6)

    def test_calc_Cov_from_ErrorMatrix(self):
        errorMatrix = [[1.6, 1.7],
                       [0.6, 0.6],
                       [-0.4, -0.4],
                       [-1.4, -1.4],
                       [-0.3, -0.4]]

        cov = self.photo.calc_cov_from_error_matrix(errorMatrix)
        var_x = cov[0, 0]
        var_y = cov[1, 1]
        cov_xy = cov[0, 1]
        self.assertAlmostEqual(var_x, 1.026, 4)
        self.assertAlmostEqual(var_y, 1.10600, 4)
        self.assertAlmostEqual(cov_xy, 1.064, 4)

    def test_gestExtremError(self):
        maxError = self.photo.get_max_error()
        self.assertAlmostEqual(maxError.x, 0.5565147, 5)
        self.assertAlmostEqual(maxError.y, 0.0790828, 5)

    def test_createSVG(self):
        photo = I3_Photo()
        photo.add_point(p1)
        photo.add_point(p2)
        photo.label = "test_Label"
        p_bottom_left = PhotoScan.Vector((0, 2000))
        p_bottom_right = PhotoScan.Vector((2000, 2000))
        p_upper_left = PhotoScan.Vector((0, 0))
        p_upper_right = PhotoScan.Vector((2000, 0))
        photo.add_point(I3_Point(measurement_I=p_bottom_left, projection_I=p_bottom_left))
        photo.add_point(I3_Point(measurement_I=p_bottom_right, projection_I=PhotoScan.Vector((2001, 0))))
        photo.add_point(I3_Point(measurement_I=p_upper_left, projection_I=p_upper_left))
        photo.add_point(I3_Point(measurement_I=p_upper_right, projection_I=p_upper_right))

        class psSensor():
            height = 2000
            width = 2000

        class psCamera():
            sensor = psSensor()

        cam_dummy = psCamera()

        photo.photoScan_camera = cam_dummy
        # Optische Kontrolle des SVGs
        #print(photo.getPhotsSVG()[0].getXML())


    @classmethod
    def getPhotoforRasterTest(cls):
        photo = I3_Photo()
        photo.add_point(p1)
        photo.add_point(p2)
        photo.label = "test_Label"
        p_bottom_left = PhotoScan.Vector((0, 2000))
        p_bottom_right = PhotoScan.Vector((2000, 2000))
        p_upper_left = PhotoScan.Vector((0, 0))
        p_upper_right = PhotoScan.Vector((2000, 0))
        photo.add_point(I3_Point(measurement_I=p_bottom_left, projection_I=PhotoScan.Vector((-1, 2001))))
        photo.add_point(I3_Point(measurement_I=p_bottom_right, projection_I=PhotoScan.Vector((2001, 2001))))
        photo.add_point(I3_Point(measurement_I=p_upper_left, projection_I=PhotoScan.Vector((-1, -1))))
        photo.add_point(I3_Point(measurement_I=p_upper_right, projection_I=PhotoScan.Vector((2001, -1))))

        class psSensor():
            height = 2000
            width = 1990

        class psCamera():
            sensor = psSensor()

        cam_dummy = psCamera()

        photo.photoScan_camera = cam_dummy
        return photo


class TestMyPoint(unittest.TestCase):
    pass


class TestMyProject(unittest.TestCase):
    pho1 = I3_Photo()
    pho1.sigma_I = PhotoScan.Vector((2, 13))

    pho2 = I3_Photo()
    pho2.sigma_I = PhotoScan.Vector((5, 11))

    project = I3_Project()
    project.photos = [pho1, pho2]

    def test_calcGlobalSigma(self):
        rms_x, rms_y = self.project._get_RMS_4_all_photos()
        self.assertAlmostEqual(rms_x, 3.807886553, 6)
        self.assertAlmostEqual(rms_y, 12.04159458, 6)

    def test_createProjectSVG(self):
        pass
        #self.project.createProjectSVG()


class TestSVG_Photo_Representation(unittest.TestCase):
    def getSVGObject(self):
        photo = I3_Photo()
        photo.add_point(p1)
        photo.add_point(p2)
        photo.label = "test_Label"
        p_bottom_left = PhotoScan.Vector((0, 2000))
        p_bottom_right = PhotoScan.Vector((2000, 2000))
        p_upper_left = PhotoScan.Vector((0, 0))
        p_upper_right = PhotoScan.Vector((2000, 0))
        photo.add_point(I3_Point(measurement_I=p_bottom_left, projection_I=p_bottom_left))
        photo.add_point(I3_Point(measurement_I=p_bottom_right, projection_I=PhotoScan.Vector((2001, 2001))))
        photo.add_point(I3_Point(measurement_I=p_upper_left, projection_I=p_upper_left))
        photo.add_point(I3_Point(measurement_I=p_upper_right, projection_I=p_upper_right))

        class psSensor():
            height = 2000
            width = 2000

        class psCamera():
            sensor = psSensor()

        cam_dummy = psCamera()

        photo.photoScan_camera = cam_dummy
        return SVG_Photo_Representation([photo], 700)

    def test_get_raw_error_vector_svg(self):
        # Optische Kontrolle des SVGs
        # print(photo.getPhotsSVG()[0].getXML())
        svgPhoto = self.getSVGObject()

        # print(svgPhoto.get_raw_error_vector_svg(40)[0].getXML())


    def test_colormap(self):
        colormap = [0, 1, 2]
        true_cat = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        for i in range(1, 16):
            cat = SVG_Photo_Representation.get_color_4_value([0, 15], i, colormap)
            # print(i,cat)
            self.assertEqual(true_cat[i - 1], cat)
            # print(SVG_Photo_Representation.get_color_4_value([0,15],15))

    def test_get_category_ranges(self):
        colormap = [0, 1, 2]
        true_border = [1, 2, 3]

        cat_borders = SVG_Photo_Representation.get_categroy_ranges([0, 3], colormap)[0]
        self.assertEqual(true_border, [int(i) for i in cat_borders])
        # print(SVG_Photo_Representation.get_categroy_ranges([0,15]))

    def test_raster_legend(self):
        photo = I3_Photo()
        svg_photo = self.getSVGObject()
        svg_photo.set_count_legend(SVG_Photo_Representation.colormap, [1, 9])

        print(svg_photo.count_legend.getXML())


class TestPeseudo_3D_intersection_adjustment(unittest.TestCase):
    def test_get_jacobian_row_for_point(self):
        pass

    def test_eig(self):
        adju = Peseudo_3D_intersection_adjustment()
        m = PhotoScan.Matrix([[504, 360, 180], [360, 360, 0], [180, 0, 720]])
        eig_valu, eig_vec = adju._get_eigen_vel_vec(m)
        # print(eig_vec[0])
        # print(eig_vec[1])
        # print(eig_vec[2])
        # print(eig_valu)
        self.assertAlmostEqual(eig_valu[0], 910.06995, 4)
        self.assertAlmostEqual(eig_valu[1], 44.81966, 4)
        self.assertAlmostEqual(eig_valu[2], 629.11038, 4)

        self.assertAlmostEqual(eig_vec[0][0], -0.65580, 4)
        self.assertAlmostEqual(eig_vec[0][1], 0.64879, 4)
        self.assertAlmostEqual(eig_vec[0][2], 0.38600, 4)


class Test_Py_2_OpenScad(unittest.TestCase):
    def test_errorEllipse_from_eig(self):
        adju = Peseudo_3D_intersection_adjustment()
        m = PhotoScan.Matrix([[504, 360, 180], [360, 360, 0], [180, 0, 720]])
        m = PhotoScan.Matrix([[24.66697238419596, 11.102022651894911, 29.082023223173206],
                              [11.10202265189491, 10.052229488742526, 14.941828405336427],
                              [29.082023223173206, 14.941828405336427, 42.78791682803554]])
        eig_valu, eig_vec = adju._get_eigen_vel_vec(m)
        py2scad = Py_2_OpenScad()

        scad_string = py2scad.errorEllipse_from_eig(eig_vec, eig_valu, [0, 0, 0])

        ref_string = "render(){translate([ 0.000, 0.000, 0.000])" + \
                     "rotate([-7.740,-50.259,27.649])" + \
                     "scale([ 8.365, 2.068, 1.806])" + \
                     "sphere(r =  1.000)};\n"

        self.assertTrue(True)

        path = os.path.dirname(os.path.realpath(__file__))

        f = open(path + '\\scad_ell.scad', 'w')
        f.write(scad_string)
        f.close()

        # self.assertEqual(ref_string, scad_string)




class Test_STLHeandler(unittest.TestCase):
    def test_import(self):
        stl_heandler = STL_Handler()
        stl_heandler.importSTL("sp_exp_for_test.stl")

        self.assertEqual(3, len(stl_heandler.triangle[0]))
        self.assertEqual(0.0, stl_heandler.triangle[0][0].x)
        self.assertEqual(0.0, stl_heandler.triangle[0][0].y)
        self.assertEqual(-1.5, stl_heandler.triangle[0][0].z)

        self.assertEqual(0.0, stl_heandler.triangle[-1][2].x)
        self.assertEqual(0.38823, stl_heandler.triangle[-1][2].y)
        self.assertEqual(1.44889, stl_heandler.triangle[-1][2].z)

    def test_create_ellipsoid_stl(self):
        adju = Peseudo_3D_intersection_adjustment()
        m = PhotoScan.Matrix([[504, 360, 180], [360, 360, 0], [180, 0, 720]])
        m = PhotoScan.Matrix([[24.66697238419596, 11.102022651894911, 29.082023223173206],
                              [11.10202265189491, 10.052229488742526, 14.941828405336427],
                              [29.082023223173206, 14.941828405336427, 42.78791682803554]])

        eig_valu, eig_vec = adju._get_eigen_vel_vec(m)

        stl_handler = STL_Handler()
        stl_handler.importSTL()
        stl_handler.importSTL("sp_exp_for_test.stl")
        ellipsoid_stl = "solid OpenSCAD_Model\n"

        ellipsoid_stl += stl_handler.create_ellipsoid_stl(eig_vec, eig_valu, [10, 0, 0], 1, False)
        # print(ellipsoid_stl)
        self.assertEqual('vertex 11.997  0.635 -1.716', ellipsoid_stl.splitlines()[3])

        ellipsoid_stl += "endsolid OpenSCAD_Model"

        path = os.path.dirname(os.path.realpath(__file__))

        f = open(path + '\\stl_ell.stl', 'w')
        f.write(ellipsoid_stl)
        f.close()


if __name__ == '__main__':


    test_classes_to_run = [TestMyPhoto,
                           TestMyPoint,
                           TestMyProject,
                           TestSVG_Photo_Representation,
                           TestPeseudo_3D_intersection_adjustment,
                           Test_Py_2_OpenScad,
                           Test_STLHeandler]

    loader = unittest.TestLoader()

    suites_list = []
    for test_class in test_classes_to_run:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)

    big_suite = unittest.TestSuite(suites_list)

    runner = unittest.TextTestRunner(verbosity=2)
    results = runner.run(big_suite)
    # suite = unittest.TestLoader().loadTestsFromTestCase(MyTestCase)
    # unittest.TextTestRunner(verbosity=2).run(suite)

