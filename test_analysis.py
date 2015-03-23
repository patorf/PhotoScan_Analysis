__author__ = 'philipp.atorf'

from analysis import MyPoint
from analysis import MyPhoto
from analysis import MyGlobalPoint
from analysis import MyProject
import analysis
import imp
imp.reload(analysis)
from analysis import MyPhoto
from analysis import MyPoint
from analysis import MyGlobalPoint
from analysis import MyProject

import sys
import unittest
import PhotoScan


#analysis.__dict__.clear()


p1 = MyPoint(coord_W=PhotoScan.Vector([0.5794213274747677, 1.0299438009060218, 7.661802396272032]),
                 projection_I=PhotoScan.Vector([672.8680381754943, 1508.2357157087156]),
                 coord_C= PhotoScan.Vector([0.0009213739394838983, 3.56654678465967e-05, 4.941896795797742]),
                 error_W=PhotoScan.Vector([-0.000790515730769048, 0.00014052612581316737, 0.00045336436706833183]),
                 measurement_I=PhotoScan.Vector([672.3115234375, 1508.2142333984375]),
                 ratio_I_2_W=604.0027888776877)
p2 = MyPoint(coord_W=PhotoScan.Vector([0.5885454295971861, 1.0351837514683544, 7.67401271573676]),
                 projection_I=PhotoScan.Vector([698.624257342221, 1582.757549644876]),
                 coord_C= PhotoScan. Vector([-0.00010325672377653524, 0.000130867965612362, 4.925811420721138]),
                 error_W=PhotoScan.Vector([7.870425463041286e-05, 0.00011070577140159799, -9.663461031195197e-05]),
                 measurement_I=PhotoScan.Vector([698.6868286132812, 1582.678466796875]),
                 ratio_I_2_W=604.9411311441258)


class TestMyPhoto(unittest.TestCase):
    photo = MyPhoto()
    photo.points.append(p1)
    photo.points.append(p2)


    def test_addPoint(self):
        photo= MyPhoto()
        photo.add_point(p1)

        self.assertEqual(photo.points[-1], p1)
        self.assertIs(len(photo.points),1)

        photo.add_point()
        self.assertIsInstance(photo.points[-1], MyPoint)


    def test_calc_sigma(self):
        # sigma, error_quad, count = self.photo.calc_sigma()
        sigma = self.photo.calc_sigma()
        self.assertAlmostEqual(sigma.x, 0.395994834, 6)
        self.assertAlmostEqual(sigma.y, 0.057946469, 6)

        # self.assertAlmostEqual(error_quad.x, 0.313623818, 6)
        #self.assertAlmostEqual(error_quad.y, 0.006715587, 6)

    def test_gestExtremError(self):
        maxError = self.photo.get_max()
        self.assertAlmostEqual(maxError.x, 0.5565147, 5)
        self.assertAlmostEqual(maxError.y, 0.0790828, 5)

    def test_createSVG(self):
        photo = MyPhoto()
        photo.add_point(p1)
        photo.add_point(p2)
        photo.label = "test_Label"
        p_bottom_left = PhotoScan.Vector((0, 2000))
        p_bottom_right = PhotoScan.Vector((2000, 2000))
        p_upper_left = PhotoScan.Vector((0, 0))
        p_upper_right = PhotoScan.Vector((2000, 0))
        photo.add_point(MyPoint(measurement_I=p_bottom_left, projection_I=p_bottom_left))
        photo.add_point(MyPoint(measurement_I=p_bottom_right, projection_I=PhotoScan.Vector((2001, 0))))
        photo.add_point(MyPoint(measurement_I=p_upper_left, projection_I=p_upper_left))
        photo.add_point(MyPoint(measurement_I=p_upper_right, projection_I=p_upper_right))

        class psSensor():
            height = 2000
            width = 2000

        class psCamera():
            sensor = psSensor()

        cam_dummy = psCamera()

        photo.photoscanCamera = cam_dummy
        # Optische Kontrolle des SVGs
        #print(photo.getPhotsSVG()[0].getXML())


    def getPhotoforRasterTest(self):
        photo = MyPhoto()
        photo.add_point(p1)
        photo.add_point(p2)
        photo.label = "test_Label"
        p_bottom_left = PhotoScan.Vector((0, 2000))
        p_bottom_right = PhotoScan.Vector((2000, 2000))
        p_upper_left = PhotoScan.Vector((0, 0))
        p_upper_right = PhotoScan.Vector((2000, 0))
        photo.add_point(MyPoint(measurement_I=p_bottom_left, projection_I=PhotoScan.Vector((-1, 2001))))
        photo.add_point(MyPoint(measurement_I=p_bottom_right, projection_I=PhotoScan.Vector((2001, 2001))))
        photo.add_point(MyPoint(measurement_I=p_upper_left, projection_I=PhotoScan.Vector((-1, -1))))
        photo.add_point(MyPoint(measurement_I=p_upper_right, projection_I=PhotoScan.Vector((2001, -1))))

        class psSensor():
            height = 2000
            width = 1990

        class psCamera():
            sensor = psSensor()

        cam_dummy = psCamera()

        photo.photoscanCamera = cam_dummy
        return photo

    def test_getErrorRaster(self):
        photo = self.getPhotoforRasterTest()
        errorRaster = photo.get_error_raster(cols=5)

        # upper left
        self.assertTrue(errorRaster[0][0].x == -1.0)
        self.assertTrue(errorRaster[0][0].y == -1.0)

        #upper right
        self.assertTrue(errorRaster[0][4].x == 1.0)
        self.assertTrue(errorRaster[0][4].y == -1.0)

        #bottom left
        self.assertTrue(errorRaster[4][0].x == -1.0)
        self.assertTrue(errorRaster[4][0].y == 1.0)

        #bottom right
        self.assertTrue(errorRaster[4][4].x == 1.0)
        self.assertTrue(errorRaster[4][4].y == 1.0)

        self.assertTrue(errorRaster[2][2].x == 0)
        self.assertTrue(errorRaster[2][2].y == 0)


    def test_getCountRaster(self):
        photo = self.getPhotoforRasterTest()

        countRaster, min, max = photo.get_count_raster(cols=5)
        self.assertEqual(countRaster[0][0], 1)
        self.assertEqual(countRaster[0][4], 1)
        self.assertEqual(countRaster[4][0], 1)
        self.assertEqual(countRaster[4][4], 1)
        self.assertEqual(countRaster[1][1], 0)

        self.assertEqual(min, 0)
        self.assertEqual(max, 2)


class TestMyPoint(unittest.TestCase):
    def test_projectSigma_2_W(self):
        sigma_I = 0.400212071  # (sigma from p1 and p2)
        std_error_W = p1.project_sigma_2_W(sigma_I)
        self.assertAlmostEqual(std_error_W.x, -0.0005680685, 6)
        self.assertAlmostEqual(std_error_W.y, 0.0001009828, 6)
        self.assertAlmostEqual(std_error_W.z, 0.0003257899, 6)

        p1.sigma_I = 0.400212071
        std_error_W = p1.project_sigma_2_W()
        self.assertAlmostEqual(std_error_W.x, -0.0005680685, 6)
        self.assertAlmostEqual(std_error_W.y, 0.0001009828, 6)
        self.assertAlmostEqual(std_error_W.z, 0.0003257899, 6)


class TestGlobalPoint(unittest.TestCase):
    gp1 = MyGlobalPoint()
    gp1.points.append(p1)
    gp1.points.append(p1)
    gp1.points.append(p1)


class TestMyProject(unittest.TestCase):
    pho1 = MyPhoto()
    pho1.calc_sigma = lambda: PhotoScan.Vector((2, 13))

    pho2 = MyPhoto()
    pho2.calc_sigma = lambda: PhotoScan.Vector((5, 11))

    project = MyProject()
    project.photos = [pho1, pho2]
    def test_calcGlobalSigma(self):
        rms_x, rms_y = self.project.get_RMS_4_all_photos()
        self.assertAlmostEqual(rms_x, 3.807886553, 6)
        self.assertAlmostEqual(rms_y, 12.04159458, 6)

    def test_createProjectSVG(self):
        pass
        #self.project.createProjectSVG()

class TestAnalysis(unittest.TestCase):
    errorMatrix = [[1.6, 1.7],
                   [0.6, 0.6],
                   [-0.4, -0.4],
                   [-1.4, -1.4],
                   [-0.3, -0.4]]


    def test_calc_Cov_from_ErrorMatrix(self):
        cov = analysis.calc_Cov_from_ErrorMatrix(self.errorMatrix)
        var_x = cov[0, 0]
        var_y = cov[1, 1]
        cov_xy = cov[0, 1]
        self.assertAlmostEqual(var_x, 1.026, 4)
        self.assertAlmostEqual(var_y, 1.10600, 4)
        self.assertAlmostEqual(cov_xy, 1.064,4)



if __name__ == '__main__':


    test_classes_to_run = [TestMyPhoto, TestMyPoint, TestAnalysis, TestGlobalPoint, TestMyProject]

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

