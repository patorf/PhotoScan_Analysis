__author__ = 'philipp.atorf'

from analysis import MyPoint
from analysis import MyPhoto
from analysis import MyGlobalPoint
import analysis
import imp
imp.reload(analysis)
from analysis import MyPhoto
from analysis import MyPoint
from analysis import MyGlobalPoint

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
        photo.addPoint(p1)

        self.assertEqual(photo.points[-1], p1)
        self.assertIs(len(photo.points),1)

        photo.addPoint()
        self.assertIsInstance(photo.points[-1], MyPoint)


    def test_calc_sigma(self):
        sigma, error_quad, count = self.photo.calc_sigma()
        self.assertAlmostEqual(sigma.x, 0.395994834, 6)
        self.assertAlmostEqual(sigma.y, 0.057946469, 6)

        self.assertAlmostEqual(error_quad.x, 0.313623818, 6)
        self.assertAlmostEqual(error_quad.y, 0.006715587, 6)


class TestMyPoint(unittest.TestCase):
    def test_projectSigma_2_W(self):
        sigma_I = 0.400212071  # (sigma from p1 and p2)
        std_error_W = p1.projectSigma_2_W(sigma_I)
        self.assertAlmostEqual(std_error_W.x, -0.0005680685, 6)
        self.assertAlmostEqual(std_error_W.y, 0.0001009828, 6)
        self.assertAlmostEqual(std_error_W.z, 0.0003257899, 6)

        p1.sigma_I = 0.400212071
        std_error_W = p1.projectSigma_2_W()
        self.assertAlmostEqual(std_error_W.x, -0.0005680685, 6)
        self.assertAlmostEqual(std_error_W.y, 0.0001009828, 6)
        self.assertAlmostEqual(std_error_W.z, 0.0003257899, 6)


class TestGlobalPoint(unittest.TestCase):
    gp1 = MyGlobalPoint()
    gp1.points.append(p1)
    gp1.points.append(p1)
    gp1.points.append(p1)






if __name__ == '__main__':


    test_classes_to_run = [TestMyPhoto, TestMyPoint, TestGlobalPoint]

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

