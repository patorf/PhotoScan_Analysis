__author__ = 'philipp.atorf'

from analysis import I3_Point
from analysis import I3_Photo
from analysis import I3_GlobalPoint
from analysis import I3_Project
from analysis import SVG_Photo_Representation
from analysis import peseudo_3D_intersection_adjustment
import analysis
import imp
imp.reload(analysis)
from analysis import I3_Photo
from analysis import I3_Point
from analysis import I3_GlobalPoint
from analysis import I3_Project
from analysis import SVG_Photo_Representation
from analysis import peseudo_3D_intersection_adjustment



import sys
import unittest
import PhotoScan


#analysis.__dict__.clear()


p1 = I3_Point(coord_W=PhotoScan.Vector([0.5794213274747677, 1.0299438009060218, 7.661802396272032]),
                 projection_I=PhotoScan.Vector([672.8680381754943, 1508.2357157087156]),
                 coord_C= PhotoScan.Vector([0.0009213739394838983, 3.56654678465967e-05, 4.941896795797742]),
                 error_W=PhotoScan.Vector([-0.000790515730769048, 0.00014052612581316737, 0.00045336436706833183]),
                 measurement_I=PhotoScan.Vector([672.3115234375, 1508.2142333984375]),
                 ratio_I_2_W=604.0027888776877)
p2 = I3_Point(coord_W=PhotoScan.Vector([0.5885454295971861, 1.0351837514683544, 7.67401271573676]),
                 projection_I=PhotoScan.Vector([698.624257342221, 1582.757549644876]),
                 coord_C= PhotoScan. Vector([-0.00010325672377653524, 0.000130867965612362, 4.925811420721138]),
                 error_W=PhotoScan.Vector([7.870425463041286e-05, 0.00011070577140159799, -9.663461031195197e-05]),
                 measurement_I=PhotoScan.Vector([698.6868286132812, 1582.678466796875]),
                 ratio_I_2_W=604.9411311441258)


class TestMyPhoto(unittest.TestCase):
    photo = I3_Photo()
    photo.points.append(p1)
    photo.points.append(p2)


    def test_addPoint(self):
        photo = I3_Photo()
        photo.add_point(p1)

        self.assertEqual(photo.points[-1], p1)
        self.assertIs(len(photo.points),1)

        photo.add_point()
        self.assertIsInstance(photo.points[-1], I3_Point)


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

        photo.photoscanCamera = cam_dummy
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

        photo.photoscanCamera = cam_dummy
        return photo







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




class TestMyProject(unittest.TestCase):
    pho1 = I3_Photo()
    pho1.calc_sigma = lambda: PhotoScan.Vector((2, 13))

    pho2 = I3_Photo()
    pho2.calc_sigma = lambda: PhotoScan.Vector((5, 11))

    project = I3_Project()
    project.photos = [pho1, pho2]
    def test_calcGlobalSigma(self):
        rms_x, rms_y = self.project.get_RMS_4_all_photos()
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

        photo.photoscanCamera = cam_dummy
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
        adju = peseudo_3D_intersection_adjustment()
        m = PhotoScan.Matrix([[504, 360, 180], [360, 360, 0], [180, 0, 720]])
        eig_valu, eig_vec = adju.neweig(m)
        self.assertAlmostEqual(eig_valu[0], 910.06995, 4)
        self.assertAlmostEqual(eig_valu[1], 44.81966, 4)
        self.assertAlmostEqual(eig_valu[2], 629.11038, 4)

        self.assertAlmostEqual(eig_vec[0, 0], -0.65580, 4)
        self.assertAlmostEqual(eig_vec[0, 1], 0.64879, 4)
        self.assertAlmostEqual(eig_vec[0, 2], 0.38600, 4)

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


    test_classes_to_run = [TestMyPhoto,
                           TestMyPoint,
                           TestAnalysis,
                           TestMyProject,
                           TestSVG_Photo_Representation,
                           TestPeseudo_3D_intersection_adjustment]

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

