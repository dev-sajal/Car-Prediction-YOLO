import tensorflow as tf
import nbformat
import pytest
import numpy as np

from ipynb.fs.full.AutonomousDrivingApplicationCarDetectionV1 import *


class TestModel:
    @pytest.fixture(autouse=True)
    def get_model(self):
        gradeX, gradeY, gradeZ = gradeF1()
        self.x = gradeX
        self.y = gradeY
        self.z = gradeZ

        iou = gradeF2()
        self.iou = iou
        #
        # a3 = gradeF3()
        # self.a3 = str(a3)

        x3, y3, z3 = gradeF3()
        self.x3 = x3
        self.y3 = y3
        self.z3 = z3

        x4, y4, z4 = gradeF4()
        self.x4 = x4
        self.y4 = y4
        self.z4 = z4

    def test_nn(self):
        # assert self.x1 == '1'
        assert self.x == '10.750582'
        assert self.y == '[ 8.426533   3.2713668 -0.5313436 -4.9413733]'
        assert self.z == '7'


        assert self.iou == '0.14285714285714285'

        assert self.x3 == '6.938395'
        assert self.y3 == '[-5.299932    3.1379814   4.450367    0.95942086]'
        assert self.z3 == '-2.2452729'

        assert self.x4 == '138.79124'
        assert self.y4 == '[1292.3297  -278.52167 3876.9893  -835.56494]'
        assert self.z4 == '54'

        # y = np.matrix([[ 1.4416984 , -0.24909666,  5.450499  , -0.2618962 , -0.20669907, 1.3654671 ],
        #                 [ 1.4070846 , -0.02573211,  5.08928   , -0.48669922, -0.40940708, 1.2624859 ]])
        # assert self.a == str(y)
        #
        # assert self.a3 == str(4.6648693)


print('done')
