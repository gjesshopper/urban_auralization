import logging
import matplotlib.pyplot as plt
import numpy as np
from urban_auralization.logic.acoustics import constants
from urban_auralization.logic.acoustics.air_absorption import air_absorption
import yulewalker as yw
from urban_auralization.logic.acoustics.abs_coeffs import AbsorptionCoeff

class Path():
    def __init__(self, points : np.ndarray = np.array([]), images_sources = None, reflection_planes = None, path_type = None):
        """

        Parameters
        ----------
        points : ndarray with points from source to receiver [[0,0,0], [1,2,3], [3,4,5],...]
        reflection_planes : sorted list of memory addresses to every reflection point in plane (if any), else None
        """
        self.points = points
        if reflection_planes is None:
            reflection_planes = []
        self.reflection_planes = reflection_planes

        if path_type not in ['d', 'r', 'td', 'sd']:
            logging.error("Path object constructed with invalid path type")
        self.path_type = path_type # 'd' : direct, 'r' : reflected, 'td' : top diffraction, 'sd' : side_diffraction


        """-------------------DIFFRACTION--------------------"""
        self.features = [] #only relevant for diffraction
        self.top_path = False # only relevant for diffraction



        if images_sources is None:
            self.image_sources = []
        else:
            self.image_sources = images_sources


    def __eq__(self, other):
        if other is None:
            return False
        if np.array_equal(self.points, other.points):
            return True
        return False

    def __str__(self):
        return f"{self.points}"

    def __len__(self):
        return len(self.points)

    def get_length(self):
        """

        Returns
        -------
        l : float, length of the path
        """
        l = 0
        for i in range(1, len(self.points)):
            x0, y0, z0 = self.points[i - 1][0], self.points[i - 1][1], self.points[i - 1][2]
            x1, y1, z1 = self.points[i][0], self.points[i][1], self.points[i][2]
            l += np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)
        return l

    def add_point(self, point):
        """
        Adds single point at the end of path
        """
        if len(self.points) == 0:
            self.points = [point]
        else:
            self.points = np.append(self.points, [point], axis=0)

    def add_points(self, points):
        """
        add a list of points to self.points
        """
        if len(self.points) == 0:
            self.points = points
        else:
            self.points = np.append(self.points, points, axis = 0)

    def reverse(self):
        self.points = self.points[::-1]

    def get_total_reflection_gain(self):

        total_gain = np.zeros(len(constants.f_third))
        for i, plane in enumerate(self.reflection_planes):
            abs = AbsorptionCoeff(plane.abs_coeff).extrapolate()
            if i == 0:
                total_gain = np.sqrt(1-abs)
            else:
                total_gain *= np.sqrt(1-abs)

        return total_gain

    def get_air_absorption(self):
        # air absorption
        rh = 10  # % relative humidity #todo: maybe should be changed based on scene...
        t = 20  # celcius
        ap = 1013 * 10 ** 2  # Pa ambient pressure

        alpha_air = air_absorption(freq=constants.f_third, relative_humidity=rh, temperature_c=t,
                                   ambient_pressure=ap) * self.get_length()  # in dB
        alpha_gain = 10 ** (-alpha_air / 20)
        return alpha_gain


    def diff_is_clockwise(self) -> bool:
        """
        This function is for side-diffracted paths!
        It checks if the path is clockwise/anti-clockwise (top-down view)
        around the respective features by calulating the signed area.
        Returns
        -------
        True/False
        """
        if not self.features:
            logging.critical(msg="Path has no features, cannot check if path clockwise.")


        points_2d = np.empty(shape=(len(self.points),2), dtype = float)
        points_2d[:,0] = self.points[:,0]
        points_2d[:,1] = self.points[:,1]

        signed_area = 0
        for i in range(len(points_2d)):
            x1 = points_2d[i][0]
            y1 = points_2d[i][1]
            if i == len(points_2d)-1:
                x2 = points_2d[0][0]
                y2 = points_2d[0][1]
            else:
                x2 = points_2d[i+1][0]
                y2 = points_2d[i+1][1]
            signed_area += (x1 * y2 - x2 * y1)

        signed_area /= 2  #if signed area < 0: clockwise


        return signed_area < 0

    def has_same_features(self, path):
        """
        This function's purpose is for sorting diffracted paths. Compares the features in self and other, regardless of order.
        Parameters
        ----------
        path

        Returns
        -------

        """
        if len(self.features) != len(path.features):
            return False

        eq_count = 0

        for feature1 in self.features:
            for feature2 in path.features:
                if feature1 == feature2:
                    eq_count += 1
        if eq_count == len(self.features):
            return True
        return False

    def get_filter_coeffs(self, doppler_factor, fs):
        """
        Calculates the attenuation of the path based on the path type.

        Returns
        -------
        gain
        """

        if self.path_type == 'd':
            #direct
            gain = (1*doppler_factor/(4*np.pi*self.get_length()))*np.ones(len(constants.f_third))
            gain *= self.get_air_absorption()
            return gain

        elif self.path_type == 'r':
            #reflected path, filter
            gain = (1*doppler_factor/(4*np.pi*self.get_length()))*self.get_total_reflection_gain()
            gain *= self.get_air_absorption()
            return gain


        elif self.path_type == 'td' or self.path_type == 'sd':
            #diffracted path
            k = 2*np.pi*np.array(constants.f_third) / constants.C_AIR # wave number
            lam = 2*np.pi / k #wave lenght

            Adiff = np.array([])

            for i in range(len(self.points)-2):
                s = self.points[i]
                r = self.points[-1]
                R = r-s
                RS = self.points[i+1] - s
                RR = r - self.points[i+1]
                R = np.sqrt(R[0]**2+R[1]**2+R[2]**2)
                RS = np.sqrt(RS[0]**2+RS[1]**2+RS[2]**2)
                RR = np.sqrt(RR[0]**2+RR[1]**2+RR[2]**2)

                heff = 1 #todo: invesigate how to calc heff (effective height)

                Nf = np.sign(heff)*2*(RS+RR-R)/lam
                if i == 0:
                    Adiff.resize(Nf.shape)

                #calc attenuation Adiff (gives answer in dB)
                for j, n in enumerate(Nf):
                    if n <= -0.25:
                        Adiff[j] += 0
                    elif -0.25 <= n < 0:
                        Adiff[j] += 6 - 12*np.sqrt(-n)
                    elif 0 <= n < 0.25:
                        Adiff[j] += 6 + 12*np.sqrt(n)
                    elif 0.25 <= n < 1:
                        Adiff[j] += 8 + 8*np.sqrt(n)
                    else: #n > 1
                        Adiff[j] += 16 + 10*np.log10(n)


            gain = 10**(-Adiff/20)*doppler_factor/(4*np.pi*self.get_length())
            gain *= self.get_air_absorption()
            return gain


if __name__ == "__main__":

    path = Path()
    path.add_point([3,4,5])
    print(path.points)
    path.add_points([[15,0,0], [2,3,4], [1,1,1]])
    path.add_point([1,1,1])
    print(path.points)
    print(path.get_length())






