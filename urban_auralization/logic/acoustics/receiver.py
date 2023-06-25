import logging
import matplotlib.pyplot as plt
import numpy as np
import glob
import sofa #for reading HRTR sofa-files
import librosa
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

from urban_auralization.logic.geometry import Point
from urban_auralization.definitions import ROOT_DIR

htrf_dir_sofa = ROOT_DIR +'/data/sofa/*.sofa'
_SOFA = glob.glob(htrf_dir_sofa)

class Receiver(Point):
    """
    Human receiver with ears
    """
    def __init__(self, x, y, z, azimuth = 0, elevation = 90, sofa = _SOFA[0]):
        """
        Parameters
        ----------
        azimuth : int/float
            az angle in degrees
        elevation : int/float
            elv angle in degrees
        sofa : string, optional, default = Neumann KU100 dummy head, Gauss-Legendre 2 deg
            filepath to .sofa-file
        """
        super().__init__(x, y, z)
        self.x = x
        self.y = y
        self.z = z
        self.azimuth = azimuth #"0 to 360"
        self.elevation = elevation #0 - 180, 90 is neutral (0 is up, 180 is down)
        self.sofa = sofa

    def get_postion(self):
        return [self.x, self.y, self.z]

    def get_direction(self):
        x = np.sin(np.deg2rad(self.elevation))*np.cos(np.deg2rad(self.azimuth))
        y = np.sin(np.deg2rad(self.elevation))*np.sin(np.deg2rad(self.azimuth))
        z = np.cos(np.deg2rad(self.elevation))
        return np.array([x,y,z])



    def _get_relative_incidence(self, inbound_pos):
        """
        Gets the relative incident direction from a point and returns
        a phi and theta (spherical coordinates)
        ----------
        inbound_pos : ndarray, np.array([x,y,z])
            the last position of the path before it reaches the receiver (self)

        Returns
        -------
        delta_phi_deg, delta_theta_deg
            direction of the incoming wave relative to the receivers orientation
            NB! phi ∈ [0,360), theta ∈ [0,180]
        """

        #find angle relative to receiver
        self_pos = np.array([self.x, self.y, self.z])
        if (self_pos == inbound_pos).all():
            logging.warning(msg="Receiver is at det same point as last image source, check input.")

        inbound_direction = inbound_pos - self_pos
        ix, iy, iz = inbound_direction[0], inbound_direction[1], inbound_direction[2]

        if ix > 0:
            phi_inbound = np.arctan(iy / ix)
        elif ix < 0 and iy >= 0:
            phi_inbound = np.arctan(iy / ix) + np.pi
        elif ix < 0 and iy < 0:
            phi_inbound = np.arctan(iy / ix) - np.pi
        elif ix == 0 and iy > 0:
            phi_inbound = np.pi / 2
        elif iz == 0 and iy < 0:
            phi_inbound = - np.pi / 2
        else:
            phi_inbound = None
            logging.info("source is above receiver")


        if iz > 0:
            theta_inbound = np.arctan(np.sqrt(ix**2+iy**2)/iz)
        elif iz < 0:
            theta_inbound = np.arctan(np.sqrt(ix ** 2 + iy ** 2) / iz) + np.pi
        elif iz == 0 and (ix or iy) != 0:
            theta_inbound = np.pi/2
        else:
            theta_inbound = None
            logging.info("Undefined theta")

        if phi_inbound is not None:
            while phi_inbound < 0:
                phi_inbound += 2*np.pi
            if phi_inbound == 2*np.pi:
                phi_inbound = 0

            delta_phi_deg = np.rad2deg(phi_inbound) - self.azimuth #azimuth
            while delta_phi_deg < 0:
                delta_phi_deg += 360

        else:
            delta_phi_deg = None

        if theta_inbound is not None:
            delta_theta_deg = np.rad2deg(theta_inbound) - self.elevation #elevation
        else:
            delta_theta_deg = None

        return delta_phi_deg, delta_theta_deg

    def get_hrir(self, inbound_pos, fs):
        """

        Parameters
        ----------
        inbound_pos : ndarray, np.array([x,y,z])
        fs : int
            sampling frequency in Hz

        Returns
        -------
        hrir : ndarray
            impulse response on the left and right channel, shape(2,n)
        """

        phi, theta = self._get_relative_incidence(inbound_pos) #phi is from 0 to 359, theta
        while (phi and theta) is None:
            phi, theta = self._get_relative_incidence(inbound_pos + 0.001)
        theta = theta-90

        #print(phi, theta)

        def find_closest(array, target):
            try:
                #finds the closest incidence to a target value
                array = np.asarray(array)
                idx = (np.abs(array-target)).argmin()
                return idx, array[idx]
            except:
                print(array, target)

        hrtf = sofa.Database.open(self.sofa)
        fs_sofa = hrtf.Data.SamplingRate.get_values()[0]

        # get availible positions from the sofa file
        positions = hrtf.Source.Position.get_values(system="spherical")  # az, elev, radius

        #find the closest
        az_idx, az_angle = find_closest(positions[:,0], phi)
        valid_elev_angles = positions[np.where(positions[:,0]==az_angle)]
        el_idx, el_angle = find_closest(valid_elev_angles[:,1], theta)

        l = hrtf.Data.IR.get_values({"M": az_idx + el_idx, "R": 0, "E": 0})
        r = hrtf.Data.IR.get_values({"M": az_idx + el_idx, "R": 1, "E": 0})

        H = np.zeros(shape=(len(l), 2))
        H[:,0], H[:,1] = l, r

        if fs != fs_sofa:
            H = librosa.core.resample(H.transpose(), fs_sofa, fs).transpose()

        return H

    def get_sofa_data(self):
        sofa.Database.open(self.sofa).Metadata.dump()










if __name__ == "__main__":
    r = Receiver(0,0,0, azimuth=0, elevation=0)

    inbound_pos1 = np.array([10,10,0]) # left
    inbound_pos2 = np.array([10,-10, 0])

    fs = 48000

    h1 = r.get_hrir(inbound_pos=inbound_pos1, fs = fs) #phi, theta
    h2 = r.get_hrir(inbound_pos = inbound_pos2, fs = fs)
    fig, (ax1, ax2) = plt.subplots(1,2, sharey=True, figsize = (11,5))

    ax1.plot(h1)
    ax1.legend(["left", "right"])
    ax1.set_ylabel("Sound pressure in Pa")
    ax1.set_xlabel("Time in samples")
    ax2.set_xlabel("Time in samples")
    ax2.plot(h2)
    ax2.legend(["left", "right"])
    ax1.grid()
    ax2.grid()
    plt.tight_layout()
    plt.show()






