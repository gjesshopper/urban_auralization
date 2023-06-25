import logging
import glob
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
import os
import numpy as np
import time
import sounddevice as sd
from scipy import io, signal
from scipy.interpolate import interp1d
from scipy.fft import fft, ifft
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

from urban_auralization.logic.acoustics import constants
from urban_auralization.definitions import ROOT_DIR
from urban_auralization.logic.acoustics.image_source import ImageSource
from urban_auralization.logic.acoustics.harmonoise import Harmonoise
from urban_auralization.logic.geometry.path import Path






class Model():
    def __init__(self, scene = None, source_fs : int = 10, fs : int = 48000):
        """

        Parameters
        ----------
        scene : Scene object
            keeps info about buildings, sources, receivers etc
        source_fs : int, optinal, default = 10
            How many times should we sample the source position along its trajectory
        """
        self.source_fs = source_fs
        self.fs = fs
        self.scene = scene
        #add model to scene
        if self.scene is not None:
            self.scene.add_model(model=self)
            self.diffraction_model = Harmonoise(scene=self.scene, model = self)
            self.time_vec = np.arange(start=0, stop=self.scene.source.get_total_time(), step=1 / self.source_fs)




        #default settings
        self._settings = {"direct_sound" : True,
                         "specular_reflections" : True,
                         "ism_order" : 2,
                         "diffraction" : True,
                         "refraction" : False,
                         "output_directory" : "../data/results",
                         "source_fs" : self.source_fs,
                         "fs" : self.fs,
                         "auralizing_engine" : "mdr",
                          "only_shortest_diff" : False}

        self.results = {}

    @property
    def settings(self):
        return self._settings
    @settings.setter
    def settings(self, key, value):
        self._settings[key] = value

    def _elapsed_time(func):
        def wrapper(self):
            start = time.time()
            func(self)
            self.settings["elapsed_time"] = f"{round(time.time()-start,4)} seconds"

        return wrapper

    @_elapsed_time
    def start(self):
        """
        This function starts the calculations for an urban environment, and stores the results in the
        class attribute self.results
        Returns
        -------

        """

        #check inputs
        assert self.scene != None


        metadata = self.settings #todo: should have info about the scene as well

        self.results["metadata"] = metadata
        self.results["timeframes"] = {}


        for i, t in enumerate(self.time_vec):
            print(f"Starting calculation {i+1}/{len(self.time_vec)}")
            self.results["timeframes"][t] = {}
            self.results["timeframes"][t]["paths"] = {}
            if self.settings["direct_sound"] == True:
                self.results["timeframes"][t]["paths"]["direct"] = self._direct_sound(t)
            if self.settings["specular_reflections"] == True:
                self.results["timeframes"][t]["paths"]["specular"] = self._image_source_method(t, order = self.settings["ism_order"])
            if self.settings["diffraction"] == True:
                self.results["timeframes"][t]["paths"]["diffraction"] = self._diffraction(t)
            if self.settings["refraction"] == True:
                self.results["timeframes"][t]["paths"]["refraction"] = self._refraction(t)

    def _direct_sound(self, t):
        #get direct sound
        direct = Path(points=np.array([self.scene.source.get_position(t=t), self.scene.receiver.get_postion()]), path_type='d')

        if self._obstructed(path = direct) == False:
            return direct


    def _image_source_method(self, t, order : int = 1):
        """
        Finds the specular reflections with the image source method
        Returns
        -------
        paths : ndarray
            paths[0] = all paths (one path is a list of points) for 1st order reflection,
            paths[1] for 2nd order etc.
        """
        mreceiver = self.scene.receiver.get_postion()
        msource = self.scene.source.get_position(t=t)

        all_is = [] #all_is[1]
        all_is.append([ImageSource(pos=msource, mirror_plane=None, mirror_source=None,order=0)])
        paths = []
        count_is = 0
        count_valid_is = 0

        for n in range(1,order+1):
            all_is_order_n = []
            path_order_n = []
            for source in all_is[n - 1]:
                for building in self.scene.features:
                    for plane in building.get_all_planes():
                        count_is+=1
                        is_cand = ImageSource(pos=plane.mirror(source.pos), mirror_plane=plane, mirror_source = source, order=n)
                        all_is_order_n.append(is_cand)
                        path = is_cand.get_visible_path(receiver=mreceiver, n = is_cand.order)
                        if path is not None:
                            #do obstruction check:
                            if self._obstructed(path) == False:
                                #if not obstructed, add all refl planes to path
                                ms = is_cand
                                while ms.mirror_source:
                                    path.reflection_planes.append(ms.mirror_plane)
                                    ms = ms.mirror_source
                                path.reverse()
                                count_valid_is += len(path)-2
                                path_order_n.append(path)

            paths.append(path_order_n)
            all_is.append(all_is_order_n)
        else:
            logging.info(msg=f"Completed ism calculations for paths up to order {order}, at time = {round(t, 2)} seconds")
        return paths

    def _obstructed(self, path):
        """
        Returns True if path is obstructed, otherwise, False
        Parameters
        ----------
        path : path object

        Returns
        -------
        True/False
        """

        #path is n points so have to extract line segments of path
        for i in range(1,len(path.points)):
            line_segment = [path.points[i-1], path.points[i]]
            for plane in self.scene.get_all_planes():
                if plane.bounding_box:
                    if plane.bounding_box.intersect(line_segment) is None:
                        logging.debug("Plane has bounding box, but no intersection point")
                        pass
                    else:
                        logging.debug("Plane has bounding box, and intersection point")
                        return True
                else:
                    if plane.intersect(line_segment) is None:
                        logging.debug("Plane has no bounding box, and no intersection point")
                        pass
                    else:
                        logging.info("Plane has no bounding box, and intersection point")
                        return True
        return False

    def plot_path_groups(self):
        path_groups = self._get_path_groups()
        for path_group in path_groups:
            for t, path in path_group.items():
                if path is not None:
                    plt.plot(path.points[:,0], path.points[:,1])

            plt.show()

    def _diffraction(self, t : float):
        return self.diffraction_model.get_diffracted_paths(t = t)

    def _refraction(self, t):
        pass

    def _moves_away(self, tau) -> bool:
        """

        Parameters
        ----------
        tau : emission time

        Returns
        -------
        True if source moves away from receiver, False otherwise
        """
        delta_t = 0.001

        receiver = self.scene.receiver.get_postion()
        pos1 = self.scene.source.get_position(tau)
        pos2 = self.scene.source.get_position(tau+delta_t)

        path1 = Path(points = [pos1, receiver]).get_length()
        path2 = Path(points = [pos2, receiver]).get_length()

        if path1 < path2:
            return True
        else:
            return False

    def _get_longest_path_length(self):
        path_lengths = []

        if self.results["metadata"]["direct_sound"]:
            for t in self.time_vec:
                if self.results["timeframes"][t]["paths"]["direct"] is not None:
                    path_lengths.append(self.results["timeframes"][t]["paths"]["direct"].get_length())
        if self.results["metadata"]["specular_reflections"]:
            for n in range(self.results["metadata"]["ism_order"]):
                for t in self.time_vec:
                    for path in self.results["timeframes"][t]["paths"]["specular"][n]:
                        path_lengths.append(path.get_length())
        if self.results["metadata"]["diffraction"]:
            for t in self.time_vec:
                for path in self.results["timeframes"][t]["paths"]["diffraction"]:
                    path_lengths.append(path.get_length())

        if path_lengths:
            return max(path_lengths)
        else:
            return 0

    def _render_audio_mdr(self, sig = None, plot : bool = True):
        """
        Renders a binaural audio signal with a variable delay line.
        Parameters
        ----------
        sig : signal

        Returns
        -------

        """

        if sig is None:
            sig = self.scene.source.signal

        [q, fs_q] = sf.read(sig)
        # check if stereo and collapse to mono if it is
        if q.ndim > 1:
            q = np.mean(q, axis=1)

        # check sampling frequency
        if fs_q != self.fs:
            q = librosa.core.resample(q.transpose(), fs_q, self.fs).transpose()

        signal_time = len(q) / self.fs
        longest_prop_delay = self._get_longest_path_length() / constants.C_AIR
        total_simulation_time = self.scene.source.get_total_time() + longest_prop_delay

        # make sure signal is long enough for simulation
        while signal_time < total_simulation_time:
            q = np.concatenate((q, q))  # doubles the signal
            signal_time = len(q) / self.fs

        #anti-alias filter:
        q = self._anti_alias_filter(q, r = 1)

        # this is our accumulated block write!
        master_acum_bus = np.zeros(shape = (round(self.fs * total_simulation_time) + 2 * self.fs,2)) ####2

        block_size = int(self.fs/self.source_fs) # in samples
        buffer = block_size + round(longest_prop_delay * self.fs) + 10000

        #interpolating function
        int_func = interp1d(np.arange(len(q)), q)

        #get path groups (paths that share refl. planes etc.)
        path_groups = self._get_path_groups()


        for ii, path_group in enumerate(path_groups):
            prev_end_idx = 0
            hrirs = {}
            filters = {}
            temp_acum_bus = np.zeros(round(self.fs * total_simulation_time) + 2 * self.fs)

            for i, t in enumerate(self.time_vec, start=1):
                if t not in path_group.keys() or path_group[t] is None:
                    continue
                path = path_group[t]


                # find amp from Greens function
                R = path.points[1] - path.points[0]
                R_length = np.sqrt(R[0] ** 2 + R[1] ** 2 + R[2] ** 2)
                vs = self.scene.source.get_vs(t)
                vs_length = np.sqrt(vs[0] ** 2 + vs[1] ** 2 + vs[2] ** 2)
                alpha = np.arccos((np.dot(vs, R)) / (R_length * vs_length))  # in radians
                path_length = path.get_length()  # not necessarily same length as R!!

                doppler_factor = 1 / (1 - vs_length * np.cos(alpha) / constants.C_AIR)

                #propagation delay in samples (not int!)
                ndelay = path_length / constants.C_AIR*self.fs

                #find wind
                vm = np.array([0,0,0])
                if self.scene.wind is not None:
                    vm = self.scene.wind.get_v_m()

                rn = R / np.linalg.norm(R)
                #find readout block size to auralize the doppler shift
                r = (1 + np.dot(rn, vm)/constants.C_AIR)/ (1 - np.dot(rn, vs-vm) / constants.C_AIR)
                read_out_block_size = block_size * r

                #keep track of indices/prev indices
                if i == 1 or self.time_vec[i-2] not in path_group.keys() or path_group[self.time_vec[i-2]] is None:
                    start_idx = round(t*self.fs - ndelay - read_out_block_size + buffer)
                else:
                    start_idx = prev_end_idx
                end_idx = round(start_idx + read_out_block_size)
                prev_end_idx = end_idx

                #interpolate q at samlping points in n_int
                n_int = np.linspace(start_idx, end_idx, block_size)
                q_dopplershifted = int_func(n_int)

                #get hrir and filters and put in dicts for later
                hrirs[t] = self.scene.receiver.get_hrir(inbound_pos=path.points[-2], fs=self.fs)
                gain = path.get_filter_coeffs(doppler_factor=doppler_factor, fs=self.fs)
                filters[t] = gain

                # write to temp accum bus!
                temp_acum_bus[round(t * self.fs):round(t * self.fs) + block_size] += q_dopplershifted

            #apply filters, hrirs
            temp_acum_bus = self.apply_filters_fft_ifft(temp_acum_bus, filters, block_size, 144) # not the cause!!!
            temp_acum_bus = self.apply_hrirs(temp_acum_bus,hrirs,block_size,144)

            #put signal on master bus
            master_acum_bus[:len(temp_acum_bus)] += temp_acum_bus

        if plot:
            plt.plot(master_acum_bus)
            plt.show()

        return master_acum_bus

    def _anti_alias_filter(self, monaudio, r):
        """
        General pre anti-aliasing filter to suppress artefacts
        r_max : max resampling factor
        Returns
        -------

        """
        cutoff = 16e3 # Hz
        b, a = signal.butter(N=4, Wn=cutoff,btype='low',fs = self.fs)
        return signal.filtfilt(b, a, monaudio)

    def apply_hrirs(self, x : np.ndarray, hrirs : dict, blocksize : int, noverlap : int):
        """
        Applies hrirs on the composed signal from a path_group
        Parameters
        ----------
        x : signal
        hrirs
        noverlap : overlap in samples (on one side)
        blocksize : size of the fixed size blocks

        Returns
        -------
        sig : binaural signal, ndim = (len(signal), 2)
        """

        sig = np.zeros(shape=(len(x), 2))

        t_vec = list(hrirs.keys())

        for i, t in enumerate(t_vec, start=1):

            if i == 1:
                start_idx = round(t * self.fs)
                end_idx = start_idx + blocksize + noverlap
            else:
                start_idx = round(t*self.fs) - noverlap
                end_idx = start_idx + blocksize + noverlap

            block = x[start_idx:end_idx]
            #convolve and window
            window = signal.windows.tukey(M=end_idx-start_idx, alpha=2 * noverlap / (end_idx-start_idx))

            L = np.multiply(window, signal.fftconvolve(hrirs[t][:, 0], block)[:(end_idx-start_idx)])
            R = np.multiply(window, signal.fftconvolve(hrirs[t][:, 1], block)[:(end_idx-start_idx)])
            sig[start_idx:end_idx, 0] += L[:end_idx - start_idx]
            sig[start_idx:end_idx, 1] += R[:end_idx - start_idx]

        return sig


    def apply_filters(self, x : np.ndarray, iir_coeffs : dict, blocksize : int, noverlap : int):
        """
        Applies filters on the composed signal from a path_group
        Parameters
        ----------
        x : signal
        iir_coeffs : (a, b) (tuple)
        noverlap : overlap in samples (on one side)
        blocksize : size of the fixed size blocks

        Returns
        -------
        sig : binaural signal, ndim = (len(signal), 2)
        """

        sig = np.zeros(shape=(len(x), 2))

        t_vec = list(iir_coeffs.keys())

        for i, t in enumerate(t_vec, start=1):

            if i == 1:
                start_idx = round(t * self.fs)
                end_idx = start_idx + blocksize + noverlap
            else:
                start_idx = round(t*self.fs) - noverlap
                end_idx = start_idx + blocksize + noverlap

            block = x[start_idx:end_idx]
            #convolve and window
            window = signal.windows.tukey(M=end_idx-start_idx, alpha=2 * noverlap / (end_idx-start_idx))

            L = np.multiply(window, signal.filtfilt(iir_coeffs[t][1], iir_coeffs[t][0], block[:,0]))
            R = np.multiply(window, signal.filtfilt(iir_coeffs[t][1], iir_coeffs[t][0], block[:,1]))
            sig[start_idx:end_idx, 0] += L[:end_idx - start_idx]
            sig[start_idx:end_idx, 1] += R[:end_idx - start_idx]

        return sig

    def apply_filters_fft_ifft(self, x : np.ndarray, freq_resps : dict, blocksize : int, noverlap : int, plot = False):
        sig = np.zeros(len(x))

        t_vec = list(freq_resps.keys())
        if plot:
            ax = plt.figure(figsize=(10,10)).add_subplot(projection='3d')

        dist = np.linspace(-80,80, len(t_vec))

        for i, t in enumerate(t_vec, start=1):

            if i == 1:
                start_idx = round(t * self.fs)
                end_idx = start_idx + blocksize + noverlap
            else:
                start_idx = round(t * self.fs) - noverlap
                end_idx = start_idx + blocksize + noverlap # this, however i think is wrong...

            block = x[start_idx:end_idx]


            block_fft = fft(block)


            fr_int = interp1d(constants.f_third, freq_resps[t], fill_value=(freq_resps[t][0], freq_resps[t][-1]), bounds_error=False)
            fr_desired = np.linspace(0,self.fs//2, len(block_fft)//2) # can only filter up to fs/2 so interpolate
            fr_os = fr_int(fr_desired) # one sided freq resp of filter
            fr_ts = np.concatenate([fr_os, np.flip(fr_os, axis = 0)], dtype=complex)
            while len(fr_ts) < len(block_fft):
                fr_ts = np.append(fr_ts, fr_ts[-1])


            if plot:

                y = np.full(len(fr_os), dist[i-1])
                ax.plot(fr_desired, y, 20*np.log10(fr_os/2e-5))
                a = ax.get_yticks().tolist()

                ax.set_xlabel("Frequency in Hz")
                ax.set_ylabel("x in meters")
                ax.set_zlabel("Sound pressure level in dB (re. 20 $\mu$Pa)")
                ax.set_xticks([2500,5000,10000,20000], ["2.5k","5k","10k","20k"])
                #ax.set_yticks([2,4,6,8,10])

                #fig, axs = plt.subplots(1,2)
                #axs[0].plot(fr_desired, fr_os)
                #axs[1].plot(fr_ts)
                #plt.show()
            block_fft_attenuated = np.multiply(block_fft, fr_ts)
            block_filtered = ifft(block_fft_attenuated)


            # convolve and window
            window = signal.windows.tukey(M=end_idx - start_idx, alpha=2 * noverlap / (end_idx - start_idx))

            windowed_block = np.multiply(window, block_filtered)

            sig[start_idx:end_idx] += np.real(windowed_block)
            #sig[start_idx:end_idx, 1] += R[:end_idx - start_idx]
        if plot:
            plt.show()
        return sig

    def _render_audio_sdr(self, sig=None, plot: bool = True):
        """
        Renders a binaural audio signal with a variable delay line.
        Parameters
        ----------
        sig : signal

        Returns
        -------

        """

        if sig is None:
            sig = self.scene.source.signal

        [q, fs_q] = sf.read(sig)
        # check if stereo and collapse to mono if it is
        if q.ndim > 1:
            q = np.mean(q, axis=1)

        # check sampling frequency
        if fs_q != self.fs:
            q = librosa.core.resample(q.transpose(), fs_q, self.fs).transpose()

        signal_time = len(q) / self.fs
        longest_prop_delay = self._get_longest_path_length() / constants.C_AIR
        total_simulation_time = self.scene.source.get_total_time() + longest_prop_delay

        # make sure signal is long enough for simulation
        while signal_time < total_simulation_time:
            q = np.concatenate((q, q))  # doubles the signal
            signal_time = len(q) / self.fs

        # this is our accumulated block write!
        master_acum_bus = np.zeros(shape=(round(self.fs * total_simulation_time) + 2 * self.fs, 2))

        block_size = int(self.fs / self.source_fs)  # in samples
        buffer = block_size + round(longest_prop_delay * self.fs) + 10000

        # interpolating function
        #int_func = interp1d(np.arange(len(q)), q)

        # get path groups (paths that share refl. planes etc.)
        path_groups = self._get_path_groups()

        for ii, path_group in enumerate(path_groups):
            prev_end_idx = 0
            hrirs = {}
            filters = {}
            temp_acum_bus = np.zeros(round(self.fs * total_simulation_time) + 2 * self.fs)

            for i, t in enumerate(self.time_vec, start=1):
                if t not in path_group.keys() or path_group[t] is None:
                    continue
                path = path_group[t]

                # find amp from Greens function
                R = path.points[1] - path.points[0]
                R_length = np.sqrt(R[0] ** 2 + R[1] ** 2 + R[2] ** 2)
                vs = self.scene.source.get_vs(t)
                vs_length = np.sqrt(vs[0] ** 2 + vs[1] ** 2 + vs[2] ** 2)
                alpha = np.arccos((np.dot(vs, R)) / (R_length * vs_length))  # in radians
                path_length = path.get_length()  # not necessarily same length as R!!

                doppler_factor = 1 / (1 - vs_length * np.cos(alpha) / constants.C_AIR)

                # amp = (1 * doppler_factor / (4 * np.pi * path_length)) * (constants.REFL_COEFF**(len(path) - 2))
                # todo: this should be a filter for reflections and diffraction..
                #amp = (1 * doppler_factor / (4 * np.pi * path_length))  # * np.prod(path.get_total_reflection_gain())

                # propagation delay in samples (not int!)
                ndelay = path_length / constants.C_AIR * self.fs

                # find wind
                vm = np.array([0, 0, 0])
                if self.scene.wind is not None:
                    vm = self.scene.wind.get_v_m()

                #rn = R / np.linalg.norm(R)
                # find readout block size to auralize the doppler shift
                # r = 1 /(1 - np.dot(r_n, vs) / constants.C_AIR
                #r = (1 + np.dot(rn, vm) / constants.C_AIR) / (1 - np.dot(rn, vs - vm) / constants.C_AIR)
                #read_out_block_size = block_size * r

                # keep track of indices/prev indices
                if i == 1 or self.time_vec[i - 2] not in path_group.keys() or path_group[self.time_vec[i - 2]] is None:
                    start_idx = round(t * self.fs - ndelay - block_size + buffer)
                else:
                    start_idx = prev_end_idx
                end_idx = round(start_idx + block_size)
                prev_end_idx = end_idx

                # interpolate q at samlping points in n_int
                #n_int = np.linspace(start_idx, end_idx, block_size)
                q_block = q[start_idx:end_idx]

                # get hrir and put in dict for later
                hrirs[t] = self.scene.receiver.get_hrir(inbound_pos=path.points[-2], fs=self.fs)

                # fade signal if block before/after is empty??

                # filter signal
                # todo: something wrong with the gain for refl and direct...
                gain = path.get_filter_coeffs(doppler_factor=doppler_factor, fs=self.fs)
                filters[t] = gain
                # y = signal.filtfilt(b, a, q_dopplershifted, method = 'pad', padlen=500, padtype="odd")
                # plt.plot(q_dopplershifted)
                # plt.plot(41*y, linewidth = 2)
                # plt.grid()
                # plt.show()

                # write to temp accum bus!
                temp_acum_bus[round(t * self.fs):round(t * self.fs) + block_size] += q_block  # *amp


            # apply hrir
            temp_acum_bus = self.apply_filters_fft_ifft(temp_acum_bus, filters, block_size, 144)
            temp_acum_bus = self.apply_hrirs(temp_acum_bus, hrirs, block_size, 144)

            # temp_acum_bus = \

            # put signal on master bus
            master_acum_bus[:len(temp_acum_bus)] += temp_acum_bus

        if plot:
            plt.plot(master_acum_bus)
            plt.show()
        # apply doppler

        t_current = np.linspace(0, self.scene.source.get_total_time()-1, round(self.scene.source.get_total_time()*self.fs)-1)
        master_acum_int_l = interp1d(t_current, master_acum_bus[:len(t_current),0], fill_value=(0,0), bounds_error=False)
        master_acum_int_r = interp1d(t_current, master_acum_bus[:len(t_current),1], fill_value = (0,0), bounds_error=False)
        t_int = []
        p2 = np.array(self.scene.receiver.get_postion())
        for tii in t_current:
            p1 = np.array(self.scene.source.get_position(tii))
            ri = p2 - p1 #direct sound vector
            ril = np.sqrt(ri[0]**2+ri[1]**2+ri[2]**2) #length
            t_int.append(tii - ril / constants.C_AIR)

        t_int = np.array(t_int)
        master_acum_bus_l = master_acum_int_l(t_int)
        master_acum_bus_r = master_acum_int_r(t_int)

        master_acum_bus = np.zeros(shape=(len(master_acum_bus_r), 2))
        master_acum_bus[:, 0] = master_acum_bus_l
        master_acum_bus[:, 1] = master_acum_bus_r

        return master_acum_bus
    def _get_path_groups(self):
        """
        Sorts all calculated paths into groups. Paths with simililar refl. planes/objects make up one path group.
        Ex. direct path for all timeframes make up one group, all specular refl. that first hits building 1 then 3
        for all timeframes is another path group. There can also be single paths in one group
        Returns
        -------
        path_groups
        """
        #todo: is this really doing what we want?? Perhaps it should istead return path groups that are "time-continous" i.e.
        # if direct path is occluded somewhere along its trajectory, the paths before and after make up one path group respectively??

        if not self.results:
            self.start()

        print("Starting sorting of paths...")

        path_groups = [] # list of dictionaries, {key : value} = {t : path}

        #sort direct paths
        if self.settings["direct_sound"]:
            direct_path_group = {}
            for t in self.time_vec:
                direct_path_group[t] = self.results["timeframes"][t]["paths"]["direct"]
            if any(elem is None for elem in direct_path_group.values()):
                if self.settings["diffraction"] and self.settings["only_shortest_diff"]:
                    #find at what times we have no direct path:
                    t_fill = []
                    for tf, value in direct_path_group.items():
                        if value == None:
                            t_fill.append(tf)
                    #find the diff path closest to direct path in length for every t in t t_fill
                    fill_paths = {}
                    for tg in t_fill:
                        phantom_dct =  Path(points=np.array([self.scene.source.get_position(t=tg), self.scene.receiver.get_postion()]), path_type='d')
                        d_length = phantom_dct.get_length()
                        smallest_diff = 0
                        path_cand = None
                        idx = 0
                        for i, path in enumerate(self.results["timeframes"][tg]["paths"]["diffraction"]):
                            if i == 0:
                                smallest_diff = abs(path.get_length() - d_length)
                                path_cand = path
                                idx = i
                                continue
                            if abs(path.get_length()-d_length) < smallest_diff:
                                smallest_diff = abs(path.get_length()-d_length)
                                path_cand = path
                                idx = i
                        fill_paths[tg] = path_cand
                        del self.results["timeframes"][tg]["paths"]["diffraction"][idx]
                    direct_path_group.update(fill_paths)
                else:
                    filtered = {k: v for k, v in direct_path_group.items() if v is not None}
                    direct_path_group.clear()
                    direct_path_group.update(filtered)


            path_groups.append(direct_path_group)
            #return path_groups


        if self.settings["specular_reflections"]:

            #sort specular refl. paths
            paths = self.results["timeframes"].copy()

            for n in range(self.results["metadata"]["ism_order"]):
                for i, t in enumerate(self.time_vec, start = 1):
                    for j, path in enumerate(paths[t]["paths"]["specular"][n]):
                        path_group = {}
                        path_group[t] = path
                        for ii, ti in enumerate(self.time_vec[i:]):
                            for jj, pathi in enumerate(paths[ti]["paths"]["specular"][n]):
                                if path.reflection_planes == pathi.reflection_planes:
                                    path_group[ti] = pathi
                                    del paths[ti]["paths"]["specular"][n][jj]

                        path_groups.append(path_group)
                        del paths[t]["paths"]["specular"][n][j]


        if self.settings["diffraction"] and not self.settings["only_shortest_diff"]:

            paths = self.results["timeframes"].copy()
            for i, t in enumerate(self.time_vec, start = 1):
                for path in paths[t]["paths"]["diffraction"]:
                    if path.path_type == "td":
                        continue
                    path_group = {}
                    path_group[t] = path
                    for ii, ti in enumerate(self.time_vec[i:]):
                        for jj, pathi in enumerate(paths[ti]["paths"]["diffraction"]):
                            if pathi.path_type == "td":
                                continue

                            if path.has_same_features(pathi):

                                if path.diff_is_clockwise() == pathi.diff_is_clockwise():
                                    path_group[ti] = pathi

                                    del paths[ti]["paths"]["diffraction"][jj] #might be this
                    path_groups.append(path_group)


        return path_groups

    def _render_audio_linear_fir(self):
        """
        Renders audio from scene with fir convolutions in linear time steps.
        Cannot render audio with Doppler effect.


        Parameters
        ----------
        Returns
        -------
        stereo_audio : ndarray
            binaural_audio

        """
        sig = self.scene.source.signal
        [x, fs_x] = sf.read(sig)
        #check if stereo and collapse to mono if it is
        if x.ndim > 1:
            x = np.mean(x, axis=1)

        #check sampling frequency
        if fs_x != self.fs:
            x = librosa.core.resample(x.transpose(), fs_x, self.fs).transpose()

        signal_time  = len(x) / self.fs

        total_simulation_time = self.scene.source.get_total_time()

        while signal_time < total_simulation_time:
            x = np.concatenate((x,x)) #doubles the signal
            signal_time = len(x)/self.fs

        #pick out equally many samples from the signal as the source moves
        x_temp = x[:int(self.fs*total_simulation_time)]

        #now x has the lenght we want and we can get the window size
        #where each ir will be convolved


        window_size = int(self.fs/2/self.source_fs*2)

        n_windows = int(len(x_temp)/window_size)

        x = x[:n_windows*window_size]

        n_overlap = 144 #one sided overlap
        #ndarray, each frame is a list containing a part of x
        frames = librosa.util.frame(x=x, frame_length = window_size + 2*n_overlap, hop_length = window_size, axis = 0)

        # allocate the auralized audio
        stereo_audio = np.zeros(shape=(len(x), 2))

        for i, frame in enumerate(frames,start=1):

            ir = self._get_impulse_response(t=self.time_vec[i - 1], fs = self.fs)

            #have to trim the leadning zeros, NB!! Has to trim the l/r channels at same idx! else itd lost...
            ir = self._trim_leading_zeros(ir)

            l = signal.fftconvolve(frame, ir[:,0])
            r = signal.fftconvolve(frame, ir[:,1])

            l = l[:len(frame)]
            r = r[:len(frame)]

            #check if empty impulse response
            if len(l) == 0:
                l = r = np.zeros(len(frame))

            window = signal.windows.tukey(M=len(l), alpha=2*n_overlap/len(l))

            ix_l = np.multiply(l,window)
            ix_r = np.multiply(r,window)

            if i == 1:
                start = 0
                stop = len(frame)

            else:
                start = (i - 1) * (len(frame) - n_overlap)
                stop = start + len(frame)


            #last iteration is not same size right...
            if len(stereo_audio[start:stop, 0]) != len(ix_l):
                continue

            stereo_audio[start:stop, 0] += ix_l
            stereo_audio[start:stop, 1] += ix_r
        plt.plot(stereo_audio)
        plt.plot()

        return stereo_audio

    def _get_impulse_response(self, t, fs : int):
        """
        Gets the impulse response in the receiver position given in self.scene for ONE timeframe.

        Parameters
        ----------
        t : float
            current timeframe
        paths : dict

        fs : int
            desired sampling frequency

        Returns
        -------
        h : ndarray
            multichannel impulse response for timeframe t with hrir info included
        """

        #a dict containing all paths (direct, specular, diffracted..)
        paths = self.results["timeframes"][t]["paths"]

        if "direct" in paths.keys():
            direct_path = paths["direct"]
        else:
            direct_path = None
        if "specular" in paths.keys():
            specular_paths = paths["specular"]
        else:
            specular_paths = None
        #todo: add diffracted paths also

        def get_lenght(path):
            """

            Parameters
            ----------
            path : list of points (list)
                Ex. [[0,0,0],[1,1,1],...,[n,m,o]]

            Returns
            -------
            l : float
                lenght of path
            """
            l = 0
            for i in range(1,len(path)):
                x0, y0, z0 = path.points[i-1][0], path.points[i-1][1], path.points[i-1][2]
                x1, y1, z1 = path.points[i][0], path.points[i][1], path.points[i][2]
                l+= np.sqrt((x1-x0)**2+(y1-y0)**2+(z1-z0)**2)
            return l

        rp = np.sqrt(0.9) #refl coeff

        distvec = []  #placeholder for distances
        ampvec = [] #placeholder for amplitudes
        hrirs = []
        if direct_path is not None:
            l = get_lenght(direct_path)
            distvec.append(l)
            ampvec.append(1/l)
            hrirs.append(self.scene.receiver.get_hrir(inbound_pos=direct_path.points[-2], fs = fs))
        if specular_paths is not None:
            for order in range(len(specular_paths)):
                for path in specular_paths[order]:
                    l = get_lenght(path)
                    distvec.append(l)
                    ampvec.append((1/l)*rp**(len(path)-2))
                    hrirs.append(self.scene.receiver.get_hrir(inbound_pos=path.points[-2], fs=fs))

        #todo: add diffraction

        #place every pulse at the nearest integer:
        ndelay = [round(x/constants.C_AIR * fs) + 1 for x in distvec]

        #similar to accumarray in matlab
        mono_ir = np.bincount(ndelay, weights=ampvec)
        #plt.plot(mono_ir)
        #plt.show()
        ir_L = np.array([])
        ir_R = np.array([])
        for i, sample in enumerate(ndelay):
            single_ir = np.zeros(sample)
            single_ir[-1] = mono_ir[sample]

            hi_L = signal.fftconvolve(single_ir, hrirs[i][:,0])
            hi_R = signal.fftconvolve(single_ir, hrirs[i][:,1])

            ir_L.resize(hi_L.shape)
            ir_R.resize(hi_R.shape)

            ir_L += hi_L
            ir_R += hi_R

        h = np.zeros(shape=(len(ir_L), 2))
        h[:, 0] += ir_L
        h[:, 1] += ir_R

        return h

    def _get_nonbinaural_impulse_response(self, t, fs: int = 44100):
        """
        Gets the impulse response in the receiver position given in self.scene for ONE timeframe.

        Parameters
        ----------
        t : float
            current timeframe
        paths : dict

        fs : int
            desired sampling frequency

        Returns
        -------
        h : ndarray
            multichannel impulse response for timeframe t with hrir info included
        """

        #a dict containing all paths (direct, specular, diffracted..)
        #paths = self.results["timeframes"][t]["paths"]

        direct_path = self._direct_sound(t = t)
        specular_paths = self._image_source_method(t=t, order=self.settings["ism_order"])


        rp = np.sqrt(0.9)  # refl coeff

        distvec = []  # placeholder for distances
        ampvec = []  # placeholder for amplitudes

        if direct_path is not None:
            l = direct_path.get_length()
            distvec.append(l)
            ampvec.append(1 / l)

        for order in range(len(specular_paths)):
            for path in specular_paths[order]:
                l = path.get_length()
                distvec.append(l)
                ampvec.append(1 / l) #* rp ** (len(path) - 2)

        # todo: add diffraction
        # place every pulse at the nearest integer:
        ndelay = [round(x / constants.C_AIR * fs) + 1 for x in distvec]
        # similar to accumarray in matlab
        mono_ir = np.zeros(np.max(ndelay)+1)    #np.bincount(ndelay, weights=ampvec)
        for delay, amplitude in zip(ndelay, ampvec):
            mono_ir[delay] = amplitude
        return mono_ir

    def _trim_leading_zeros(self, ir):
        """
        Removes the leading zeros from a binaural impulse response without disturbing itd.
        Parameters
        ----------
        ir : ndarray
            binaural impulse response

        Returns
        -------
        ir with no leading zeros, hrir (itd) is kept.
        """
        #round numbers close to zero to exactly zero
        ir[:, 0] = [round(x, 6) for x in ir[:, 0]]
        ir[:, 1] = [round(x, 6) for x in ir[:, 1]]

        #check which signal has the fewest leading zeros (this is the signal we have to trim after)
        leading_0s_L = np.min(np.nonzero(np.hstack((ir[:,0], 1))))
        leading_0s_R = np.min(np.nonzero(np.hstack((ir[:,1], 1))))

        if leading_0s_R < leading_0s_L:
            ir_L = ir[leading_0s_R:,0]
            ir_R = ir[leading_0s_R:, 1]
        else:
            ir_L = ir[leading_0s_L:, 0]
            ir_R = ir[leading_0s_L:, 1]

        length = np.max([len(ir_L), len(ir_R)])
        ir_trim = np.zeros(shape=(length, 2))
        ir_trim[:len(ir_L),0], ir_trim[:len(ir_R),1] = ir_L, ir_R

        return ir_trim

    def auralize(self,
                 play : bool = True,
                 save_to_file : bool = False,
                 dir : str = None,
                 filename : str = None,
                 ambience :bool = False,
                 level : float = 0.06,
                 seed : int = 1,
                 dtype : str = 'int16'):
        """

        Parameters
        ----------
        play : bool
            play sound if True
        save_to_file
        dir
        filename

        Returns
        -------
        None
        """

        if self.settings["auralizing_engine"] == "mdr":
            stereo_audio = self._render_audio_mdr(plot = False)
        elif self.settings["auralizing_engine"] == "sdr":
            stereo_audio = self._render_audio_sdr(plot = False)

        if ambience:
            stereo_audio = self.add_ambience(stereo_audio, level, seed)

        if play:
            # normalize:
            m = np.max([np.abs(stereo_audio[:, 0]), np.abs(stereo_audio[:, 1])])
            if m != 0:
                stereo_audio[:, 0] = stereo_audio[:, 0] / m
                stereo_audio[:, 1] = stereo_audio[:, 1] / m
            #plt.plot(stereo_audio)
            #plt.show()
            sd.play(stereo_audio, samplerate=self.fs)
            sd.wait()

        if save_to_file:
            m = np.max([np.abs(stereo_audio[:, 0]), np.abs(stereo_audio[:, 1])])
            stereo_audio[:, 0] = stereo_audio[:, 0] / m
            stereo_audio[:, 1] = stereo_audio[:, 1] / m
            if dir is None:
                dir = ROOT_DIR + "/data/results/audio_files/"
            #scipy.io.wavfile.write(os.path.join(dir,filename), self.fs, stereo_audio.astype(np.float32))
            sf.write(file = os.path.join(dir, filename), data=stereo_audio, samplerate=self.fs)

    def add_ambience(self, x, gain = 0.08, seed = 1):
        """
        Adds ambience (background noise) to x
        Parameters
        ----------
        x : ndarray
        level : float

        Returns
        -------
        x
        """
        import scipy.io
        path = r"/data/ambient_sounds/background_036_GPS.mat"
        mat_file = scipy.io.loadmat(ROOT_DIR + path)
        fs = 48000
        left = mat_file["shdf"][0][0][-1][0,:]
        right = mat_file["shdf"][0][0][-1][1, :]
        mono = mat_file["shdf"][0][0][-1][2, :]

        import random
        random.seed(seed)
        start_s = random.randint(1,30)*self.fs

        x[:,0] += left[start_s:start_s+len(x)]*gain
        x[:, 1] += right[start_s:start_s+len(x)]*gain
        return x

    def save_animation(self, filename):
        frame_filepath =  ROOT_DIR + "/data/animations/frames/"
        audio_filepath = ROOT_DIR + "/data/animations/audio/"
        video_filepath = ROOT_DIR + "/data/animations/video/"
        animation_filepath = ROOT_DIR + r"/data/animations/"

        if len(os.listdir(frame_filepath)) > 0:
            #dir is not emtpy, clean
            files = glob.glob(f"{frame_filepath}*.png")
            for f in files:
                os.remove(f)

        for i,t in enumerate(self.time_vec):
            p = self.scene.plot(t=t,off_screen=True)
            #set camera position
            if self.scene.camera_position is not None:
                p.camera.position = self.scene.camera_position
            if self.scene.camera_zoom is not None:
                p.camera.zoom(self.scene.camera_zoom)
            if self.scene.camera_focus is not None:
                p.camera.focal_point = (14, 10, 1.8)
            p.show(screenshot=os.path.join(frame_filepath, f'{i}frame.png'), full_screen = True)

        time.sleep(5)

        images = sorted(glob.glob(f"{frame_filepath}*.png"), key=os.path.getmtime)

        video_name = f'{filename}.mp4'
        video_name = os.path.join(video_filepath, video_name)

        from moviepy.editor import AudioFileClip, ImageSequenceClip

        clip = ImageSequenceClip(images, fps=self.source_fs)
        #now add audio
        if not self.results:
            self.start()

        self.auralize(play=False, filename ="aniaudio.wav", dir = audio_filepath, save_to_file=True)
        tot_time = int(self.scene.source.get_total_time()) #rounds down
        # load video
        clip = clip.subclip(0,tot_time)
        audio = AudioFileClip(os.path.join(audio_filepath,"aniaudio.wav"))
        audio = audio.subclip(0,tot_time)
        clip = clip.set_audio(audio)

        outputname = os.path.join(animation_filepath, filename)

        clip.write_videofile(outputname, fps=self.source_fs,
                             codec = "libx264",
                             audio = True,
                             audio_fps=self.fs, audio_codec="aac",
                             temp_audiofile='temp-audio.m4a',
                             remove_temp=True)


if __name__ == "__main__":
    from urban_auralization.logic.scene import Building, Scene, Ground
    from urban_auralization.logic.acoustics import Receiver, MovingSource, constants
    from urban_auralization.logic.meteorology import Wind


    screen1 = Building(vertices=[(0, 3), (4, 3)], height=3, abs_coeff=constants.wood_paneling)
    ground = Ground(buildings = [screen1], abs_coeff=constants.grass)
    features = [ground, screen1]

    receiver = Receiver(3, 0, 1.8,
                        elevation=90,
                        azimuth=90)
    sources_dir = ROOT_DIR + "/data/sources/*.wav"
    _SOURCES = glob.glob(sources_dir)

    # pick a source to auralize spatially:
    source = _SOURCES[3]

    source = MovingSource(waypoints=[(-80,5.5,1), (80,5.5,1)],
                          velocity=11.1,
                          signal=source)

    scene = Scene(features=features,
                  wind=Wind(heading=40, windspeed=10),
                  receiver=receiver,
                  source=source)

    model = Model(scene=scene, source_fs=10, fs=44100)
    model.settings["ism_order"] = 3
    model.settings["direct_sound"] = True
    model.settings["specular_reflections"] = True
    model.settings["diffraction"] = True
    model.settings["auralizing_engine"] = "mdr" # multiple Doppler, "sdr" = single Doppler

    #uncomment to plot scene
    # p = model.scene.plot()
    # p.show()

    model.start()
    model.auralize()
    #model.save_animation(filename="bus50kmph.mp4")

