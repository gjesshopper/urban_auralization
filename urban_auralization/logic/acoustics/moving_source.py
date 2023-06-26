import logging

import numpy as np
import matplotlib.pyplot as plt

from urban_auralization.logic.geometry import Vector

class MovingSource():
    def __init__(self, waypoints, velocity: float, signal: object) -> object:
        """

        Parameters
        ----------
        waypoints : list of points that receiver goes through
        velocity : float
            velocity in m/s
        signal : str
            filepath to .wav file (preferrably anechoic audio_files signal)
        """
        self.waypoints = waypoints
        self.init_pos = self.waypoints[0]
        self.velocity = velocity
        self.signal = signal

    def get_position(self, t : float):
        if t == 0:
            return np.array([self.waypoints[0][0], self.waypoints[0][1], self.waypoints[0][2]])
        current_distance = self.velocity * t
        for i in range(len(self.waypoints)-1):
            r0 = Vector(self.waypoints[i][0], self.waypoints[i][1], self.waypoints[i][2])
            r1 = Vector(self.waypoints[i+1][0], self.waypoints[i+1][1], self.waypoints[i+1][2])
            delta_r = r1 - r0

            if current_distance < delta_r.length():
                #distance traveled in the segment
                #calc direction with unit vector
                direction = delta_r.normalize()
                #displacement between the two points:
                displacement = current_distance
                new_position = r0 + displacement * direction
                return [new_position.x, new_position.y, new_position.z]
            elif current_distance == delta_r.length():
                return [self.waypoints[i+1][0], self.waypoints[i+1][1], self.waypoints[i+1][2]]
            elif current_distance > delta_r.length():
                current_distance -= delta_r.length()

    def get_total_distance(self):
        tot_distance = 0
        for i in range(len(self.waypoints)-1):
            r0 = Vector(self.waypoints[i][0], self.waypoints[i][1], self.waypoints[i][2])
            r1 = Vector(self.waypoints[i + 1][0], self.waypoints[i + 1][1], self.waypoints[i + 1][2])
            delta_r = r1 - r0
            tot_distance += delta_r.length()
        return tot_distance

    def get_total_time(self):
        return self.get_total_distance()/self.velocity

    def get_vs(self, t):
        """
        Returns
        -------
        vs : ndarray
            velocity vector
        """
        if t >= self.get_total_time():
            logging.error(msg="Could not get vs vector, source is beyond trajectory..")
            return

        seg_dist = 0
        distance_travelled = self.velocity*t
        for i in range(len(self.waypoints) - 1):
            r0 = Vector(self.waypoints[i][0], self.waypoints[i][1], self.waypoints[i][2])
            r1 = Vector(self.waypoints[i + 1][0], self.waypoints[i + 1][1], self.waypoints[i + 1][2])
            delta_r = r1 - r0
            seg_dist += delta_r.length()
            if distance_travelled < seg_dist:
                direction = delta_r.normalize()
                r = np.sqrt(direction.x**2 + direction.y**2)
                theta = np.arcsin(r)
                phi = np.arcsin(direction.y/r)
                x = self.velocity*np.sin(theta)*np.cos(phi)
                y = self.velocity*np.sin(theta)*np.sin(phi)
                z = self.velocity*np.cos(theta)
                return np.array([x, y, z])



if __name__ == "__main__":
    source = MovingSource(waypoints=[(0,0,0), (10,10,0)],
                          velocity=2.0,
                          signal=1)
    print("total distance: ", source.get_total_distance())
    #print(source.get_position(t=20.32))
    print(source.get_vs(11))
