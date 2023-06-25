import numpy as np

class Wind():
    def __init__(self, heading, windspeed):
        """

        Parameters
        ----------
        heading : int
            heading in degrees between 0 and 359
        windspeed : float
            windspeed in m/s
        """
        self.heading = heading
        self.windspeed = windspeed

    def get_cent_dir_grids(self, xmin, ymin, xmax, ymax, spacing):
        """

        Parameters
        ----------
        xmin
        ymin
        xmax
        ymax
        spacing

        Returns
        -------
        points
            center point of every arrow
        dir
            direction vector of every arrow
        """
        #todo: make it return points in every height as well (linear temperature gradient) That means we should return mag as well
        x = np.arange(start = xmin, step = spacing, stop = xmax+spacing, dtype = float)
        y = np.arange(start = ymin, step = spacing, stop = ymax+spacing, dtype = float)

        g = np.meshgrid(x,y)
        points = np.append(g[0].reshape(-1,1),g[1].reshape(-1,1),axis=1)
        points = np.pad(points, [(0, 0), (0, 1)])


        dirx = np.cos(self._deg_to_rad(self.heading))
        diry = np.sin(self._deg_to_rad(self.heading))
        dirz = 0

        dir = np.array([[dirx, diry, dirz] for i in range(len(points))])

        return points, dir

    def _deg_to_rad(self, deg):
        return deg*np.pi/180

    def get_v_m(self):
        """
        Returns the wind velocity vector, v_m.
        Returns
        -------
        """
        x = self.windspeed*np.cos(self._deg_to_rad(self.heading))
        y = self.windspeed*np.sin(self._deg_to_rad(self.heading))
        z = 0
        return np.array([x,y,z])

if __name__ == "__main__":
    #print(np.random.random((10, 3)))
    wind = Wind(heading = 230, windspeed=5)
    print(wind.get_cent_dir_grids(-5,-5,35,35,5))