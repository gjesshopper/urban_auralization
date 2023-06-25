import numpy as np
from urban_auralization.logic.acoustics import constants

class Plane():
    """
    Formula:
    ax + by + cz + d = 0,
    where n = [a,b,c] and d =
    """
    def __init__(self, point, normalvector, bounding_box= None, abs_coeff = constants.concrete):
        """

        Parameters
        ----------
        point : tuple
         (x,y,z)
        normalvector : Vector
            normalvector of the plane
        bounding_box : list[tuple], optional, default = None
            A bound surface in the plane. The method ...
            can be called to determine if a vector passes through the bounding box
        """

        self.point = point
        self.normalvector = normalvector #(a,b,c)
        self.bounding_box = bounding_box
        # a(x−x0)+b(y−y0)+c(z−z0)=0.
        self.d = self.normalvector.x*(-self.point[0])+self.normalvector.y*(-self.point[1])+self.normalvector.z*(-self.point[2])
        self.abs_coeff = abs_coeff

    def __eq__(self, other):
        if self.normalvector != other.normalvector \
            and self.bounding_box != other.bounding_box:
            return False
        return True


    def __str__(self):
        a, b, c, d = self.normalvector.x, self.normalvector.y, self.normalvector.z, self.d
        if d == 0:
            return f"Plane with equation {a}x+{b}y+{c}z=0"
        else:
            return f"Plane with equation {a}x+{b}y+{c}z+{d}=0"


    def mirror(self, point):
        """
        Mirror a point about a plane
        Parameters
        ----------
        point

        Returns
        -------

        """
        a, b, c, d = self.normalvector.x, self.normalvector.y, self.normalvector.z, self.d
        x, y, z = point[0], point[1], point[2]
        if not self.normalvector.length() == 1:
            self.normalvector = self.normalvector.normalize()
        k = (-a * x - b * y - c * z - d) / (a * a + b * b + c * c)
        x2 = a * k + x
        y2 = b * k + y
        z2 = c * k + z
        x3 = 2 * x2 - x
        y3 = 2 * y2 - y
        z3 = 2 * z2 - z
        return np.array([x3,y3,z3])

    def intersect(self, line):
        """
        This function takes in two points given in the line argument.
        If the line intersect the closed surface of the polygon (self), it returns
        the point of intersection. If not it return None.

        Parameters
        ----------
        line

        Returns
        -------
        """

        # Points in line: pn  Points in Polygon: qn
        p1, p2 = line[0], line[1]
        q1 = np.array(self.point)

        # vector normal to Plane
        n = np.array([self.normalvector.x, self.normalvector.y, self.normalvector.z])
        u = p2 - p1  # Segment's direction vector
        w = p1 - q1  # vector from plane ref point to segment ref point

        d = n[0] * (-q1[0]) + n[1] * (-q1[1]) + n[2] * (-q1[2])

        def dist_to_plane(points, normal_vec, d):
            """
            Takes in a list of points and a plane and returns the max distance from the plane and the corresponding point
            Parameters
            ----------
            points : list of points
            normal_vec : normal vector of the plane
            d : d variable of the plane

            Returns
            -------
            max_dist, point
            """
            a, b, c, d = normal_vec[0], normal_vec[1], normal_vec[2], d
            max_dist = 0
            max_idx = -1
            for i, point in enumerate(points):
                x, y, z = point[0], point[1], point[2]
                di = np.abs((a * x + b * y + c * z + d)) / (np.sqrt(a ** 2 + b ** 2 + c ** 2))

                if di > max_dist:
                    max_dist = di
                    max_idx = i
            return max_dist, points[max_idx]

        ## Tests parallelism
        if np.dot(n, u) == 0:
            return None

        ## if intersection is a point
        else:
            ## Si is the scalar where P(Si) = P0 + Si*u lies in Plane
            Si = np.dot(-n, w) / np.dot(n, u)
            PSi = p1 + Si * u

            # check if line is long enough to intersect face
            line_lenght = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2)
            # calculate distance to plane from each point in the line segment
            maxdist2plane, max_point = dist_to_plane(points=[p1, p2], normal_vec=n, d=d)

            # distance between max_point and PSi
            l = np.sqrt((PSi[0] - max_point[0]) ** 2 + (PSi[1] - max_point[1]) ** 2 + (PSi[2] - max_point[2]) ** 2)
            if line_lenght > l:
                return PSi




    def dist_to_plane(self, point):
        x, y, z = point[0],point[1], point[2]
        a, b, c, d = self.normalvector.x, self.normalvector.y, self.normalvector.z, self.d
        dist = np.abs((a*x+b*y+c*z+d)/(np.sqrt(a**2+b**2+c**2)))
        return dist






if __name__ == "__main__":
    import urban_auralization.logic.geometry
    norm = urban_auralization.logic.geometry.Vector(0, 0, 1)
    plane = Plane(point=(0,0,0), normalvector=norm)
    point = np.array([0,0,10])
    print(plane.intersect(line=np.array([[5,0,5], [-2,0,-5]])))