import matplotlib.pyplot as plt
import matplotlib.path as mpth
import pyvista as pv
import numpy as np
from shapely import geometry

from urban_auralization.logic.geometry import Vector

class Polygon():
    """
    A polygon with all vertices in the same plane.
    """
    def __init__(self, vertices : list[tuple]):
        """

        Parameters
        ----------
        vertices : list of tuples
        """
        self.vertices = vertices
        #check for duplicates in list:
        if len(self.vertices) != len(set(self.vertices)):
            self.closed = True
        else:
            self.closed = False

    def __str__(self):
        return f"{self.vertices}"

    def plot(self):
        if np.shape(self.vertices)[1] == 2:
            x, y = zip(*self.vertices)
            fig = plt.plot(x,y, 'black')
            plt.grid(True)
            return fig


    def intersect(self, line : np.ndarray):
        """
        This function takes in two points given in the line argument.
        If the line intersect the closed surface of the polygon (self), it returns
        the point of intersection. If not it return None.
        Note: if a line ends at the plane, it returns None

        Parameters
        ----------
        line

        Returns
        -------

        """
        # Points in line: pn  Points in Polygon: qn
        p1, p2 = line[0], line[1]


        q1 = np.array([self.vertices[0][0], self.vertices[0][1], self.vertices[0][2]])
        q2 = np.array([self.vertices[1][0], self.vertices[1][1], self.vertices[1][2]])
        q3 = np.array([self.vertices[2][0], self.vertices[2][1], self.vertices[2][2]])

        #vectors in the plane of polygon
        v1 = q3 - q1
        v2 = q2 - q1


        # vector normal to Plane
        n = np.cross(v1, v2) / np.linalg.norm(np.cross(v1, v2))
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

            #now check if point is inside polygon by first tranforming to 2d space

            proj_axis = max(range(3), key=lambda i: abs(n[i]))
            def project(x):
                return tuple(c for i, c in enumerate(x) if i != proj_axis)

            vertices_2d = [project(x) for x in self.vertices]
            vertices_2d.append(vertices_2d[0])
            PSi_2d = np.array(project(PSi))


            #check if point is on the edge:
            line_geom = geometry.LineString(vertices_2d)
            point_geom = geometry.Point(PSi_2d[0], PSi_2d[1])
            if line_geom.contains(point_geom):
                return None #not intersecting if on edge

            path = mpth.Path(vertices_2d)
            if path.contains_points(np.array([PSi_2d])):
                #check if line is long enough to intersect face
                line_lenght = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2)
                # calculate distance to plane from each point in the line segment
                maxdist2plane, max_point = dist_to_plane(points=[p1, p2], normal_vec=n, d=d)

                #distance between max_point and PSi
                l = np.sqrt((PSi[0]-max_point[0])**2+(PSi[1]-max_point[1])**2+(PSi[2]-max_point[2])**2)
                if line_lenght > l:
                    return PSi
                else:
                    return None








if __name__ == "__main__":
    p1 = Polygon(vertices= [(0,0,0), (0,10,0), (0,10,10), (0,0,10)])
    line = np.array([[-10,0,5], [10,0,5]])
    print(p1.intersect(line))



