import numpy as np
import pyvista as pv

from urban_auralization.logic.geometry import Ray, Polygon, Plane, Vector
from urban_auralization.logic.acoustics import constants

class Building(Polygon):
    """
    Square building with a base rectangle extruded to height
    """
    def __init__(self, vertices, height, color = "white", abs_coeff = constants.concrete, texture = None):
        super().__init__(vertices)
        self.vertices = vertices
        self.height = height
        self.body = self._extrude(self.height)
        self.color = color
        self.texture = texture
        self.abs_coeff = abs_coeff #todo: is this really the reflection coeff?? abs coeff maybe, could use reflection coefficient

    def __str__(self):
        return f"Building with vertices {self.vertices}"

    def __eq__(self, other):
        if (self.vertices == other.vertices) and (self.height == other.height):
            return True
        else:
            return False

    def plot(self):
        self.body.plot(color=self.color, specular = 1)

    def _extrude(self, height):
        N = len(self.vertices)
        points3d = np.pad(self.vertices, [(0, 0), (0, 1)])
        face = [N + 1] + list(range(N)) + [0]  # cell connectivity for a single cell
        polygon = pv.PolyData(points3d, faces=face)
        # extrude along z and plot
        body = polygon.extrude((0, 0, height), capping=True)
        return body

    def get_body(self):
        return self.body

    def get_all_planes(self) -> list[Plane]:
        """
        Returns
        -------
        All planes with normal vector pointing outwars from the building (convex).
        If the building has 2 vertices, the function will return a normalvector with an arbit direction of the two. Buildings
        with roof will
        """
        planes = []
        for i in range(len(self.vertices) - 1):
            #tree points defining the plane
            p1 = np.array([self.vertices[i][0], self.vertices[i][1], 0])
            p2 = np.array([self.vertices[i+1][0], self.vertices[i+1][1], 0])
            p3 = np.array([self.vertices[i+1][0], self.vertices[i+1][1], self.height])
            p4 = np.array([self.vertices[i][0], self.vertices[i][1], self.height])
            #vectors in the given plane
            v1 = p3 - p1
            v2 = p2 - p1
            #cross product yield normal vector
            a, b, c = np.cross(v1, v2)
            normal_vec = Vector(a, b, c).normalize()

            planes.append(Plane(point=p3, normalvector=normal_vec, bounding_box=Polygon([tuple(p1),tuple(p2),tuple(p3),tuple(p4)]), abs_coeff=self.abs_coeff))
        #roof plane
        if len(self.vertices) > 2:
            p1 = np.array([self.vertices[0][0], self.vertices[0][1], self.height])
            p2 = np.array([self.vertices[1][0], self.vertices[1][1], self.height])
            p3 = np.array([self.vertices[2][0], self.vertices[2][1], self.height])

            v1 = p3 - p1
            v2 = p2 - p1
            # cross product yield normal vector
            a, b, c = np.cross(v1, v2)
            normal_vec = Vector(a, b, c).normalize()
            bounding_box = []
            for vertice in self.vertices:
                bounding_box.append((vertice[0], vertice[1], self.height))

            planes.append(Plane(point=p3, normalvector=normal_vec, bounding_box=Polygon(bounding_box), abs_coeff=self.abs_coeff))

        return planes


if __name__ == "__main__":
    b1 = Building(vertices=[(0,0), (0,4),(4,4),(4,0),(0,0)], height=40)
    for plane in b1.get_all_planes():
        print(plane)
