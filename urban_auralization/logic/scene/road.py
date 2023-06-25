from urban_auralization.logic.geometry import Plane, Vector, Polygon
import numpy as np
import pyvista as pv
from urban_auralization.definitions import ROOT_DIR

class Road():
    def __init__(self, vertices : list[tuple],
                 road_lines : bool = True,
                 rl_width : float = 0.3,
                 rl_length : float = 4,
                 rl_spacing : float = 1.5,
                 abs_coeff : list[float] = 0,
                 color : str = "gray",
                 texture  : str = ""):
        self.road_lines = road_lines
        self.rl_width = rl_width
        self.rl_length = rl_length
        self.rl_spacing = rl_spacing
        self.vertices = vertices
        self.height = 0
        self.abs_coeff =abs_coeff
        bounding_box = [i + (0,) for i in self.vertices]
        self.plane = Plane(point=(0, 0, 0), normalvector=Vector(0, 0, 1), bounding_box=Polygon(bounding_box), abs_coeff=self.abs_coeff)
        self.sizex = 1
        self.sizey = 100
        if texture:
            self.texture = pv.read_texture(texture)
        else:
            self.texture = pv.read_texture(ROOT_DIR + "/data/textures/road.jpg")

    def get_all_planes(self) -> list[Plane]:
        return [self.plane]


    def plot(self):
        self.get_body().plot(texture = self.texture)


    def get_body(self):
        x, y = zip(*self.vertices)
        ctr = (np.max(x) - (np.max(x)-np.min(x))/2, np.max(y) - ((np.max(y)-np.min(y))/2), 0)
        mesh = pv.Plane(center = ctr, direction=(0,0,1), i_size=np.max(x)-np.min(x), j_size=np.max(y)-np.min(y), i_resolution=self.sizex, j_resolution=self.sizey)
        #mesh.cell_data["colors"] = self.get_road_pixmap()
        return mesh


if __name__ == "__main__":
    verts = [(0,0), (10,0), (10,2), (0,2), (0,0)]
    road = Road(vertices=verts)
    road.plot()