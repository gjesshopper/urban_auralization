
from urban_auralization.logic.geometry import Plane, Vector
from urban_auralization.logic.acoustics import constants
from urban_auralization.logic.scene import Building


class Ground(Building):
    """
    Default ground is grass
    """

    def __init__(self, buildings : list[Building] = None,
                 abs_coeff : float = constants.neutral,
                 color :str = "green",
                 margin : float = 5.0,
                 vertices :list[tuple] = None,
                 texture = None):
        self.abs_coeff = abs_coeff
        self.plane = Plane(point=(0,0,0), normalvector=Vector(0,0,1), bounding_box = None, abs_coeff=self.abs_coeff)
        self.color = color
        self.buildings = buildings
        self.texture = texture
        self.margin = margin
        if vertices is not None:
            super().__init__(vertices=vertices, height=0, color=color, abs_coeff=abs_coeff)
        else:
            super().__init__(vertices=self.get_plot_parameter(), height=0, color=color, abs_coeff=abs_coeff)

    def get_all_planes(self) -> list[Plane]:
        return [self.plane]

    def get_plot_parameter(self):
        if self.buildings == None:
            return [(-self.margin, -self.margin),
                    (self.margin,-self.margin),
                    (self.margin,self.margin),
                    (-self.margin,self.margin),
                    (-self.margin,-self.margin)]
        allx = []
        ally = []
        for building in self.buildings:
            #find xmax, xmin, ymax, ymin:
            x, y = zip(*building.vertices)
            allx += x
            ally += y
        xmin, xmax = min(allx), max(allx)
        ymin, ymax = min(ally), max(ally)

        vertices=[(xmin - self.margin, ymin - self.margin),
                                    (xmax + self.margin, ymin - self.margin),
                                    (xmax + self.margin, ymax + self.margin),
                                    (xmin - self.margin, ymax + self.margin),
                                    (xmin - self.margin, ymin - self.margin)]
        return vertices

if __name__ == "__main__":

    verts = [(0,0), (10,0), (10,2), (0,2), (0,0)]
    ground = Ground(vertices=verts, color = "blue")
    ground.plot()
    print(ground.abs_coeff)



