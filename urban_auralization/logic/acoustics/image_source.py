import numpy as np
from urban_auralization.logic.geometry.path import Path


class ImageSource():
    def __init__(self, pos, mirror_plane, mirror_source, order, name = None):
        self.pos = pos
        self.mirror_plane = mirror_plane #nb! can be None type object
        self.mirror_source = mirror_source #nb! can be None type object
        self.order = order
        self.name = name

    def __str__(self):
        return f"({self.pos[0]}, {self.pos[1]}, {self.pos[2]})"

    def get_visible_path(self, receiver, n):
        path = Path(path_type='r') #list of points (list) [[],[],[]] for order 1
        path.add_point(receiver)


        #check if source is visible for receiver: (visible if intersection point is not None)
        line = np.array([[self.pos[0], self.pos[1], self.pos[2]], [receiver[0], receiver[1], receiver[2]]])
        if self.mirror_plane.bounding_box:
            intersection_point = self.mirror_plane.bounding_box.intersect(line)
        else:
            intersection_point = self.mirror_plane.intersect(line)

        if intersection_point is not None:

            if n == 1:
                """
                now we know that this image source is the last is before the receiver 
                """
                path.add_point(intersection_point)
                path.add_point(self.mirror_source.pos)
                return path

            else:
                if self.mirror_source.get_visible_path(receiver=intersection_point, n = n-1) is not None:
                    path.add_points(self.mirror_source.get_visible_path(receiver=intersection_point, n = n - 1).points)
                    return path
        else:
            return None



if __name__ == "__main__":
    pass
