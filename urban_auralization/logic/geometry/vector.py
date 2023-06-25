import numpy as np
import matplotlib.pyplot as plt

class Vector():
    def __init__(self, x : float, y :float, z : float):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return f"({self.x},{self.y},{self.z})"

    def __eq__(self, other):
        if self.x == other.x \
            and self.y == other.y \
            and self.z == other.z:
            return True
        return False


    def __add__(self, other):
        return Vector(self.x+other.x, self.y+other.y, self.z+other.z)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        return Vector(other*self.x, other*self.y, other*self.z)

    def __rmul__(self, other):
        return self*other

    def length(self):
        return np.sqrt(self.x**2+self.y**2+self.z**2)

    def normalize(self):
        """
        has length = 1
        Returns
        -------
        """
        return Vector(self.x/self.length(), self.y/self.length(), self.z/self.length())
    def dot(self, other):
        return self.x*other.x + self.y*other.y+self.z*other.z

    def plot(self):
        #from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(0, 0, 0, self.x, self.y, self.z, color='black')
        ax.set_xlim([-self.x, self.x])
        ax.set_ylim([-self.y, self.y])
        ax.set_zlim([-self.z, self.z])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        return fig


if __name__ == "__main__":
    v1 = Vector(1,1,0)
    v1.plot()
    plt.show()


