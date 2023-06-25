import coloredlogs, logging, sys
import itertools
logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.CRITICAL, logger=logger, isatty=True,
                    fmt="%(asctime)s %(levelname)-8s %(message)s",
                    stream=sys.stdout,
                    datefmt='%Y-%m-%d %H:%M:%S')




import numpy as np
from urban_auralization.logic.geometry.path import Path
from scipy.spatial import ConvexHull


class Harmonoise:
    def __init__(self, scene = None, model = None):
        self.scene = scene
        self.model = model

    def get_diffracted_paths(self, t : float = 0):
        """
        Gets all side and top diffracted paths
        Parameters
        ----------
        t : float
            time

        Returns
        -------
        paths : list[Path]
            list of diffracted Path objects
        """
        # 1. check if direct path is obstructed:
        r = self.scene.receiver.get_postion()
        s = self.scene.source.get_position(t=t)
        phantom_dct = Path(points=np.array([s, r]))
        paths = [] #list of path objects
        if self.model._obstructed(phantom_dct):
            logging.debug(msg="direct path is obstructed, diffraction on obstructed obj.")
            # we have diffraction... Find all points of intersection
            intersection_points = self.find_intersection_points(path=phantom_dct)
            paths.append(self.find_top_path(source=s, receiver=r, intersection_points=intersection_points))

        paths += self.get_all_side_diffraction_paths(source=s, receiver=r)
        paths = [x for x in paths if len(x) > 2]
        return paths


    def find_shortest_path(self, paths : list[Path]):
        """
        Parameters
        ----------
        paths : list of Path objects

        Returns
        -------
        shortest : Path
        """
        shortest = None
        for i, path in enumerate(paths):
            if i == 0:
                shortest = path
            if path.get_length() > shortest.get_length():
                shortest = path
        if shortest is None:
            logging.warning("Shortest diffracted path is None type object")
        return shortest


    def find_top_path(self, source : list, receiver : list, intersection_points : list):
        """
        Finds the top diffracted path when the direct path is obstructed.
        Parameters
        ----------
        source : list, [x,y,z]
        receiver : list, [x,y,z]
        intersection_points : list [[point, feature, plane], [point, feature, plane],....]

        Returns
        ------
        path

        """
        logging.info(msg = "find_top_path-function is called")

        #calc convex_hull
        points = np.empty(shape = (0,3), dtype=float)
        points = np.append(points, np.array([source]), axis = 0)

        for set in intersection_points:
            points = np.append(points, np.array([[set[0][0], set[0][1], set[1].height]]), axis = 0)
        points = np.append(points, np.array([receiver]), axis = 0)

        x, y, z = points[:,0], points[:,1], points[:,2]

        #check if all elements are equal in x
        equal_x = np.all(x == x[0]) #bool, True if all equal

        if not equal_x:
            points_2d = np.array([[x, z] for x, z in zip(x,z)])
        else:
            points_2d = np.array([[y, z] for y, z in zip(y,z)])

        hull = ConvexHull(points_2d)
        indices = np.sort(hull.vertices) #indices of points in intersection points
        points_3d = points[indices]

        #find also the corresponding planes where the paths edges
        edge_planes = []
        for index in indices[1:-1]-1:
            edge_planes.append(intersection_points[index][2])
        #reflection_planes = [i[2] for i in intersection_points(indices[1:-1]-1)]
        return Path(points = points_3d, reflection_planes=edge_planes, path_type='td')




    def find_intersection_points(self, path : Path):
        """
        Finds all intersection points along the path from all obstructing object.
        Path must be a single line segment, i.e. two points.
        Parameters
        ----------
        path : Path

        Returns
        -------
        intersection_points : list
            [[intersection_points, feature, plane], [[0,0,1], feature_object, plane_object]]
        """
        if len(path.points) != 2:
            logging.critical(msg="Path has len != 2.")
            return

        intersection_points  = []

        for feature in self.scene.features:
            if feature.height != 0:
                for plane in feature.get_all_planes():
                    intersection_point = plane.bounding_box.intersect(path.points)
                    if intersection_point is not None:
                        intersection_points.append([intersection_point, feature, plane])

        #sort intersection points from the one closest to the source
        n = len(intersection_points)
        swap = False
        for i in range(n - 1):
            for j in range(0, n - i - 1):
                dist1 = np.sqrt((intersection_points[j][0][0] - path.points[0][0])**2 +
                                (intersection_points[j][0][1] - path.points[0][1]) ** 2 +
                                (intersection_points[j][0][2] - path.points[0][2]) ** 2)
                dist2 = np.sqrt((intersection_points[j+1][0][0] - path.points[0][0]) ** 2 +
                                (intersection_points[j+1][0][1] - path.points[0][1]) ** 2 +
                                (intersection_points[j+1][0][2] - path.points[0][2]) ** 2)
                if dist1 > dist2:
                    swap = True
                    intersection_points[j], intersection_points[j + 1] = intersection_points[j + 1], intersection_points[j]
            if not swap:
                break

        logging.debug(f"Number of intersection points from source to receiver: {len(intersection_points)}")

        return intersection_points

    def get_all_feature_combinations(self):
        "returns all feature combinations except ground "
        feature_combs = []
        for L in range(len(self.scene.features) + 1):
            for subset in itertools.combinations(self.scene.features, L):
                if subset:
                    feature_combs.append([feature for feature in subset if feature.height != 0])
        return feature_combs

    def get_all_side_diffraction_paths(self, source : list, receiver : list):
        paths_3d = []
        feature_combs = self.get_all_feature_combinations()
        for feature_comb in feature_combs:
            if not feature_comb:
                continue
            subset_cluster_points = [[source[0], source[1]]]
            for feature in feature_comb:
                subset_cluster_points += [list(a) for a in feature.vertices]
            subset_cluster_points.append([receiver[0], receiver[1]])
            subset_cluster_points = np.array(subset_cluster_points)

            hull = ConvexHull(subset_cluster_points)
            points_in_hull = hull.points[hull.vertices]
            #check if receiver/source not in hull
            if not [receiver[0], receiver[1]] in points_in_hull:
                continue
            if not [source[0], source[1]] in points_in_hull:
                continue

            paths_from_hull = self.get_paths_from_hull(hull=hull,
                                     source_2d=[source[0], source[1]],
                                     receiver_2d=[receiver[0], receiver[1]])

            for path in paths_from_hull:
                if len(path) > 1:
                    paths_3d.append(self.get_3d_path(path_2d=path, source_3d=source, receiver_3d=receiver, feature_comb = feature_comb))

        paths_3d = self.filter_out_obstructed_paths(paths_3d)
        return paths_3d


    def get_paths_from_hull(self, hull, source_2d, receiver_2d):
        """
        This function separates the two paths between source and receiver from the 2d hull.
        If any of the paths is equal to the direct path, it will be neglected. The paths are returned as a list with points (np.arrays),
        because the have no height yet (2D). Note: No obstruction is checked here....

        Parameters
        ----------
        hull : the output from ConvexHull
        source_2d : list, the x and y coordinates of the source
        receiver_2d : list, the x and y coordinates of the receiver

        Returns
        -------
        paths_2d : list[np.array], 2d paths from the convexhull
        """
        points_in_hull = hull.points[hull.vertices]

        source_idx = 0
        receiver_idx = 0
        #find index of source and receiver:
        for i, point in enumerate(points_in_hull):
            if np.array_equal(point,source_2d):
                source_idx = i
            if np.array_equal(point, receiver_2d):
                receiver_idx = i

        #get 2d paths (from source to receiver)
        if source_idx < receiver_idx:
            path1 = points_in_hull[source_idx:receiver_idx+1]
            path2 = np.flip(np.concatenate((points_in_hull[receiver_idx:], points_in_hull[:source_idx+1]), axis = 0), axis = 0)
        else:
            path1 = np.flip(points_in_hull[receiver_idx:source_idx+1], axis = 0)
            path2 = np.concatenate((points_in_hull[source_idx:], points_in_hull[:receiver_idx+1]), axis=0)

        #check that path1/2 is not same as direct path
        direct = np.array([source_2d, receiver_2d])
        paths_2d = []
        if not np.array_equal(path1,direct):
            paths_2d.append(path1)
        if not np.array_equal(path2, direct):
            paths_2d.append(path2)
        return paths_2d

    def filter_out_equal_paths(self, paths : list[Path]):

        for i, pathi in enumerate(paths):
            for j, pathj in enumerate(paths[i+1:]):
                if pathi == pathj:
                    paths.remove(pathj)
        return paths


    def filter_out_obstructed_paths(self, paths : list[Path]):
        pts = []
        for path in paths:
            if not self.model._obstructed(path):
                pts.append(path)
        return pts

    def get_3d_path(self, path_2d, source_3d, receiver_3d, feature_comb):
        """
        Returns the 3D Path object from the 2d projected path
        Parameters
        ----------
        path_2d
        source_3d
        receiver_3d
        feature_comb : the respective feature combination that make up the 2d path

        Returns
        -------
        Path
        """

        path_3d = np.zeros((len(path_2d), 3))
        path_3d[:, :-1] = path_2d


        path_3d_zero_z = path_3d.copy() #copy of expanded 2d path with z = 0

        #add source height, receiver height
        source_heigth = source_3d[-1]
        receiver_height = receiver_3d[-1]
        path_3d[0,2] = source_heigth #add source height to the first point
        path_3d[-1, 2] = receiver_height #add receiver height to the last point


        #parameterize line from source to receiver to find heigths
        l = Path(points=path_3d_zero_z).get_length() #ground distance travelled from source to receiver
        a = (receiver_height - source_heigth)/l
        b = source_heigth
        def get_height(x, a, b):
            return a*x+b

        for i in range(len(path_2d) - 2):
            x = Path(points=np.array([path_3d_zero_z[0,:], path_3d_zero_z[i+1,:]])).get_length()
            z = get_height(x, a, b)
            path_3d[i+1, 2] = z
        #todo: potentially add some way to sort the paths here...
        path = Path(points = path_3d, path_type='sd')
        path.features = feature_comb
        return path




if __name__ == "__main__":
    pass