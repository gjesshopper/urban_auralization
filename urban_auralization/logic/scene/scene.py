import numpy as np
import pyvista as pv
import imageio
import os
import glob
from pprint import pprint
import logging
logging.basicConfig(level = logging.INFO)

from urban_auralization.logic.scene import Building
from urban_auralization.logic.geometry import Ray, Plane, Vector
from urban_auralization.logic.meteorology import Wind
from urban_auralization.logic.acoustics import MovingSource, Receiver, Model

class Scene():
    """class cointaining all data for a geographical scene"""
    def __init__(self, features : list[Building],
                 margin = 5.0,
                 ground_impedance = 1.0,
                 wind : Wind = None,
                 show_wind : bool = True,
                 receiver : Receiver = None,
                 source : MovingSource = None):
        """

        Parameters
        ----------
        features : Building
            all objects in the scene including ground, buildings, screens etc
        margin : float, default = 5.0
            distance to edge of scene from buildings max/min edges
        """
        self.features = features
        self.margin = margin
        self.ground_impedance = ground_impedance
        self.wind = wind
        self.show_wind = show_wind
        self.receiver = receiver
        self.source = source
        self.model = None

        #camera
        self.camera_position = None
        self.camera_zoom = None
        self.camera_focus = None


    def add_model(self, model):
        self.model = model

    def plot(self, t : float = 0.0, rays : list[Ray] = None, off_screen : bool = False):
        """
        Plots the scene.
        Parameters
        ----------
        t : what time to plot
        rays : list of Rays at time t

        Returns
        -------

        """
        plotter = pv.Plotter(off_screen=off_screen)


        #add buildings
        if self.features:
            allx = []
            ally = []

            for feature in self.features:
                if feature.texture is not None:
                    plotter.add_mesh(feature.get_body(), texture= feature.texture, opacity = 1, show_edges=False)
                else:
                    plotter.add_mesh(feature.get_body(), color=feature.color, opacity=1,
                                     show_edges=False)

                #find xmax, xmin, ymax, ymin:
                x, y = zip(*feature.vertices)
                allx += x
                ally += y
            xmin, xmax = min(allx), max(allx)
            ymin, ymax = min(ally), max(ally)

        # add ground:
        #ground = Building(vertices=[(xmin - self.margin, ymin - self.margin),
                                    #(xmax + self.margin, ymin - self.margin),
                                    #(xmax + self.margin, ymax + self.margin),
                                    #(xmin - self.margin, ymax + self.margin),
                                    #(xmin - self.margin, ymin - self.margin)], height=0).get_body()
        #plotter.add_mesh(ground, color="green", show_edges=True)


        plotter.add_axes()
        if self.wind is not None and self.show_wind == True:
            cent, direction = self.wind.get_cent_dir_grids(xmin,ymin,xmax,ymax,spacing=abs(-xmin+xmax)/7)
            plotter.add_arrows(cent=cent, direction = direction, mag = 1)

        #set backgroundcolor
        plotter.set_background('blue', top='lightblue')

        #add receiver
        if self.receiver:
            plotter.add_points(np.array(self.receiver.get_postion()),
                               render_points_as_spheres=True,
                               color = 'red',
                               point_size = 15)

            #add arrow where source is pointed
            plotter.add_arrows(cent=np.array(self.receiver.get_postion()),
                               direction=self.receiver.get_direction(), mag=2, color = 'blue')

        #add source path
        if self.source:
            plotter.add_lines(np.asarray(self.source.waypoints), color='grey', width=3, label=None, name=None)
            # add source at given time
            plotter.add_points(np.array(self.source.get_position(t = t)),
                               render_points_as_spheres=True,
                               color='black',
                               point_size=15)

        # add direct sound to the plot if settings say so
        if self.model:
            if self.model.settings["direct_sound"]:
                if self.model.results:
                    if t in self.model.time_vec:
                        direct_line = self.model.results["timeframes"][t]["paths"]["direct"]
                    else:
                        direct_line = self.model._direct_sound(t)
                else:

                    direct_line = self.model._direct_sound(t)
                if direct_line is not None:
                    plotter.add_lines(np.array(direct_line.points), color="black", width=2, label=None, name=None)


            #add specular reflection to the plot if settings say so
            if self.model.settings["specular_reflections"]:
                if self.model.results:
                    if t in self.model.time_vec:
                        #now we already have the result so we dont need to calculate again (save time)
                        ism_lines = self.model.results["timeframes"][t]["paths"]["specular"]
                    else:
                        ism_lines = self.model._image_source_method(t, order=self.model.settings["ism_order"])
                else:
                    ism_lines = self.model._image_source_method(t, order=self.model.settings["ism_order"])

                for n in range(len(ism_lines)):
                    for path in ism_lines[n]:
                        line = np.array(path.points)
                        plotter.add_lines(line, color="orange", width=2, label=None, name=None)


            if self.model.settings["diffraction"]:
                if self.model.results:
                    if t in self.model.time_vec:
                        # now we already have the result so we dont need to calculate again (save time)
                        diffraction_paths = self.model.results["timeframes"][t]["paths"]["diffraction"]
                    else:
                        diffraction_paths = self.model._diffraction(t)
                else:
                    diffraction_paths = self.model._diffraction(t)


                if diffraction_paths is not None:
                    for path in diffraction_paths:
                        plotter.add_lines(np.array(path.points), color="blue", width=2, label=None, name=None)

        #set camera properties
        #plotter.camera.position = self.camera_position
        #plotter.camera.zoom(self.camera_zoom)
        plotter.hide_axes()
        return plotter


    def get_all_planes(self) -> list[Plane]:
        planes = []
        #append ground plane
        planes.append(Plane(point=(0,0,0), normalvector=Vector(0,0,1)))
        for building in self.features:
            for plane in building.get_all_planes():
                planes.append(plane)
        return planes

    def save_to_file(self, filename):
        #save model to .json file
        pass

    def load_file(self, filename):
        #load scene into self from .json file
        pass

    def set_camera_position(self, position : tuple):
        self.camera_position = position

    def set_camera_zoom(self, zoom : float):
        self.camera_zoom = zoom


if __name__ == "__main__":
    pass






