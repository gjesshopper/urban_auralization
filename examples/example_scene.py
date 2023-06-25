from urban_auralization.logic.scene import Ground, Scene, Building, Road
from urban_auralization.logic.acoustics import MovingSource, Receiver, Model
import glob
from urban_auralization.logic.acoustics import constants
from urban_auralization.definitions import ROOT_DIR

"""----------------BUILD SCENE-----------------------"""
ac= constants.neutral
b1 = Building(vertices=[(0, 0), (0, 10), (5, 10), (5, 5), (10, 5), (10, 0), (0, 0)], height=7, abs_coeff=ac)
# b2 = Building(vertices=[(20, 5), (20,10), (25,10), (25,5), (20,5)], height=9)
b2 = Building(vertices=[(20, 5), (22, 10), (25, 10), (25, 5), (20, 5)], height=9, abs_coeff=ac)
b3 = Building(vertices=[(0, -5), (30, -5), (30, -10), (0, -10), (0, -5)], height=7, abs_coeff=ac)
# screen = Building(vertices=[(7, 15), (13, 15), (13, 14.9), (7, 14.9), (7, 15)],
#                   height=4, abs_coeff=ac, color = "brown")
screen1 = Building(vertices=[(5, 15), (12, 15), (12, 14.9), (5, 14.9), (5, 15)],
                   height=4, abs_coeff=ac, color = "brown")
screen2 = Building(vertices=[(26.5, 15), (34, 15), (34, 14.9), (26.5, 14.9), (26.5, 15)],
                   height=4, abs_coeff=ac, color = "brown")
ground1 = Ground(vertices=[(-5,-15),(35,-15),(35,17),(-5,17),(-5,-15)], abs_coeff=ac, color = "green")
road = Road(vertices=[(-5,17),(35,17),(35,23),(-5,23),(-5,17)], abs_coeff=ac)
ground3 = Ground(vertices=[(-5,23),(35,23),(35,25),(-5,25),(-5,23)], abs_coeff=ac, color = "green")

features = [b1, b2, b3, ground1, road, ground3]

receiver = Receiver(15, 12.5, 1.8,
                    elevation=90,
                    azimuth=90)

sources_dir = ROOT_DIR + "/data/sources/*.wav"
_SOURCES = glob.glob(sources_dir)

# pick a source to auralize spatially:
source = _SOURCES[4]

source = MovingSource(waypoints=[(-5, 18.5, 1.2), (35, 18.5, 1.2)],
                      velocity=10,
                      signal=source)

scene = Scene(features=features,
              wind=None,
              receiver=receiver,
              source=source)

model = Model(scene=scene, source_fs=30, fs=44100)
scene.add_model(model)


"""------------------SETTINGS----------------------"""
model.settings["ism_order"] = 3
model.settings["specular_reflections"] = True
model.settings["direct_sound"] = True
model.settings["diffraction"] = True
model.settings["auralizing_engine"] = "mdr"
model.settings["only_shortest_diff"] = False

"""------------------PLOT SCENE--------------------"""
p = scene.plot(t=2.94)
p.camera.position = (20,65,32)
p.camera.focal_point = (14, 10, 1.8)
p.show()


"""-------------------AURALIZE---------------------"""
# model.start()
# model.auralize()

"""-----------------CREATE ANIMATION---------------"""
# model.start
# filename = "user_defined.mp4"
# model.save_animation(filename=filename)