#import the modules we need
from urban_auralization.logic.scene import Scene, Building, Ground
from urban_auralization.logic.acoustics import Model, MovingSource, Receiver, constants
import glob

#compose the scene
b1 = Building(height = 10,
              vertices=[(0,0),(15,0),(15,10),(25,10),(25,5),(40,5),(40,10),(50,10),(50,0),(65,0),
                        (65,20),(50,20),(50,22.5),(40,22.5),(40,20),(25,20),(25,22.5),(15,22.5),(15,20),(0,20)],
              abs_coeff=0.7)

b2 = Building(height = 10,
              vertices=[(80,-5),(95,-5),(95,5),(105,5),(105,-5),(115,-5),(115,5),(125,5),(125,-5),(135,-5),
                        (135,5),(145,5),(145,-5),(160,-5),(160,15),(145,15),(145,17.5),(135,17.5),
                        (135,15),(125,15),(125,17.5),(115,17.5),(115,15),(105,15),(105,17.5),(95,17.5),
                        (95,15), (80,15), (80,-5)], abs_coeff=0.7)

b3 = Building(vertices=[(175, -5), (190, -5), (190,5), (200, 5), (200, 0), (215,0), (215,5), (225, 5),
                        (225,-5),(240,-5),(240,15),(225,15),(225,17.5),(215,17.5),(215,15),
                        (200,15),(200,17.7),(190,17.5),(190,15),(175,15),(175,-5)], height=10, abs_coeff=0.7)

b4 = Building(height = 12, vertices=[(225,15),((235,15)),(235,30),(225,30),(225,15)])

fysiken = Building(height=20, vertices=[(265, 5),(275,-10),(290,-10),(300,5),(290,20),(275,20),(265,5)], color = "cyan")
b5 = Building(height=8, vertices=[(310,50),(320,50),(320,70),(310,70),(310,50)])

b6 = Building(height=20, vertices=[(70,120),(165,120),(165,135),(70,135),(70,120)], color = "pink")
b7 = Building(height=20, vertices=[(195,120),(300,120),(300,135),(195,135),(195,120)], color = "pink")

features = [b1,b2,b3, b4, fysiken, b5, b6, b7]

#now lets add a source and a receiver
sources_dir = "/Users/jesperholsten/PycharmProjects/aircraft_auralization/aircraft_auralization/data/sources/*.wav"
_SOURCES = glob.glob(sources_dir)

# pick a source to auralize spatially:
source = _SOURCES[2] #13
print(source)

source = MovingSource(waypoints=[(40,105,1), (195,105,1)],
                      velocity=11.1, #40 kmph
                      signal=source)

receiver = Receiver(125, 100, 1.8,
                        elevation=90,
                        azimuth=90)


ground = Ground(buildings=features, color="gray", abs_coeff=0.8)
features.append(ground)

scene = Scene(features=features, source=source, receiver=receiver)

#now we have our scene, lets create the model
model = Model(scene=scene, source_fs=10, fs=44100)

#tweak model settings
model.settings["direct_sound"] = True
model.settings["specular_reflections"] = False
model.settings["diffraction"] = False
model.settings["ism_order"] = 2



#plot scene if we want:
#p = scene.plot()
#p.camera.position = (490,-10,70)
#p.camera.zoom(1.5)
#p.camera.position = (160,70,450)

#p.camera.zoom(1)
#scene.set_camera_zoom(1.5)
#scene.set_camera_position((490,-10,70))
scene.set_camera_position((160,70,450))

#p.show()

model.start()
model.auralize(play = True,save_to_file=False, filename="gibraltargatan.wav")
#model.save_animation(filename="gibraltargatan.mp4")