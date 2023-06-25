#import the modules we need
from urban_auralization.logic.scene import Scene, Building, Ground
from urban_auralization.logic.acoustics import Model, MovingSource, Receiver, constants
import glob
import random

random.seed(1)

#compose the scene
b1 = Building(height = random.uniform(2,10),
              vertices=[(-3,1),(3,1)], color = "brown")

b2 = Building(height = random.uniform(2,10),
              vertices=[(-5,2),(3,2)], color = "brown")
b2c = Building(height = random.uniform(2,10),
              vertices=[(-5,2),(4,2)], color = "brown")

b3 = Building(height = random.uniform(2,10),
              vertices=[(-1,3),(1,3)], color = "brown")

b4 = Building(height = random.uniform(2,10),
              vertices=[(-7,4),(7,4)], color = "brown")


features = [b1,b2]#,b3, b4]

#now lets add a source and a receiver
sources_dir = "/Users/jesperholsten/PycharmProjects/aircraft_auralization/aircraft_auralization/data/sources/*.wav"
_SOURCES = glob.glob(sources_dir)

# pick a source to auralize spatially:
source = _SOURCES[2] #13
print(source)

source = MovingSource(waypoints=[(-10,0,1), (10,0,1)],
                      velocity=3,
                      signal=source)

receiver = Receiver(0, 5, 1.8,
                        elevation=90,
                        azimuth=90)


ground = Ground(buildings=[b1,b2c,b3,b4], color="gray", abs_coeff=0.8)
features.append(ground)

scene = Scene(features=features, source=source, receiver=receiver)

#now we have our scene, lets create the model
model = Model(scene=scene, source_fs=10, fs=44100)

#tweak model settings
model.settings["direct_sound"] = False
model.settings["specular_reflections"] = False
model.settings["diffraction"] = True
model.settings["ism_order"] = 2



#plot scene if we want:
p = scene.plot(t=6.5)

p.show()
#model.start()
#model.auralize(save_to_file=True, filename="gibraltargatan.wav")
#model.save_animation(filename="gibraltargatan.mp4")