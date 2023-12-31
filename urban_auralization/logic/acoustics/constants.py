"---------------CONSTANTS-------------------"
C_AIR = 343 #m/s
RHO_AIR = 1.2 #kg/m3

"---------------FREQUENCY-DEPENDANT VARIABLES-------------------"
f_third = [2,2.5,3.15,4,5,6.3,8,10,12.5,16,20,25,31.5,40,50,63,80,100,
               125,160,200,250,315,400,500,630,800,1000,1250,1600,2000,2500,
               3150,4000,5000,6300,8000,10000,12500,16000,20000,25000,31500]

#Random Incidence Absorption Coefficients
water = {125 : 0.01, 250 : 0.01, 500: 0.01, 1000 : 0.01, 2000 : 0.02, 4000 : 0.02}
concrete = {125 : 0.02, 250 : 0.03, 500: 0.03, 1000 : 0.03, 2000 : 0.04, 4000 : 0.04}
painted_bricks = {125 : 0.05, 250 : 0.04, 500: 0.02, 1000 : 0.04, 2000 : 0.05, 4000 : 0.05}
wood_paneling = {125 : 0.30, 250 : 0.25, 500: 0.20, 1000 : 0.17, 2000 : 0.15, 4000 : 0.10}
asphalt = {125 : 0.02, 250 : 0.03, 500: 0.03, 1000 : 0.03, 2000 : 0.03, 4000 : 0.02}
metal_panel = {125 : 0.59, 250 : 0.80, 500: 0.82, 1000 : 0.65, 2000 : 0.27, 4000 : 0.23}
grass = {63 : 0.50 ,125 : 0.62, 250 : 0.75, 500: 0.87, 1000 : 0.95, 2000 : 0.98, 4000 : 0.99}
neutral = {63 : 0 ,125 : 0, 250 : 0, 500: 0, 1000 : 0, 2000 : 0, 4000 : 0}
