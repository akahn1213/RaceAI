import pyglet
import config
import neuralnet
from system.component import Component
from entities.car import Car
from random import randint
from pyglet.window import key
from pyglet.image import Animation, AnimationFrame
import numpy as np
import time as tim
from time import gmtime, strftime
import os, sys

wbatch = pyglet.graphics.Batch() #Idk wtf this does I tried to make the cars explode and this seemed relevant

window = pyglet.window.Window(height=config.window_height,
                              width=config.window_width)
keys = [False, False, False, False] #Used to debug the level with keyboard controls
window.push_handlers(keys)

cars = []
n_cars = 10
n_enabled_cars = 0
n_chromosomes = n_cars #aka bad programming
chromosomes = []
population = [] #When a car crashes or gets stuck, its chromosome will be appended to the "population" list
population_fitness = []
stuck_time = 10. #After this many seconds, if a car hasn't moved stuck_distance pixels, then it is considered stuck and sent to the netherlands
stuck_distance = 300. #If a car hasn't moved this many pixels in stuck_time seconds, it is stuck
starting_positions = []


#Lists 10 starting positions which will be randomly assigned to the 10 cars each time a run occurs
#For tracks 2 & 3
starting_positions.append([472, 137]) #[x, y] in number of pixels. The origin is at the bottom left of the window
starting_positions.append([572, 137])
starting_positions.append([672, 137])
starting_positions.append([772, 137])
starting_positions.append([872, 137])
starting_positions.append([472, 83])
starting_positions.append([572, 83])
starting_positions.append([672, 83])
starting_positions.append([772, 83])
starting_positions.append([872, 83])
#For track 1
"""
starting_positions.append([472, 109])
starting_positions.append([572, 109])
starting_positions.append([672, 109])
starting_positions.append([772, 109])
starting_positions.append([872, 109])
starting_positions.append([472, 46])
starting_positions.append([572, 46])
starting_positions.append([672, 46])
starting_positions.append([772, 46])
starting_positions.append([872, 46])
"""


#Load a set of saved chromosomes
chromosomes = neuralnet.import_chromosomes('02_24_13_47_02') #Uncomment to use saved chromosomes, then comment the random chromosome generation
for i in range(n_chromosomes):
    #OR Start with 10 cars with random chromosomes
    #chromosomes.append(neuralnet.generate_random_chromosome()) #Uncomment to use random chromosomes, then comment the loading of saved chromosomes
    cars.append(Car(x = starting_positions[i][0], y = starting_positions[i][1], chromosome = chromosomes[i]))
n_enabled_cars = n_cars #Enable all cars. Disabled cars will not be drawn, and once all cars are disabled, either after crashing or getting stuck, a new run will start
    
    
time_start = tim.time() #Our good boy tim, he has an expensive wrist watch
time_passed = 0 #This will be used to count up to stuck_time, to check if any cars are stuck

def enableCars(chromes): #In the future, everything is chromes
    global starting_positions
    global cars
    global n_enabled_cars
    for i in range(len(cars)):
        cars[i].enable(starting_positions[i], chromes[i])
        n_enabled_cars += 1
    print(len(cars))
    
def draw():
    """
    Clears screen and then renders our list of cars
    """
    window.clear()
    config.background.blit(0, 0) #Draw the background image defined in config.py
    for car in cars: #Draw each car
        if(car.enabled): car.draw_self()

def update(time):
    """
    Updates our cars and clocks and all that good stuff
    """
    global time_start
    global time_passed
    global population
    global population_fitness
    global chromosomes
    global starting_positions
    global n_enabled_cars
    global cars
    global stuck_time
    time_passed = tim.time() - time_start #Keep up the good work tim


    #Check if any cars are stuck after stuck_time seconds have passed
    if(time_passed > stuck_time):
        for car in cars:
            if(car.enabled):
                if(np.absolute(car.distance_traveled - car.distance_check) < stuck_distance): #Condition for a car being stuck
                    car.stuck = True
                car.distance_check = car.distance_traveled #If it is not stuck, set it's starting distance for the next round of stuck-checking to be its current distance
        time_start = tim.time() #Countin' on you, tim
        time_passed = 0. #Reset the clock
    
    #When all cars have been deleted
    if(n_enabled_cars == 0):
        pyglet.clock.unschedule(update) #I though this would help fix some strange bugs, by turning off the game clock until the re-initialization of cars is complete. Idk if that's true but I'm keeping it anyway
        #Generate a new population
        chromosomes = neuralnet.get_new_generation(population, population_fitness) #Sends the previous population of chromosomes to the neural net and produces a new generation
        np.random.shuffle(chromosomes) #Reduce bias by randomizing the chromosomes
        population = []
        population_fitness = []
        np.random.shuffle(starting_positions) #Reduce bias by also randomizing the starting positions
        for i in range(len(cars)): #Enable each car with a new chromosome
            cars[i].enable(starting_positions[i], chromosomes[i]) 
            n_enabled_cars += 1
        pyglet.clock.schedule_interval(update, 1/60.0) #Restart the game clock, which ticks at (ideally) 60fps
        time_start = tim.time() #Tell tim to do what he does best
        time_passed = 0. #Reset the clock
    
    #Check if any cars have crashed, and disable them
    for car in cars:
        #Uncomment the next 4 lines to enable the arrow keys to control the cars
        #car.steering_change = 5*(int(keys[0]) - int(keys[1]))
        #if(not (keys[0] or keys[1])):
        #    car.steering_angle = 0
        #car.acceleration = (int(keys[2])-int(keys[3]))*(0.06*int(keys[2]) + 0.3*int(keys[3]))
        if(car.enabled):
            car.update_self()
            if (car.collision or car.stuck):
                population.append(car.chromosome)
                population_fitness.append(car.distance_traveled)
                car.disable()
                n_enabled_cars -= 1
                
@window.event
def on_key_press(k, modifiers):
    if k==key.LEFT: keys[0] = True
    if k==key.RIGHT: keys[1] = True
    if k==key.UP: keys[2] = True
    if k==key.DOWN: keys[3] = True
    if k==key.ENTER: #Save current chromosomes to a directory titled by the date and time
        timestamp = strftime("%m_%d_%H_%M_%S", gmtime())
        directory = sys.path[0]+'\chromosomes\\'
        if not os.path.exists(directory+str(timestamp)):
            os.makedirs(directory+str(timestamp))
        for i in range(len(chromosomes)):
            np.save(directory+str(timestamp)+'/chromosome_'+str(i)+'.npy', chromosomes[i])
            
@window.event
def on_key_release(k, modifiers):
    if k==key.LEFT: keys[0] = False
    if k==key.RIGHT: keys[1] = False
    if k==key.UP: keys[2] = False
    if k==key.DOWN: keys[3] = False

	
def main():
    @window.event
    def on_draw():
        draw()
    pyglet.clock.schedule_interval(update, 1/60.0)
    pyglet.app.run()

main()
