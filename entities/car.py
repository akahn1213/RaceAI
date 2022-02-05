import pyglet
from system.component import Component
import config
import neuralnet
import numpy as np
from pyglet.window import key
from pyglet.gl import *



#Get the alpha value of the pixel at the point [x, y]. Input is a numpy array
def getPixelAlpha(pos):
    if(((config.image_data_width*pos[1] + pos[0])*4 + 4 > config.n_pixels) or ((config.image_data_width*pos[1] + pos[0])*4 + 4 < 0)):
        return 0
    else: return (config.pixels[(config.image_data_width*pos[1] + pos[0])*4:(config.image_data_width*pos[1] + pos[0])*4 + 4])[3]
    
def addVec(v1, length, angle):
    return np.add(v1, length*np.asarray([np.cos(np.radians(angle)), np.sin(np.radians(angle))]))      


#Recursively determines the point of a wall within 5 pixels given an initial point and an angle to search for the wall 
def calcCollisionPoint(pos, length, angle):
    if(length < 5): return pos
    else:
        if(getPixelAlpha(np.around(pos).astype(int)) < 250):
            return calcCollisionPoint(addVec(pos, -np.round(length/2), angle), np.absolute(np.round(length/4)), angle)
        else:
            return calcCollisionPoint(addVec(pos, length, angle), np.absolute(length), angle)

def distance(vec1, vec2):
    return np.sqrt(np.square(vec2[0]-vec1[0]) + np.square(vec2[1]-vec1[1]))

    
#Gets a color given 3 numbers between -2 and 2
#To avoid dark colors, if the sum of the resulting RGB values is less than 256, it adds the difference to all 3 channels    
def getColor(colors):
    color1 = int((colors[0]+2)*256/4)
    color2 = int((colors[1]+2)*256/4)
    color3 = int((colors[2]+2)*256/4)
    if(color1+color2+color3 < 256): #Add color to avoid very dark cars
        add_value = 256 - (color1+color2+color3)
        color1+=add_value
        color2+=add_value
        color3+=add_value      
    return np.asarray([color1, color2, color3]).astype(int)
    

#THE CAR
class Car(Component):

    def __init__(self, *args, **kwargs):
        """
        Creates a sprite using a car image.
        """
        super(Car, self).__init__(*args, **kwargs)
        
        #Status variables
        self.collision = False
        self.stuck = False
        self.past_line = False
        self.enabled = True
        self.distance_traveled = 0
        self.distance_check = 0
        self.last_distance = 0
        self.distance_past_line = 0
        self.time_alive = 0
        
        #Image  and shape variables
        self.car_details_image = pyglet.image.load('assets/car_details.png')
        self.car_body_image = pyglet.image.load('assets/car_body.png')
        self.wheel_image = pyglet.image.load('assets/wheel.png')
        self.collision_image = pyglet.image.load('assets/ball.png')
        self.collision_image.anchor_x = self.collision_image.width//2
        self.collision_image.anchor_y = self.collision_image.height//2
        self.car_details_image.anchor_x = self.car_details_image.width//2
        self.car_details_image.anchor_y = self.car_details_image.height//2
        self.car_body_image.anchor_x = self.car_body_image.width//2
        self.car_body_image.anchor_y = self.car_body_image.height//2
        self.wheel_image.anchor_x = self.wheel_image.width//2
        self.wheel_image.anchor_y = self.wheel_image.height//2
        self.width = self.car_body_image.width
        self.height = self.car_body_image.height
        self.half_diagonal = 0.8*np.sqrt(np.square(self.width) + np.square(self.height))/2.
        self.corner_angle = np.degrees(np.arctan2(self.height, self.width))
        self.car_details_sprite = pyglet.sprite.Sprite(self.car_details_image, self.x, self.y)
        self.car_body_sprite = pyglet.sprite.Sprite(self.car_body_image, self.x, self.y)
            

        #Position and direction (car + wheels)       
        self.half_wheelbase = 27
        self.wheel_offset = 15
        self.pos = np.asarray([self.x, self.y]) #Position of the center of the car
        self.direction = kwargs.get('direction', 0) #Angle of the car
        self.steering_angle = kwargs.get('angle', 0)
        self.front_pos = addVec(self.pos, self.half_wheelbase, self.direction)
        self.rear_pos = addVec(self.pos, -self.half_wheelbase, self.direction)        
        self.wheelpos_r = addVec(self.front_pos, self.wheel_offset, self.direction+90)
        self.wheelpos_l = addVec(self.front_pos, self.wheel_offset, self.direction-90)
        self.wheel_l_sprite = pyglet.sprite.Sprite(self.wheel_image, self.wheelpos_l[0], self.wheelpos_l[1])
        self.wheel_r_sprite = pyglet.sprite.Sprite(self.wheel_image, self.wheelpos_r[0], self.wheelpos_r[1]) 
        self.l_corner = np.around(addVec(self.pos, self.half_diagonal, self.direction - self.corner_angle)).astype(int)
        self.r_corner = np.around(addVec(self.pos, self.half_diagonal, self.direction + self.corner_angle)).astype(int)
        self.l_back_corner = np.around(addVec(self.pos, -self.half_diagonal, self.direction + self.corner_angle)).astype(int)
        self.r_back_corner = np.around(addVec(self.pos, -self.half_diagonal, self.direction - self.corner_angle)).astype(int)        

        #Motion Variables
        self.speed = kwargs.get('speed', 0)       
        self.acceleration = 0
        self.steering_change = 0
        self.max_speed = 10
        self.max_angle = 60
        self.min_angle = 45
       
        #Neural Net related variables
        self.chromosome = kwargs.get('chromosome', np.zeros(neuralnet.n_genes + 3))
        self.car_body_sprite.color = getColor(self.chromosome[neuralnet.n_genes::])   
        self.fitness = 1
                
        #Collision related variables
        self.collision_point_forward = calcCollisionPoint(self.pos, config.search_length, self.direction)
        self.collision_point_left1 = calcCollisionPoint(self.pos, config.search_length, self.direction + self.corner_angle+10)
        self.collision_point_right1 = calcCollisionPoint(self.pos, config.search_length, self.direction - self.corner_angle-10)
        self.collision_point_left2 = calcCollisionPoint(self.pos, config.search_length, self.direction + self.corner_angle-20)
        self.collision_point_right2 = calcCollisionPoint(self.pos, config.search_length, self.direction - self.corner_angle+20)
        self.collision_point_side_left = calcCollisionPoint(self.pos, config.search_length, self.direction + 90)
        self.collision_point_side_right = calcCollisionPoint(self.pos, config.search_length, self.direction + 90)
        self.distance_forward = distance(self.pos, self.collision_point_forward)
        self.distance_left1 = distance(self.pos, self.collision_point_left1)
        self.distance_right1 = distance(self.pos, self.collision_point_right1)
        self.distance_left2 = distance(self.pos, self.collision_point_left2)
        self.distance_right2 = distance(self.pos, self.collision_point_right2)
        self.distance_side_left = distance(self.pos, self.collision_point_side_left)
        self.distance_side_right = distance(self.pos, self.collision_point_side_right)
        self.collision_sprite_f = pyglet.sprite.Sprite(self.collision_image, self.collision_point_forward[0], self.collision_point_forward[1])
        self.collision_sprite_l = pyglet.sprite.Sprite(self.collision_image, self.collision_point_left1[0], self.collision_point_left1[1])
        self.collision_sprite_r = pyglet.sprite.Sprite(self.collision_image, self.collision_point_right1[0], self.collision_point_right1[1])


    #Enable the car, allows it to be drawn and controlled
    def enable(self, pos, chromosome):
        self.pos = pos
        self.chromosome = chromosome
        self.collision = False
        self.stuck = False
        self.past_line = False
        self.enabled = True
    
    #Disable the car
    def disable(self):
        self.speed = 0
        self.direction = 0       
        self.car_body_sprite.color = getColor(self.chromosome[neuralnet.n_genes::])
        self.distance_traveled = 0
        self.distance_check = 0
        self.collision_point_forward = calcCollisionPoint(self.pos, config.search_length, self.direction)
        self.collision_point_left1 = calcCollisionPoint(self.pos, config.search_length, self.direction + self.corner_angle+10)
        self.collision_point_right1 = calcCollisionPoint(self.pos, config.search_length, self.direction - self.corner_angle-10)
        self.collision_point_left2 = calcCollisionPoint(self.pos, config.search_length, self.direction + self.corner_angle-20)
        self.collision_point_right2 = calcCollisionPoint(self.pos, config.search_length, self.direction - self.corner_angle+20)
        self.collision_point_side_left = calcCollisionPoint(self.pos, config.search_length, self.direction + 90)
        self.collision_point_side_right = calcCollisionPoint(self.pos, config.search_length, self.direction + 90)
        self.distance_forward = distance(self.pos, self.collision_point_forward)
        self.distance_left1 = distance(self.pos, self.collision_point_left1)
        self.distance_right1 = distance(self.pos, self.collision_point_right1)
        self.distance_left2 = distance(self.pos, self.collision_point_left2)
        self.distance_right2 = distance(self.pos, self.collision_point_right2)
        self.distance_side_left = distance(self.pos, self.collision_point_side_left)
        self.distance_side_right = distance(self.pos, self.collision_point_side_right)
        self.time_alive = 0
        self.fitness = 1
        self.distance_past_line = 0
        self.collision = True
        self.enabled = False
        
    def update_self(self):
        """
        Sends inputs to the neural net, calculates acceleration + steering, moves the car, and recalculates distances and inputs for the next update
        """
        
        #Calculate Acceleration and Steering via neural net
        #The inputs to the neural net are the following:
        #1-7: The distance between the center of the car and the wall in the direction of:
        #1.     -Directly to the front of the car
        #2.     -32 Degrees counterclockwise from the front of the car
        #3.     -32 Degrees clockwise from the front of the car
        #4.     -12 Degrees counterclockwise from the front of the car
        #5.     -12 Degrees clockswise from the front of the car
        #6.     -Directly to the left of the car
        #7.     -Directly to the right of the car
        #8.     The speed of the car
        #The outputs of the neural network are the acceleration and amount to change the steering angle of the car
        self.acceleration, self.steering_change = neuralnet.compute(self.distance_forward, self.distance_left1, self.distance_right1, self.distance_left2, self.distance_right2, self.distance_side_left, self.distance_side_right, self.speed, self.chromosome[:neuralnet.n_genes])
        
        #Change speed and steering
        self.speed += self.acceleration
        if(self.speed > self.max_speed):
          self.speed = self.max_speed
        elif(self.speed < 0.5):
          self.speed = 0.5
        self.steering_angle = np.absolute(self.steering_change)*(self.steering_angle+self.steering_change)
        if(self.steering_angle > self.max_angle - self.speed*((self.max_angle - self.min_angle)/self.max_speed)):
          self.steering_angle = self.max_angle - self.speed*((self.max_angle - self.min_angle)/self.max_speed)
        elif(self.steering_angle < -(self.max_angle - self.speed*((self.max_angle - self.min_angle)/self.max_speed))):
          self.steering_angle = -(self.max_angle - self.speed*((self.max_angle - self.min_angle)/self.max_speed))

          
        #Move the car
        
        ##Get the location of the front and back wheels
        self.front_pos = addVec(self.pos, self.half_wheelbase, self.direction)
        self.rear_pos = addVec(self.pos, -self.half_wheelbase, self.direction)

        ##Move the wheels
        self.front_pos = addVec(self.front_pos, self.speed, self.direction + self.steering_angle)
        self.rear_pos = addVec(self.rear_pos, self.speed, self.direction)

        ##Update position and direction for car and wheels
        oldpos = self.pos
        self.pos = np.mean(np.asarray([self.rear_pos, self.front_pos]), axis=0)
        self.direction = np.degrees(np.arctan2(self.front_pos[1] - self.rear_pos[1], self.front_pos[0] - self.rear_pos[0]))
        if(self.past_line == False and self.pos[0] > 1000.): self.past_line = True
        self.distance_traveled += np.sqrt(np.square(self.pos[0] - oldpos[0]) + np.square(self.pos[1] - oldpos[1]))
        if(self.past_line):
            self.distance_past_line += np.sqrt(np.square(self.pos[0] - oldpos[0]) + np.square(self.pos[1] - oldpos[1]))
            self.time_alive += 1
            distance_ratio = np.min(np.array(self.distance_left1, self.distance_right1))/np.max(np.array(self.distance_left1, self.distance_right1))
            if(self.enabled): self.fitness += np.sqrt(np.square(self.pos[0] - oldpos[0]) + np.square(self.pos[1] - oldpos[1])) + 3*distance_ratio
        
        ##Recalculate Distances
        self.wheelpos_r = addVec(self.front_pos, self.wheel_offset, self.direction+90)
        self.wheelpos_l = addVec(self.front_pos, self.wheel_offset, self.direction-90)
        self.l_corner = np.around(addVec(self.pos, self.half_diagonal, self.direction - self.corner_angle)).astype(int)
        self.r_corner = np.around(addVec(self.pos, self.half_diagonal, self.direction + self.corner_angle)).astype(int)
        self.l_back_corner = np.around(addVec(self.pos, -self.half_diagonal, self.direction + self.corner_angle)).astype(int)
        self.r_back_corner = np.around(addVec(self.pos, -self.half_diagonal, self.direction - self.corner_angle)).astype(int)

        ##Set new positions and directions
        self.car_details_sprite.update(self.pos[0], self.pos[1], -self.direction)
        self.car_body_sprite.update(self.pos[0], self.pos[1], -self.direction)
        self.wheel_l_sprite.update(self.wheelpos_l[0], self.wheelpos_l[1], -self.direction - self.steering_angle)
        self.wheel_r_sprite.update(self.wheelpos_r[0], self.wheelpos_r[1], -self.direction - self.steering_angle)
        

        
        #Check for collisions
        #This is done in the following way:
        #Find the points on the left and right front corners of the car
        #If the background image's pixel value at one of those points has an alpha of less than 250, then it is a wall. The track part has an alpha of 255
        pixel_l = getPixelAlpha(self.l_corner)
        pixel_r = getPixelAlpha(self.r_corner) 
        pixel_back_l = getPixelAlpha(self.l_back_corner)
        pixel_back_r = getPixelAlpha(self.r_back_corner)
        if(pixel_l < 250 or pixel_r < 250):
            self.collision = True     

        #Calculate the distances forward, left+right, and in four directions around the two front corners, to the nearest wall
        #This has an extra step compared to the collision case above
        #1. Draw a line in the direction that you want to find the distance to the wall in
        #2. Check if the pixel that is search_length pixels in that direction from the center of the car is a wall by seeing if its alpha value is < 250
        #3. If it is:
        #-.     Move search_length/2 pixels backward, check if that is a wall, and repeat, either moving length/2 forward (not a wall) or length/2 backward(is a wall) until the search length is < 5 pixels long
        #4. If it is not:
        #-.     Check the pixel that is another search_length further along that line
        #5. Return the coordinates of the point retrieved when this recursive function stops, and calculate the distance between the car's position and that point
        self.collision_point_forward = calcCollisionPoint(self.pos, config.search_length, self.direction)
        self.collision_point_left1 = calcCollisionPoint(self.pos, config.search_length, self.direction + self.corner_angle+10)
        self.collision_point_right1 = calcCollisionPoint(self.pos, config.search_length, self.direction - self.corner_angle-10)
        self.collision_point_left2 = calcCollisionPoint(self.pos, config.search_length, self.direction + self.corner_angle-20)
        self.collision_point_right2 = calcCollisionPoint(self.pos, config.search_length, self.direction - self.corner_angle+20)
        self.collision_point_side_left = calcCollisionPoint(self.pos, config.search_length, self.direction + 90)
        self.collision_point_side_right = calcCollisionPoint(self.pos, config.search_length, self.direction + 90)
        self.distance_forward = distance(self.pos, self.collision_point_forward)
        self.distance_left1 = distance(self.pos, self.collision_point_left1)
        self.distance_right1 = distance(self.pos, self.collision_point_right1)
        self.distance_left2 = distance(self.pos, self.collision_point_left2)
        self.distance_right2 = distance(self.pos, self.collision_point_right2)
        self.distance_side_left = distance(self.pos, self.collision_point_side_left)
        self.distance_side_right = distance(self.pos, self.collision_point_side_right)


    def draw_self(self):
        """
        Draws the car to the screen
        """
        self.car_details_sprite.draw()
        self.car_body_sprite.draw()
        self.wheel_l_sprite.draw()
        self.wheel_r_sprite.draw()

        

    def get_fitness(self):
        return self.fitness
		



