# -*- coding: utf-8 -*-
"""
Physarum simulation

Based on the work of Jeff Jones, UWE https://uwe-repository.worktribe.com/output/980579
and Sage Jensen https://sagejenson.com/physarum

@author: Amitav Mitra
"""

# handle imports
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import ndimage
import matplotlib.animation as animation   # only needed if you want to make an animation

class Environment:
    def __init__(self, N=200, M=200, pp=0.15):
        '''
        pp = percentage of the map size to generate population. default 15% - 6000 particles in 200x200 environment
        '''
        self.N = N
        self.M = M
        self.data_map = np.zeros(shape=(N,M))
        self.trail_map = np.zeros(shape=(N,M))
        self.population = int((self.N*self.M)*(pp))
        self.particles = []   # holds particles 
        
    def populate(self, SA=np.pi/4, RA=np.pi/8, SO=9):
        '''
        randomly populates pp% of the map with particles of:
        SA = Sensor Angle
        RA = Rotation Angle
        SO = Sensor Offset
        '''
        while (np.sum(self.data_map) < self.population):   # loop until population size met
            rN = np.random.randint(self.N)
            rM = np.random.randint(self.M)
            if (self.data_map[rN,rM] == 0):
                p = Particle((rN,rM),SA,RA,SO)
                self.particles.append(p)   # list holds particle and its position
                self.data_map[rN,rM] = 1   # assign a value of 1 to the particle location 
            else:
                pass
            
    def deposit_food(self, pos, strength=3, rad=6):
        '''
        applies a circular distribution of food to the trail map
        '''
        n, m = pos                                  # location of food
        y, x = np.ogrid[-n:self.N-n, -m:self.M-m]   # work directly on pixels of the trail map
        mask = x**2 + y**2 <= rad**2                # create circular mask of desired radius
        self.trail_map[mask] = strength  
        
    def diffusion_operator(self, const=0.6, sigma=2):
        '''
        applies a Gaussian filter to the entire trail map, spreading out chemoattractant
        const multiplier controls decay rate (lower = greater rate of decay, keep <1)
        Credit to: https://github.com/ecbaum/physarum/blob/8280cd131b68ed8dff2f0af58ca5685989b8cce7/species.py#L52
        '''
        self.trail_map = const * ndimage.gaussian_filter(self.trail_map,sigma)
    
    def check_surroundings(self, point, angle):
        '''
        Helper function for motor_stage()
        Determines if the adjacent spot in the data map is available, based on particle angle
        '''
        n,m = point
        x = np.cos(angle)
        y = np.sin(angle)
        # periodic BCs -> %
        if (self.data_map[(n-round(x))%self.N,(m+round(y))%self.M]==0):   # position unoccupied, move there
            return ((n-round(x))%self.N,(m+round(y))%self.M)
        elif (self.data_map[(n-round(x))%self.N,(m+round(y))%self.M]==1):   # position occupied, stay
            return point
       
    def motor_stage(self):
        '''
        Scheduler function - causes every particle in population to undergo motor stage
        Particles randomly sampled to avoid long-term bias from sequential ordering
        '''
        rand_order = random.sample(self.particles, len(self.particles))
        for i in range(len(rand_order)):
            old_x, old_y = rand_order[i].pos
            new_x, new_y = self.check_surroundings(rand_order[i].pos, rand_order[i].phi)
            if ((new_x,new_y) == (old_x,old_y)):   # move invalid, stay and choose new orientation, update sensors
                rand_order[i].phi = 2*np.pi*np.random.random()
                rand_order[i].update_sensors()
            else:   # move valid: move there, change value in data map accordingly, deposit trail, AND change particle position
                rand_order[i].pos = (new_x,new_y)
                rand_order[i].update_sensors()
                self.data_map[old_x,old_y] = 0
                self.data_map[new_x,new_y] = 1
                rand_order[i].deposit_phermone_trail(self.trail_map)
    
    def sensory_stage(self):
        '''
        Makes every particle undergo sensory stage in random order
        '''
        rand_order = random.sample(self.particles, len(self.particles))
        for i in range(len(rand_order)):
            rand_order[i].sense(self.trail_map)
            
class Particle:
    def __init__(self, pos, SA=np.pi/8, RA=np.pi/4, SO=3.):
        '''
        Initializes physical characteristics of the particle
        pos = (n,m) in data map
        phi = initial random angle of orientation of the particle [0,2pi]
        SA = +- sensor angle wrt phi
        SO = sensor offset from body
        SS = step size - DONT USE
        RA = rotation angle of particle
        '''
        self.pos = pos
        self.phi = 2*np.pi*np.random.random()
        self.SA = SA
        self.RA = RA
        self.SO = SO
        # initialize sensor angles wrt body - will be updated as particle moves
        self.phi_L = self.phi - SA   # left sensor
        self.phi_C = self.phi        # center sensor  - probably redundant, can just use self.phi 
        self.phi_R = self.phi + SA   # right sensor
        
    def deposit_phermone_trail(self, arr, strength=1.):
        '''
        Applies a single trail of chemoattractant at current position
        '''
        n, m = self.pos
        arr[n,m] = strength
        
    def update_sensors(self):
        '''
        Updates the sensor positions relative to the particle's orientation
        (Left, Center, Right)
        '''
        self.phi_L = self.phi - self.SA
        self.phi_C = self.phi              
        self.phi_R = self.phi + self.SA
        
    def get_sensor_values(self, arr):
        '''
        Finds the value of the chemoattractant at each of the 3 sensors
        Pass the TrailMap array as an argument
        '''
        n,m = self.pos
        row,col = arr.shape

        xL = round(self.SO*np.cos(self.phi_L))
        yL = round(self.SO*np.sin(self.phi_L))
        xC = round(self.SO*np.cos(self.phi_C))
        yC = round(self.SO*np.sin(self.phi_C))
        xR = round(self.SO*np.cos(self.phi_R))
        yR = round(self.SO*np.sin(self.phi_R))
        
        # implement periodic BCs
        valL = arr[(n-xL)%row,(m+yL)%col] 
        valC = arr[(n-xC)%row,(m+yC)%col]
        valR = arr[(n-xR)%row,(m+yR)%col]  

        return (valL,valC,valR)
    
    def sense(self, arr):
        '''
        The particle reads from the trail map, rotates based on chemoattractant
        arr = trail map array
        '''
        L,C,R = self.get_sensor_values(arr)

        if ((C>L) and (C>R)):   # Center > both: stay facing same direction, do nothing
            self.phi += 0
            self.update_sensors()
        elif ((L==R) and C<L):   # L, R are equal, center is less - randomly rotate L/R
            rn = np.random.randint(2)
            if rn == 0:
                self.phi += self.RA
                self.update_sensors()
            else:
                self.phi -= self.RA
                self.update_sensors()
        elif (R>L):
            self.phi += self.RA
            self.update_sensors()
        elif (L>R):
            self.phi -= self.RA
            self.update_sensors()
        else:   # all three are the same - stay facing same direction
            self.phi += 0
            self.update_sensors()

def scheduler(N=200, M=200, pp=0.07, sigma=0.65, const=0.85, 
              SO=9, SA=np.pi/8, RA=np.pi/4, steps=500,  
              intervals=8, plot=True, animate=False):
    '''
    generates the environment (NxM) with pp% of environment populated
    particles: Sensor Offset, Sensor Angle, Rotation Angle
    chemoattractant: constant multiplier, sigma (gaussian filter)
    evolve simulation for 500 steps, grab plots at specific intervals
    choice to plot intervals OR animate the desired simulation 
    '''
    Env = Environment(N, M, pp)
    Env.populate(SA, RA, SO)
    
    if (plot==True):
        dt = int(steps/intervals)
        samples = np.linspace(0, dt*intervals, intervals+1)   # integer samples
        for i in range(steps):
            Env.diffusion_operator(const,sigma)
            Env.motor_stage()
            Env.sensory_stage()
            if i in samples:
                fig = plt.figure(figsize=(8,8),dpi=200);
                ax1 = fig.add_subplot(111);
                ax1.imshow(Env.trail_map);
                ax1.set_title('Chemoattractant Map, step={}'.format(i));
                # display some information about parameters used
                ax1.text(0,-10,'SA: {:.2f}  SO: {}  RA: {:.2f}  pop: {:.0f}%'.format(np.degrees(SA),SO,np.degrees(RA),pp*100));   # hard code -10, since most likely using big grid
                plt.savefig('sim_t{}.png'.format(i));
                plt.clf();
                
    elif (animate==True):
        # this can take a while for large environments, high population
        # also generates very large .gif files, play with values to get smaller files
        ims = []
        fig = plt.figure(figsize=(8,8),dpi=100);
        ax = fig.add_subplot(111);
        for i in range(steps):
            Env.diffusion_operator(const,sigma)
            Env.motor_stage()
            Env.sensory_stage()
            txt = plt.text(0,-10,'iteration: {}    SA: {:.2f}    SO: {}    RA: {:.2f}    %pop: {}%'.format(i,np.degrees(SA),SO,np.degrees(RA),pp*100));
            im = plt.imshow(Env.trail_map, animated=True);
            ims.append([im,txt])
        fig.suptitle('Chemoattractant Map');
        ani = animation.ArtistAnimation(fig,ims,interval=50,blit=True,repeat_delay=1000);
        ani.save('sim.gif');
        
def main():
    '''
    runs the scheduler as is, with default parameters except lower step size
    generates 8 plots in the directory at different time intervals.
    '''
    scheduler(steps=100)
    return 0
    
if __name__ == "__main__":
    main()

