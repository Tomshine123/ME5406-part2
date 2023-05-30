import math
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering

'''
This is a custom RL environment addressing on a underwater exploration robot in where a part of the code framework refers to openai box2D. 
In the task, the robot agent is dropped from the surface of water and required to reach the goal on the bottom bed in by the influence of gravity 
and floating. To control its pose the robot is equipped with three engines, the main of which is at the bottom of robot and the other two are mounted
in diagnal sides. Along the way there are sub-goals (shown as yellow diamonds)and the robot is encouraged to reach these waypoints to get small rewards,
also a reward is given each time it reach closer to the final goal but a negative rewardas penalty if it crashes with the bed by extreme linear velocity.
Besides, there is a flow force from the water with random direction (shown as the red arrow)for each run to make the task more challenge, the robot
is trained to overcome these difficulties and safely arrive the destination.
'''

# define the shape and size of the robot
ROBOT_POLY = [(-20, +30), (-35, +5), (-10, -20), (+10, -20), (+35, +5), (+20, +30)]

PUSHER_AWAY = 20                            # the horizontal distance between diagnal pusher and body
PUSHER_DOWN = 30                            # the vertical distance between diagnal pusher and body
PUSHER_W = 8 
PUSHER_H = 8                                # the size of diagnal pusher
pusher_SPRING_TORQUE = 70                   # spring torque for the pusher

DIAG_PUSHER_H = 14.0
DIAG_PUSHER_AWAY = 12.0

VIEWPORT_W = 1000                           # GUI window width
VIEWPORT_H = 700                            # GUI window height

FPS = 50                                    # determines how fast the game goes
SCALE = 30.0                                # define a transform of scaling

MAIN_ENGINE_POWER = 30.0                    # the power of the bottom pusher
SIDE_ENGINE_POWER = 5                       # the power of the diagnal pusher

class UnderwaterRobot(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FPS
    }

    def __init__(self):
        self.seed()                                     # define the random seed
        self.viewer = None
        self.world = Box2D.b2World(gravity=(0,-5))      # create the water world with floating 
        self.bed = None                                 # create the water bed
        self.robot = None                               # create the underwater robot
        self.bubbles = []                               # create the bubbles from the pushers

        # define the state space including 8 observations
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)

        # define the action space including the power of bottom and diagnal pushers
        self.action_space = spaces.Discrete(4)

        # initialize the world
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # create the bubbles by Box2D attributes from the pushers
    def _create_bubbles(self, mass, x, y, ttl):                          
        # define the physical properties of bubbles
        p = self.world.CreateDynamicBody(
            position=(x, y-0.5),
            angle=0.0,
            fixtures=fixtureDef(
                shape=circleShape(radius=4 / SCALE, pos=(0, 0)),
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x001,
                restitution=0.3)
        )
        p.ttl = ttl                                                      
        self.bubbles.append(p)
        self._clean_bubbles(False)
        return p

    # clean bubbles away from the robot
    def _clean_bubbles(self, all):                                                   
        while self.bubbles and (all or self.bubbles[0].ttl < 0):
            self.world.DestroyBody(self.bubbles.pop(0))
    
    # add waypoints as sub-goals for robot to reach
    def _create_waypoints(self):
        self.points_poly = []
        num_waypoints = 4   # the number of waypoints
        W = VIEWPORT_W/SCALE 
        H = VIEWPORT_H / SCALE
        # the distribution of waypoints
        position_y = np.linspace(H/10, H, num_waypoints+1, endpoint=False)[1:]
        position_x = np.linspace(self.helipad_one_x, W / 2, num_waypoints+1, endpoint=False)[1:]
        for i in range(0, num_waypoints):
            self.points_poly.append((position_x[i], position_y[i]))
        return self.points_poly
    
    # clean everything in the world
    def _destroy(self):
        if not self.bed: 
            return
        self._clean_bubbles(True)
        self.world.DestroyBody(self.bed)
        self.bed = None

        self.world.DestroyBody(self.robot)
        self.robot = None
        self.world.DestroyBody(self.pushers[0])
        self.world.DestroyBody(self.pushers[1])

    # initialize the game world
    def reset(self):
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False          # flag to check whether the game reaches termination condition
        self.prev_shaped = None         # a shaped reward from the last moment

        W = VIEWPORT_W / SCALE                                            
        H = VIEWPORT_H / SCALE                                            

        # creat the water bed
        CHUNKS = 20                                                # the number of terrain block     
        chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]    # x position of each terrain block
        
        # define the random goal position on the bed
        index = np.random.randint(1, CHUNKS-2)
        self.helipad_one_x = chunk_x[index]     # the x position of goal is randomly initialized
        self.helipad_one_y = H/10               # the height of goal is the same with bed

        # create a static body representing the bed of water
        self.bed = self.world.CreateStaticBody(fixtures=fixtureDef(shape=edgeShape(vertices=[(0, 0), (W, 0)])))     # ???
        self.bed_polys = []
        self.water_polys = []              # the block representing water
        p1 = (0, H/10)                     # the leftmost position of the bed            
        p2 = (W, H/10)                     # the leftmost position of the bed
        
        # define the physical properties of the bed
        self.bed.CreateEdgeFixture(
            vertices=[p1, p2],                                        
            density=0,                                                
            friction=0.1)                                             
        self.bed_polys.append([p1, p2, (p2[0], 0), (p1[0], 0)])
        self.water_polys.append([p1, p2, (p2[0], H), (p1[0], H)])       

        initial_y = VIEWPORT_H / SCALE   # the robot is dropped from the surface of water                               

        self._create_waypoints()         # generate some waypoints for the robot

        # generate the body of robot and its properties by Box2D
        self.robot = self.world.CreateDynamicBody(
            position=(VIEWPORT_W / SCALE / 2, initial_y),                 # starting point: (middle, top)
            angle=0.0,                                                    
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in ROBOT_POLY]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,                                      # the category the robot belongs to
                maskBits=0x001,                                           # the category the robot with collide with
                restitution=0.99)                                         # 0.99 bouncy
        )
        self.robot.color1 = (160/255, 160/255, 160/255)                   # filling color for robot
        self.robot.color2 = (160/255, 160/255, 160/255)                   # outline color for robot
        self.flow_direction = np.random.randint(-1,2)                     # the direction of force is unknwon each time
        
        # generate the two diagnal pushers shown as red blocks
        self.pushers = []
        for i in [-1, +1]:
            pusher = self.world.CreateDynamicBody(
                position=(VIEWPORT_W / SCALE / 2 - i * PUSHER_AWAY / SCALE, initial_y),    # control the position of the pushers
                angle=(i * -0.45),                                                         # control the angle between pushers and body
                fixtures=fixtureDef(
                    shape=polygonShape(box=(PUSHER_W / SCALE, PUSHER_H / SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,                                  
                    maskBits=0x001)                                      
            )
            pusher.ground_contact = False
            pusher.color1 = (1, 0, 0)                                  # filling color for engines
            pusher.color2 = (1, 0, 0)                                  # border color for engines
            rjd = revoluteJointDef(                                    # para about joints between robot and pushers
                bodyA=self.robot,
                bodyB=pusher,
                localAnchorA=(0, 0),
                localAnchorB=(i * PUSHER_AWAY / SCALE, PUSHER_DOWN / SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=pusher_SPRING_TORQUE,
                motorSpeed=+0.3 * i
            )
            if i == -1:
                rjd.lowerAngle = +0.9 - 0.6  
                rjd.upperAngle = +0.36        
            else:
                rjd.lowerAngle = -0.36                                    
                rjd.upperAngle = -0.9 + 0.6                               
            pusher.joint = self.world.CreateJoint(rjd)
            self.pushers.append(pusher)

        self.drawlist = [self.robot] + self.pushers
 
        return self.step(0)[0]

    # define the update method of the world
    def step(self, action):
        if not (self.pushers[0].ground_contact or self.pushers[1].ground_contact):
            # add the flow force in the world
            self.flow_force = 10*self.flow_direction                      # the magnitude of flow force
            self.robot.ApplyForceToCenter((                               # apply the force on the robot                   
            Box2D.b2Vec2(self.flow_force, 0)
            ), True)
        # clip the value of action into the valid range                                                            
        assert self.action_space.contains(action), "%r (%s) invalid " % (
                action,
                type(action),
            ) 

        # direction components for two pushers
        tip = (math.sin(self.robot.angle), math.cos(self.robot.angle))
        side = (-tip[1], tip[0])
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]        

        ## the action part refers to a gym environment LunarLander
        # for the bottom pusher with higher designed power
        bottom_power = 1.0
        if action == 2:
            # decompose the force into x and y positions                                                           
            ox = tip[0] * (4 / SCALE + 2 * dispersion[0]) + side[0] * dispersion[1]        
            oy = -tip[1] * (4 / SCALE + 2 * dispersion[0]) - side[1] * dispersion[1]
            impulse_pos = (self.robot.position[0], self.robot.position[1])     
            p = self._create_bubbles(3.5, impulse_pos[0], impulse_pos[1], bottom_power) 
            # apply the robot on bubbles and the robot body
            p.ApplyLinearImpulse((ox * MAIN_ENGINE_POWER * bottom_power, oy * MAIN_ENGINE_POWER * bottom_power), impulse_pos, True)
            self.robot.ApplyLinearImpulse((-ox * MAIN_ENGINE_POWER * bottom_power, -oy * MAIN_ENGINE_POWER * bottom_power), impulse_pos, True)

        # for the diagnal pusher with lower designed power
        diag_power = 1.0
        if action in [1,3]:
            direction = action - 2
            # decompose the force into x and y positions
            ox = tip[0] * dispersion[0] + side[0] * (3 * dispersion[1] + direction * DIAG_PUSHER_AWAY / SCALE)
            oy = -tip[1] * dispersion[0] - side[1] * (3 * dispersion[1] + direction * DIAG_PUSHER_AWAY / SCALE)
            impulse_pos = (self.robot.position[0] + ox - tip[0] * 17 / SCALE,
                           self.robot.position[1] + oy + tip[1] * DIAG_PUSHER_H / SCALE)
            p = self._create_bubbles(0.7, impulse_pos[0], impulse_pos[1], diag_power)
            # apply the robot on bubbles and the robot body
            p.ApplyLinearImpulse(
                (ox * SIDE_ENGINE_POWER * diag_power, oy * SIDE_ENGINE_POWER * diag_power-1),
                impulse_pos,
                True,
            )
            self.robot.ApplyLinearImpulse((-ox * SIDE_ENGINE_POWER * diag_power, -oy * SIDE_ENGINE_POWER * diag_power),
                                           impulse_pos, True)
        

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)  # world.Step(time_step, velocity_iter, position_iter)
        
        pos = self.robot.position          # get the positon of robot in the world frame
        vel = self.robot.linearVelocity    # get the velocity of robot in the world frame
        # calculate the distance in x direction between the robot and goal
        dis_x = (pos.x - self.helipad_one_x) / (VIEWPORT_W / SCALE / 2)             
        # calculate the distance in y direction between the robot and goal
        dis_y = (pos.y - (self.helipad_one_y + PUSHER_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2)
        # define the observations in state space
        state = [
            dis_x,
            dis_y,
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,  # robot's velocity in x direction                                
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,  # robot's velocity in x direction
            self.robot.angle,                       # robot's angle to avoid rollover
            # the distance to the nearest waypoint
            min(math.sqrt((pos.x - x)**2 + (pos.y - y)**2) for (x,y) in self.points_poly) if len(self.points_poly) != 0 else 0,
        ]
        assert len(state) == 6
        
        # define reward function
        reward = 0      
        # a shaped reward referring to LunarLander                                                 
        shaped = (
            # the distance between the robot and the final goal
            - 100 * np.sqrt(state[0] * state[0] + state[1] * state[1]) 
            # to avoid extreme velocity and angle that leads to crash
            - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3]) 
            - 100 * abs(state[4])  
            )                  

        # calculate the difference of shaped reward between adjacent moments                                                                          
        if self.prev_shaped is not None:
            reward = shaped - self.prev_shaped
        self.prev_shaped = shaped

        # the pusher consume energy for each frame
        reward -= bottom_power * 0.30                                                    
        reward -= diag_power * 0.03
        
        # the robot gets a small reward each time it reaches a waypoint 
        for i, (x, y) in enumerate(self.points_poly):
            distance = math.sqrt((self.robot.position.x - x)**2 + (self.robot.position.y - y)**2)
            if distance < 1.5:  
                reward += 100
                self.points_poly.pop(i)       # clean the reached point
                break                                                     
        
        # define the termination condition
        done = False
        if self.game_over or abs(state[0]) >= 1.0:                                     
            done = True
            # penalty if the robot crashes with high velocity
            if np.sqrt(state[2] * state[2] + state[3] * state[3]) > 0.3:
                reward = -200

        return np.array(state, dtype=np.float32), reward, done, {}                  

    # visualize the game world in a window
    def render(self, mode='human'):                                                 
        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE 
        if self.viewer is None:  
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W / SCALE, 0, VIEWPORT_H / SCALE)

        # generate bubbles from the pushers
        for obj in self.bubbles:
            obj.ttl -= 0.15
            obj.color1 = (0.0, 0.40, 0.80)
            obj.color2 = (1,1,1)
        self._clean_bubbles(False)

        # generate the water environment
        for p in self.water_polys:
            self.viewer.draw_polygon(p, color=(0.0, 0.40, 0.80))                           

        # generate the waypoints shown as diamonds
        for pos in self.points_poly:
            self.viewer.draw_polygon(v=((pos[0], pos[1]),
                                        (pos[0] + 0.5, pos[1]),
                                        (pos[0] + 0.7, pos[1]-0.2),
                                        (pos[0] + 0.25, pos[1]-0.6),
                                        (pos[0] - 0.2, pos[1]-0.2),
                                       ),
                                     color=(1.0, 1.0, 0))
        
        # generate an arrow showing the direction of the flow
        self.arrow_y = 7*H/8
        if self.flow_direction > 0:
            self.arrow_x = 5
        else:
            self.arrow_x = W - 5
        self.viewer.draw_polygon(v=((self.arrow_x, self.arrow_y), 
                                    (self.arrow_x-self.flow_direction*4,self.arrow_y+2),
                                    (self.arrow_x-self.flow_direction*3,self.arrow_y),
                                    (self.arrow_x-self.flow_direction*4,self.arrow_y-2)), 
                                    color=(1, 0, 0))
        
        # draw the block of water bed
        for p in self.bed_polys:
            self.viewer.draw_polygon(p, color=(0.54, 0.23, 0))
        for obj in self.bubbles + self.drawlist:
            for f in obj.fixtures:                                                 
                trans = f.body.transform
                # draw the bubbles
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans * f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                # draw the body of the robot
                else:
                    path = [trans * v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)


        # draw the final goal shown as a red flag
        flag_y1 = self.helipad_one_y; 
        flag_y2 = self.helipad_one_y + 70/SCALE
        flag_x = self.helipad_one_x
        self.viewer.draw_polyline(v=((flag_x, flag_y1), (flag_x, flag_y2)), color=(1, 1, 1))
        self.viewer.draw_polygon(v=((flag_x, flag_y2),
                                    (flag_x, flag_y2- 25/SCALE),
                                    (flag_x+40/SCALE, flag_y2-25/SCALE),
                                    (flag_x+40/SCALE, flag_y2)),
                                 color=(178/255, 34/255, 34/255))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

# function from LunarLander to check contact between two objects 
class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if self.env.robot == contact.fixtureA.body or self.env.robot == contact.fixtureB.body:
            self.env.game_over = True
        for i in range(2):
            if self.env.pushers[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.pushers[i].ground_contact = True

    def EndContact(self, contact):
        for i in range(2):
            if self.env.pushers[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.pushers[i].ground_contact = False

