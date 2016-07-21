import cv2
import numpy as np
import time
from recordclass import recordclass

namedtuple = recordclass

Wall = namedtuple('Wall', 'begin end')
Item = namedtuple('Item', 'position reward radius age')
Agent = namedtuple('Agent', 'position angle wheel1 wheel2')
Eye = namedtuple('Eye', 'angle proximity type color')

WIDTH = 700
HEIGHT = 500
RADIUS = 10
NUM_ITEMS = 80
PAD = 5

MAX_AGE = 5000
REMOVE_PROB = 0.1

EYE_RANGE = 100
EYE_RESOLUTION = 15
EYE_HALFCOUNT = 5

FRAME_SKIP = 1

REWARDS = {
    5: (0, 255, 0),
    -6: (0, 0, 255),
}

REWARDS_LIST = list(REWARDS.keys())
ACTIONS = np.array([[1, 1], [0.8, 1], [1, 0.8], [0.5, 0], [0, 0.5]])
OBSERVATION_LENGTH_MAIN = (2 * EYE_HALFCOUNT + 1) * (1 + len(REWARDS))
OBSERVATION_LENGTH = OBSERVATION_LENGTH_MAIN + len(ACTIONS)

def intersect_wall(begin, end, wall):
    pos = begin - wall.begin
    dr = end - begin
    wal = wall.end - wall.begin

    perp = np.array([[0, 1], [-1, 0]]).dot(wal)

    denom = np.dot(dr, perp)
    numer = np.dot(pos, perp)

    if np.isclose(denom, 0):
        return None
    else:
        t = - numer / denom
        if t < 0 or t > 1:
            return None
        else:
            return t

def rotate(vec, angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s], [s, c]]).dot(vec)

def intersect_item(begin, end, item):
    dr = end - begin
    v = begin - item.position

    c = np.dot(v, v) - item.radius ** 2
    b = 2 * np.dot(dr, v)
    a = np.dot(dr, dr)

    delta = b*b - 4*a*c

    if delta < 0:
        return None

    cands = [(-b + np.sqrt(delta))/(2 * a),
             (-b - np.sqrt(delta))/(2 * a)]

    cands = [cand for cand in cands if cand >= 0 and cand <= 1]

    if not cands:
        return None
    else:
        return np.min(cands)

def random_item():
    return Item(
                position=np.array([np.random.randint(PAD, WIDTH - PAD), np.random.randint(PAD, HEIGHT - PAD)]),
                reward=np.random.choice(REWARDS.keys()),
                radius=RADIUS,
                age=0,
            )

class Game(object):
    num_actions = len(ACTIONS)

    def __init__(self):
        self.walls = [
                Wall(begin=np.array([PAD, PAD]), end=np.array([WIDTH - PAD, PAD])),
                Wall(begin=np.array([PAD, PAD]), end=np.array([PAD, HEIGHT - PAD])),
                Wall(begin=np.array([PAD, HEIGHT - PAD]), end=np.array([WIDTH - PAD, HEIGHT - PAD])),
                Wall(begin=np.array([WIDTH - PAD, PAD]), end=np.array([WIDTH - PAD, HEIGHT - PAD])),
        ]

        self.items = [random_item() for _ in xrange(NUM_ITEMS)]

        self.agent = Agent(
            position=np.array([WIDTH/2, HEIGHT/2]),
            angle=0,
            wheel1=0,
            wheel2=0,
        )

        self.eyes = [
                Eye(
                    angle=2 * np.pi / EYE_RESOLUTION * x,
                    proximity=None,
                    color=(0, 0, 0),
                    type=None,
                )
                for x in range(-EYE_HALFCOUNT, EYE_HALFCOUNT + 1)
            ]

        self.update()

        self.last_action = 0

    def eye_end(self, eye):
        angle = self.agent.angle + eye.angle
        end = self.agent.position + EYE_RANGE * np.array([np.cos(angle), np.sin(angle)])
        return end

    def update(self):
        for eye in self.eyes:
            eye.proximity = None
            eye.color = (127, 127, 127)
            eye.type = None

            for wall in self.walls:
                proximity = intersect_wall(self.agent.position, self.eye_end(eye), wall)
                if proximity is None:
                    continue
                
                if eye.proximity is None or eye.proximity > proximity:
                    eye.proximity = proximity
                    eye.color = (0, 0, 0)
                    eye.type = 'wall'

            for item in self.items:
                proximity = intersect_item(self.agent.position, self.eye_end(eye), item)
                if proximity is None:
                    continue
                
                if eye.proximity is None or eye.proximity > proximity:
                    eye.proximity = proximity
                    eye.color = REWARDS[item.reward]
                    eye.type = item.reward


    def draw(self, window='preview'):
        img = np.full((HEIGHT, WIDTH, 3), 255, np.uint8)

        for wall in self.walls:
            cv2.line(img, tuple(wall.begin), tuple(wall.end), (0, 0, 0), 2)

        for item in self.items:
            cv2.circle(img, tuple(item.position), item.radius, REWARDS[item.reward], -1)

        for eye in self.eyes:
            end = self.eye_end(eye)
            thickness = 1 if eye.proximity is None else 2
            cv2.line(img, tuple(self.agent.position), tuple(map(int, end)), eye.color, thickness)

        cv2.circle(img, tuple(self.agent.position), RADIUS, (0, 0, 0), -1)

        cv2.imshow(window, img)

    def wall_between(self, begin, end):
        return any(intersect_wall(begin, end, wall) is not None for wall in self.walls)

    def dist_to_wall(self):
        return min(self.agent.position[0],
                      WIDTH - self.agent.position[0],
                      self.agent.position[1],
                      HEIGHT - self.agent.position[1])

    def observe(self):
        res = np.full((OBSERVATION_LENGTH_MAIN,), 1.0)

        c = 1 + len(REWARDS)
        
        for i, eye in enumerate(self.eyes):
            if eye.proximity is not None:
                v = eye.proximity

                if eye.type == 'wall':
                    res[c * i] = v
                else:
                    res[c * i + 1 + REWARDS_LIST.index(eye.type)] = v

        acts = np.zeros((len(ACTIONS),))
        acts[self.last_action] = 1.0 * OBSERVATION_LENGTH_MAIN
        return np.concatenate((res, acts))

    def act(self, action):
        reward = 0
        for _ in xrange(FRAME_SKIP):
            reward += self._act(action)
        self.last_action = action
        return reward

    def _act(self, action):
        [wheel1, wheel2] = ACTIONS[action]

        pos = self.agent.position
        v = rotate(np.array([0, RADIUS/2]), self.agent.angle + 0 * np.pi/2)
        w1 = pos + v
        w2 = pos - v
        vv1 = rotate(pos - w2, -wheel1)
        vv2 = rotate(pos - w1, wheel2)
        pos = np.array(map(int, (w1 + w2 + vv2 + vv1) / 2))

        self.agent.angle -= wheel1
        self.agent.angle += wheel2

        while self.agent.angle < 0:
            self.agent.angle += 2 * np.pi

        while self.agent.angle > 2 * np.pi:
            self.agent.angle -= 2 * np.pi

        pos[0] = max(0, min(WIDTH, pos[0]))
        pos[1] = max(0, min(HEIGHT, pos[1]))

        if not self.wall_between(self.agent.position, pos):
            self.agent.position = pos

        self.update()

        items_reward = 0.0
        to_remove = []
        for idx, item in enumerate(self.items):
            dr = item.position - self.agent.position
            item.age += 1

            remove = False

            if np.sqrt(dr.dot(dr)) < item.radius + RADIUS:
                if not self.wall_between(self.agent.position, item.position):
                    items_reward += item.reward
                    remove = True

            if item.age >= MAX_AGE and np.random.rand() < REMOVE_PROB:
                remove = True

            if remove:
                to_remove.append(idx)

        for idx in reversed(to_remove):
            self.items.pop(idx)
            self.items.append(random_item())

        self.update()

        proximity_reward = 0.0
        for eye in self.eyes:
            if eye.type == 'wall':
                proximity_reward += eye.proximity / EYE_RANGE
            else:
                proximity_reward += 1

        proximity_reward /= len(self.eyes)
        proximity_reward = min(1, proximity_reward * 2)

        forward_reward = 0
        if action == 0 and proximity_reward > 0.75:
            forward_reward = 0.1 * proximity_reward

        return proximity_reward + forward_reward + items_reward

if __name__ == '__main__':
    cv2.startWindowThread()
    cv2.namedWindow("preview")
    game = Game()
    game.update()
    game.draw()

    while True:
        try:
            act = int(input())
            assert act >= 0 and act < 5
        except Exception as ex:
            print ex
            continue

        print game.observe()
        reward = game.act(act)
        print "reward", reward
        game.draw()
