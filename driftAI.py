import pygame as pg
import numpy as np
import matplotlib.pyplot as plt

WIDTH, HEIGHT = 600, 600
SCREEN_WIDTH = 1200
FPS = 60
PLAYER_SIZE, COIN_SIZE = 20, 10
PLAYER_X, PLAYER_Y = 300, 300
MAX_SPEED = 15

SHOW_TEXT, GRAPH_LOG, GRAPH_NUM = True, True, 0
SAVE_MODE = False
BEST_DRAW = False

BLACK, WHITE, RED, GREEN, BLUE, YELLOW = (0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)
COLOR_LIST = [(i, 255, 0) for i in range(0, 256, 51)] + [(255, i, 0) for i in range(255, -1, -51)] + [(i, 0, 0) for i in range(255, -1, -51)] + [(0, 0, 0) for i in range(100)]

INPUTS, OUTPUTS = 7, 2
NODE_ID, INNOVATION = INPUTS + OUTPUTS, 0
POPULATION = 2000

c1, c2, c3 = 1, 1, 0.4
MAX_WEIGHT, MAX_BIAS = 5, 5
DELTA_THRESHOLD = 0.4
DEL_NODE, ADD_NODE = 0.01, 0.02
DEL_LINK, ADD_LINK = 0.05, 0.2
MUTATE_PROB = 0.7
ACTIVATION_MODE = 2
ACTIVATION_THRESHOLD = [0, 0.5, 0, 0.5]
MAX_LAYER = 4

pg.init()
screen = pg.display.set_mode((SCREEN_WIDTH, HEIGHT))
pg.display.set_caption("driftAI")
font = pg.font.Font("font.ttf", 24)
clock = pg.time.Clock()

def pseudo_random(num, min_val, max_val):
    a = 1664525
    c = 1013904223
    m = 2**32
    
    num = (a * num + c) % m
    scaled_num = min_val + (num % (max_val - min_val + 1))
    
    return scaled_num

def squash(x, n):
    if n == 0:
        return x
    if n == 1:
        return 1/(1 + np.exp(-x))
    if n == 2:
        return np.tanh(x)
    if n == 3:
        return max(0, x)
    
def log(x, n):
    if n:
        return np.log10(1+x)
    return x

class Node:
    def __init__(self, id, bias=0, activation=0):
        self.id = id
        self.bias = bias
        self.activation = activation
    
    def clone(self):
        n = Node(self.id, self.bias, self.activation)
        return n
        
class Link:
    def __init__(self, in_id, out_id, weight=0, enabled=True, innov=-1):
        self.in_id = in_id
        self.out_id = out_id
        self.weight = weight
        self.enabled = enabled
        self.innov = innov
        
    def clone(self):
        l = Link(self.in_id, self.out_id, self.weight, self.enabled, self.innov)
        return l

class Genome:
    def __init__(self):
        self.nodes = np.array([Node(i, 0, 0) for i in range(INPUTS)] + [Node(i, 0, ACTIVATION_MODE) for i in range(INPUTS, INPUTS + OUTPUTS)])
        self.links = np.array([], dtype=object)
        self.fitness = 0
        self.avg_fitness = 0
        self.score = 0
        self.avg_score = 0
        self.species = 0
        self.reload()

    def clone(self):
        g = Genome()
        g.nodes = np.array([n.clone() for n in self.nodes])
        g.links = np.array([c.clone() for c in self.links])
        g.fitness = self.fitness
        g.avg_fitness = self.avg_fitness
        g.score = self.score
        g.avg_score = self.avg_score
        g.species = self.species
        g.reload()
        return g

    def reload(self):
        for node in self.nodes[INPUTS + OUTPUTS:]:
            if not np.any([link.in_id == node.id and link.enabled for link in self.links]) or not np.any([link.out_id == node.id and link.enabled for link in self.links]):
                self.nodes = self.nodes[self.nodes != node]
        self.links = np.array([link for link in self.links if link.in_id in [n.id for n in self.nodes] and link.out_id in [n.id for n in self.nodes]])

        self.id_to_index = {n.id: i for i, n in enumerate(self.nodes)}
        self.layer = np.zeros(len(self.nodes))
        self.layer[:INPUTS] = 0
        self.layer[INPUTS:] = 1
        self.order = []
        self.list_order = []
        
        queue = [[n.id, []] for n in self.nodes]
        while queue:
            first = queue.pop(0)
            n, prev = first[0], first[1]
            for link in self.links:
                if link.in_id == n and link.enabled:
                    if link.out_id in prev:
                        link.enabled = False
                    else:
                        self.layer[self.id_to_index[link.out_id]] = max(self.layer[self.id_to_index[link.out_id]], self.layer[self.id_to_index[n]] + 1)
                        queue.append([link.out_id, prev + [n]])

        for i in range(INPUTS, INPUTS + OUTPUTS):
            self.layer[i] = max(self.layer[INPUTS+OUTPUTS:]) + 1 if self.layer[INPUTS+OUTPUTS:].size > 0 else 1
        self.max_layer = int(max(self.layer))
        self.layer_dict = {i: [n.id for n in self.nodes if self.layer[self.id_to_index[n.id]] == i] for i in range(self.max_layer + 1)}
        
        for i in range(self.max_layer + 1):
            self.order += self.layer_dict[i]

        for n in self.order:
            for link in self.links:
                if link.in_id == n and link.enabled:
                    self.list_order.append(link)

    def distance(self, other):
        excess, disjoint, weight, same = 0, 0, 0, 1
        i, j = 0, 0
        while i < len(self.links) and j < len(other.links):
            if self.links[i].innov == other.links[j].innov:
                weight += abs(self.links[i].weight - other.links[j].weight)
                same += 1
                i += 1
                j += 1
            elif self.links[i].innov < other.links[j].innov:
                disjoint += 1
                i += 1
            else:
                disjoint += 1
                j += 1
        excess += len(self.links) - i + len(other.links) - j
        n = min(max(len(self.links), len(other.links), 1), 30)
        return (c1 * excess + c2 * disjoint) / n + c3 * weight / same

    def feed_forward(self, inputs):
        self.value = np.zeros(len(self.nodes))
        self.value[:INPUTS] = inputs
        for i in range(len(self.list_order)):
            link = self.list_order[i]
            if (i == 0 or link.in_id != self.list_order[i - 1].in_id) and link.in_id >= INPUTS + OUTPUTS:
                idx = self.id_to_index[link.in_id]
                self.value[idx] = squash(self.value[idx] + self.nodes[idx].bias, self.nodes[idx].activation)
            self.value[self.id_to_index[link.out_id]] += self.value[self.id_to_index[link.in_id]] * link.weight
        for i in range(INPUTS, INPUTS + OUTPUTS):
            self.value[i] = squash(self.value[i] + self.nodes[i].bias, self.nodes[i].activation)
        return self.value[INPUTS:INPUTS+OUTPUTS]

    def add_node(self, link):
        global NODE_ID, INNOVATION
        link.enabled = False
        new_node = Node(NODE_ID, 0, 0)
        self.nodes = np.append(self.nodes, new_node)
        self.links = np.append(self.links, [Link(link.in_id, NODE_ID, 1, True, INNOVATION), 
                                            Link(NODE_ID, link.out_id, link.weight, True, INNOVATION + 1)])
        INNOVATION += 2
        NODE_ID += 1

    def add_link(self):
        global INNOVATION
        in_node = np.random.choice([n.id for n in np.concatenate((self.nodes[:INPUTS], self.nodes[INPUTS + OUTPUTS:]))])
        out_node = np.random.choice([n.id for n in self.nodes[INPUTS:]])
        if self.layer[self.id_to_index[in_node]] > self.layer[self.id_to_index[out_node]]:
            in_node, out_node = out_node, in_node
        elif in_node == out_node:
            return
        for link in self.links:
            if link.in_id == in_node and link.out_id == out_node:
                link.enabled = True
                link.weight = 0
                return
        self.links = np.append(self.links, Link(in_node, out_node, 0, True, INNOVATION))
        INNOVATION += 1

    def delete_node(self):
        node = np.random.choice([n.id for n in self.nodes[INPUTS + OUTPUTS:]])
        self.nodes = self.nodes[self.nodes != node]
        self.links = self.links[np.logical_not(np.isin([link.in_id for link in self.links], node)) & 
                                np.logical_not(np.isin([link.out_id for link in self.links], node))]

    def delete_link(self):
        if len(self.links) > 0:
            link = np.random.choice(self.links)
            link.enabled = False

    def change_weight(self):
        if len(self.links) > 0:
            link = np.random.choice(self.links)
            link.weight += np.random.randn() / 10
            link.weight = np.clip(link.weight, -MAX_WEIGHT, MAX_WEIGHT)

    def change_bias(self):
        if len(self.nodes) > INPUTS:
            node = np.random.choice(self.nodes[INPUTS:])
            node.bias += np.random.randn() / 10
            node.bias = np.clip(node.bias, -MAX_BIAS, MAX_BIAS)

    def change_activation(self):
        if len(self.nodes) > INPUTS+OUTPUTS:
            node = np.random.choice(self.nodes[INPUTS+OUTPUTS:])
            node.activation = ACTIVATION_MODE if node.activation == 0 else 0

    def mutate_node(self):
        r = np.random.rand()
        if r < DEL_NODE and len(self.nodes) > INPUTS + OUTPUTS:
            self.delete_node()
        elif r < DEL_NODE + ADD_NODE and len([link for link in self.links if link.enabled]) > 0:
            self.add_node(np.random.choice([link for link in self.links if link.enabled]))
        elif len(self.nodes) > INPUTS + OUTPUTS:
            if np.random.rand() < 0.5:
                self.change_bias()
            else:
                self.change_activation()

    def mutate_link(self):
        r = np.random.rand()
        if r < DEL_LINK and len(self.links) > 0:
            self.delete_link()
        elif r < DEL_NODE + ADD_LINK:
            self.add_link()
        elif len(self.links) > 0:
            self.change_weight()

    def mutate(self):
        r = np.random.rand()
        if r < MUTATE_PROB:
            self.mutate_link()
        else:
            self.mutate_node()
        self.reload()

    def crossover(self, other):
        child = Genome()
        child.nodes = np.array([])
        i, j = 0, 0
        while i < len(self.nodes) and j < len(other.nodes):
            if self.nodes[i].id == other.nodes[j].id:
                child.nodes = np.append(child.nodes, self.nodes[i].clone())
                i += 1
                j += 1
            elif self.nodes[i].id < other.nodes[j].id:
                child.nodes = np.append(child.nodes, self.nodes[i].clone())
                i += 1
            else:
                child.nodes = np.append(child.nodes, other.nodes[j].clone())
                j += 1
        while i < len(self.nodes):
            child.nodes = np.append(child.nodes, self.nodes[i].clone())
            i += 1
        while j < len(other.nodes):
            child.nodes = np.append(child.nodes, other.nodes[j].clone())
            j += 1

        i, j = 0, 0
        while i < len(self.links) and j < len(other.links):
            if self.links[i].innov == other.links[j].innov:
                if np.random.rand() < 0.5:
                    child.links = np.append(child.links, self.links[i].clone())
                else:
                    child.links = np.append(child.links, other.links[j].clone())
                i += 1
                j += 1
            elif self.links[i].innov < other.links[j].innov:
                if np.random.rand() < 0.9:
                    child.links = np.append(child.links, self.links[i].clone())
                i += 1
            else:
                if np.random.rand() < 0.9:
                    child.links = np.append(child.links, other.links[j].clone())
                j += 1
        while i < len(self.links):
            child.links = np.append(child.links, self.links[i].clone())
            i += 1
        while j < len(other.links):
            child.links = np.append(child.links, other.links[j].clone())
            j += 1

        child.reload()
        return child
    
class Player:
    def __init__(self, genome=None):
        self.x = PLAYER_X + np.random.uniform(-10, 10)
        self.y = PLAYER_Y + np.random.uniform(-10, 10)
        self.vx = 0
        self.vy = 0
        self.ax = 0
        self.ay = 0
        self.prev_fitness = 0
        self.penalty = 0
        self.input = []
        self.genome = genome if genome else Genome()
        
    def reset(self):
        self.x = PLAYER_X + np.random.uniform(-10, 10)
        self.y = PLAYER_Y + np.random.uniform(-10, 10)
        self.vx = 0
        self.vy = 0
        self.ax = 0
        self.ay = 0
        self.penalty = 0
        self.genome.fitness = 0
        self.genome.score = 0

    def update(self):
        coin_pos1 = coin_list[player.genome.score]
        coin_pos2 = coin_list[min(player.genome.score + 1, len(coin_list) - 1)]
        if np.sqrt((player.x - coin_pos1[0])**2 + (player.y - coin_pos1[1])**2) < PLAYER_SIZE + COIN_SIZE:
            player.genome.score += 1

        diff1 = np.array([coin_pos1[0] - self.x, coin_pos1[1] - self.y])
        diff2 = np.array([coin_pos2[0] - self.x, coin_pos2[1] - self.y])
        radius1 = np.linalg.norm(diff1)
        radius2 = np.linalg.norm(diff2)
        dir1 = diff1 / radius1 if radius1 != 0 else np.array([0, 0])
        dir2 = diff2 / radius2 if radius2 != 0 else np.array([0, 0])
        self.input = [dir1[0], dir1[1], radius1/WIDTH, dir2[0], dir2[1], self.vx/MAX_SPEED, self.vy/MAX_SPEED]
        outputs = self.genome.feed_forward(self.input)
        if outputs[0] < 0:
            self.ax = -0.5
        if outputs[0] > 0:
            self.ax = 0.5
        if outputs[1] < 0:
            self.ay = -0.5
        if outputs[1] > 0:
            self.ay = 0.5

        self.vx += self.ax
        self.vy += self.ay
        self.vx = np.clip(self.vx, -MAX_SPEED, MAX_SPEED)
        self.vy = np.clip(self.vy, -MAX_SPEED, MAX_SPEED)
        self.x += self.vx
        self.y += self.vy

        self.genome.fitness = max(self.genome.score + 1/(1 + (np.sqrt((self.x - coin_pos1[0])**2 + (self.y - coin_pos1[1])**2))/WIDTH) - self.penalty, 0)
        if self.genome.fitness < self.prev_fitness:
            self.penalty += 0.001
        self.prev_fitness = self.genome.fitness

    def draw(self):
        pg.draw.circle(screen, COLOR_LIST[self.genome.species], (int(self.x), int(self.y)), PLAYER_SIZE)
        pg.draw.circle(screen, WHITE, (int(self.x), int(self.y)), PLAYER_SIZE, 4)
        vx = self.vx / np.linalg.norm([self.vx, self.vy]) if np.linalg.norm([self.vx, self.vy]) != 0 else 0
        vy = self.vy / np.linalg.norm([self.vx, self.vy]) if np.linalg.norm([self.vx, self.vy]) != 0 else 0
        ax = self.ax / np.linalg.norm([self.ax, self.ay]) if np.linalg.norm([self.ax, self.ay]) != 0 else 0
        ay = self.ay / np.linalg.norm([self.ax, self.ay]) if np.linalg.norm([self.ax, self.ay]) != 0 else 0
        pg.draw.line(screen, BLUE, (int(self.x), int(self.y)), (int(self.x + vx*20), int(self.y + vy*20)), 5)
        pg.draw.line(screen, RED, (int(self.x), int(self.y)), (int(self.x + ax*20), int(self.y + ay*20)), 5)
        text = font.render(f"{self.genome.fitness:.2f}", True, WHITE)
        screen.blit(text, (int(self.x) - 20, int(self.y) - 65))
        text = font.render(f"-{self.penalty:.2f}", True, RED)
        screen.blit(text, (int(self.x) - 20, int(self.y) - 45))

def read_genome(file):
    genome = Genome()
    with open(file, "r") as f:
        for i in range(INPUTS + OUTPUTS):
            id, bias, activation = map(float, f.readline().split())
            genome.nodes[i] = Node(id, bias, activation)
        for line in f:
            in_id, out_id, weight, enabled, innov = map(float, line.split())
            genome.links = np.append(genome.links, Link(in_id, out_id, weight, enabled, innov))
    genome.reload()
    return genome

def push_innov():
    global NODE_ID, INNOVATION

    node_ids = np.sort(np.unique(np.array([node.id for player in population for node in player.genome.nodes])))
    link_innovs = np.sort(np.unique(np.array([link.innov for player in population for link in player.genome.links])))
    node_id_map = {node_ids[i]: i for i in range(len(node_ids))}
    link_innov_map = {link_innovs[i]: i for i in range(len(link_innovs))}
    NODE_ID = len(node_ids)
    INNOVATION = len(link_innovs)

    for player in population:
        for node in player.genome.nodes:
            node.id = node_id_map[node.id]
        for link in player.genome.links:
            link.in_id = node_id_map[link.in_id]
            link.out_id = node_id_map[link.out_id]
            link.innov = link_innov_map[link.innov]
        player.genome.reload()
        player.reset()

def speciate(population):
    species = []
    for genome in population:
        for s in species:
            if genome.distance(s[0]) < DELTA_THRESHOLD:
                s.append(genome)
                break
        else:
            if len(species) < 50:
                species.append([genome])
            else:
                species.sort(key=lambda x: max([g.avg_fitness for g in x]), reverse=True)
                species[-1].append(genome)
    species.sort(key=lambda x: max([g.avg_fitness for g in x]), reverse=True)
    for s in species:
        s.sort(key=lambda x: x.avg_fitness, reverse=True)
    for i, s in enumerate(species):
        for g in s:
            g.species = i
    return species

def reproduce(population):
    global species
    population.sort(key=lambda x: x.genome.avg_fitness, reverse=True)
    species = speciate([player.genome for player in population])
    new_population = [population[i].genome for i in range(POPULATION//100)]
    for i in range(len(species)//2+1):
        n = 0
        if i <= 5:
            n = 50
        elif i <= 10:
            n = 20
        elif i <= 15:
            n = 5
        s = species[i]
        for _ in range(POPULATION//20+n):
            j = min(int(abs(np.random.randn())*2), 5, len(s)-1)
            child = s[j].clone()
            for _ in range(int(abs(np.random.randn())*2)+1):
                child.mutate()
            if child.max_layer <= MAX_LAYER and max([len(l) for l in child.layer_dict.values()]) <= INPUTS:
                child.avg_fitness = s[j].avg_fitness
                child.avg_score = s[j].avg_score
                new_population.append(child)
    for s in species:
        for i in range(min(len(s), 30)):
            parent1 = np.random.choice(s[:min(len(s)//4+1, 5)])
            parent2 = np.random.choice(s[:min(len(s)//4+1, 5)])
            child = parent1.crossover(parent2)
            if child.max_layer <= MAX_LAYER and max([len(l) for l in child.layer_dict.values()]) <= INPUTS:
                if parent1.avg_fitness > parent2.avg_fitness:
                    child.avg_fitness = parent1.avg_fitness
                    child.avg_score = parent1.avg_score
                else:
                    child.avg_fitness = parent2.avg_fitness
                    child.avg_score = parent2.avg_score
                new_population.append(child)
    for p in sorted(population, key=lambda x: x.genome.fitness, reverse=True):
        new_population.append(p.genome.clone())
    return [Player(genome) for genome in new_population[:POPULATION]]

def draw_stats():
    text = font.render("Slow" if speed[speed_idx] == 0.1 else "" if speed[speed_idx] == 1 else "Fast", True, WHITE)
    screen.blit(text, (WIDTH - 55, 10))
    text = font.render(f"Generation: {gen}", True, WHITE)
    screen.blit(text, (WIDTH + 20, 10))
    text = font.render(f"Time: {(time/100):.2f}", True, WHITE)
    screen.blit(text, (WIDTH + 20, 40))
    text = font.render(f"Alive: {len(population)}", True, WHITE)
    screen.blit(text, (WIDTH + 20, 70))
    text = font.render(f"Species: {[len(s) for s in species]}", True, WHITE)
    screen.blit(text, (WIDTH + 20, 100))
    
    top, bottom = 140, 300
    w, h, g1, g2 = 90, 340, 80, 45
    min_value, max_value = -1, 1
    best = max(population, key=lambda x: x.genome.avg_fitness).genome
    
    if SHOW_TEXT:
        for i, value in enumerate(max(population, key=lambda x: x.genome.fitness).input):
            text = font.render(f"{value:.2f}", True, WHITE)
            screen.blit(text, (WIDTH + 20, h - 12 + i*g2))
        for i, value in enumerate(max(population, key=lambda x: x.genome.fitness).genome.value[INPUTS:INPUTS+OUTPUTS]):
            text = font.render(f"{value:.2f}", True, WHITE)
            screen.blit(text, (WIDTH + 100 + best.max_layer*g1 + 20, h - 12 + i*g2))
    for i in range(best.max_layer + 1):
        for node in best.layer_dict[i]:
            for link in best.links:
                if link.in_id == node and link.enabled:
                    pos1 = (WIDTH + w + i*g1, h + best.layer_dict[i].index(node)*g2)
                    pos2 = (WIDTH + w + best.layer[best.id_to_index[link.out_id]]*g1, h + best.layer_dict[best.layer[best.id_to_index[link.out_id]]].index(link.out_id)*g2)
                    pg.draw.line(screen, RED if link.weight > 0 else BLUE if link.weight < 0 else (50, 50, 50), pos1, pos2, abs(int(link.weight)) + 2)
        for node in best.layer_dict[i]:
            b = int(255 * (best.value[best.id_to_index[node]] - min_value) / (max_value - min_value))
            b = max(0, min(255, b))
            pg.draw.circle(screen, (b, b, b) if best.nodes[best.id_to_index[node]].bias == 0 else ((b, 0, 0) if best.nodes[best.id_to_index[node]].bias > 0 else (0, 0, b)), (WIDTH + w + i*g1, h + best.layer_dict[i].index(node)*g2), 20)
            pg.draw.circle(screen, WHITE if best.nodes[best.id_to_index[node]].activation == 0 else GREEN, (WIDTH + w + i*g1, h + best.layer_dict[i].index(node)*g2), 20, 3)
            if SHOW_TEXT:
                text = font.render(str(node), True, WHITE)
                screen.blit(text, (WIDTH + w + i*g1 - 4.7 - 5.2*int(np.log10(node if node != 0 else 1)), h + best.layer_dict[i].index(node)*g2 - 12))
    pg.draw.line(screen, WHITE, (WIDTH + 20, bottom), (SCREEN_WIDTH - 20, bottom), 5)
    pg.draw.line(screen, WHITE, (WIDTH + 20, top), (WIDTH + 20, bottom), 5)
    
    if GRAPH_NUM == 0:
        max_score = max(max(best_score + best_avg_score), 1)
        for i in range(len(best_score) - 1):
            pg.draw.line(screen, RED, (WIDTH + 20 + i*(SCREEN_WIDTH-WIDTH-40)/(len(best_score) - 1), bottom - log(best_score[i], GRAPH_LOG)*(bottom - top)/log(max_score, GRAPH_LOG)), (WIDTH + 20 + (i + 1)*(SCREEN_WIDTH-WIDTH-40)/(len(best_score) - 1), bottom - log(best_score[i + 1], GRAPH_LOG)*(bottom - top)/log(max_score, GRAPH_LOG)), 5)
        for i in range(len(best_avg_score) - 1):
            pg.draw.line(screen, BLUE, (WIDTH + 20 + i*(SCREEN_WIDTH-WIDTH-40)/(len(best_avg_score) - 1), bottom - log(best_avg_score[i], GRAPH_LOG)*(bottom - top)/log(max_score, GRAPH_LOG)), (WIDTH + 20 + (i + 1)*(SCREEN_WIDTH-WIDTH-40)/(len(best_avg_score) - 1), bottom - log(best_avg_score[i + 1], GRAPH_LOG)*(bottom - top)/log(max_score, GRAPH_LOG)), 5)
        text = font.render(f"{max_score:.2f}", True, WHITE)
        screen.blit(text, (WIDTH + 30, 135))
    elif GRAPH_NUM == 1:
        max_fitness = max(best_fitness + best_avg_fitness)
        for i in range(len(best_fitness) - 1):
            pg.draw.line(screen, (255, 255, 0), (WIDTH + 20 + i*(SCREEN_WIDTH-WIDTH-40)/(len(best_fitness) - 1), bottom - log(best_fitness[i], GRAPH_LOG)*(bottom - top)/log(max_fitness, GRAPH_LOG)), (WIDTH + 20 + (i + 1)*(SCREEN_WIDTH-WIDTH-40)/(len(best_fitness) - 1), bottom - log(best_fitness[i + 1], GRAPH_LOG)*(bottom - top)/log(max_fitness, GRAPH_LOG)), 5)
        for i in range(len(best_avg_fitness) - 1):
            pg.draw.line(screen, (255, 0, 255), (WIDTH + 20 + i*(SCREEN_WIDTH-WIDTH-40)/(len(best_avg_fitness) - 1), bottom - log(best_avg_fitness[i], GRAPH_LOG)*(bottom - top)/log(max_fitness, GRAPH_LOG)), (WIDTH + 20 + (i + 1)*(SCREEN_WIDTH-WIDTH-40)/(len(best_avg_fitness) - 1), bottom - log(best_avg_fitness[i + 1], GRAPH_LOG)*(bottom - top)/log(max_fitness, GRAPH_LOG)), 5)
        text = font.render(f"{max_fitness:.2f}", True, WHITE)
        screen.blit(text, (WIDTH + 30, 135))
    elif GRAPH_NUM == 2:
        max_score = 1 if len(score_list[gen-1]) == 0 else max(max(score_list[gen-1].keys()), 1)
        max_count = 0 if len(score_list[gen-1]) == 0 else max(score_list[gen-1].values())
        prev = (WIDTH + 20, bottom if 0 not in score_list[gen-1] else bottom - log(score_list[gen-1][0]*(bottom - top)/max_count, GRAPH_LOG))
        for j in range(1, max_score + 1):
            if j in score_list[gen-1]:
                pg.draw.line(screen, GREEN, prev, (WIDTH + 20 + j*(SCREEN_WIDTH-WIDTH-40)/max_score, bottom - log(score_list[gen-1][j]*(bottom - top)/max_count, GRAPH_LOG)), 5)
                prev = (WIDTH + 20 + j*(SCREEN_WIDTH-WIDTH-40)/max_score, bottom - log(score_list[gen-1][j]*(bottom - top)/max_count, GRAPH_LOG))
            else:
                pg.draw.line(screen, GREEN, prev, (WIDTH + 20 + j*(SCREEN_WIDTH-WIDTH-40)/max_score, bottom), 5)
                prev = (WIDTH + 20 + j*(SCREEN_WIDTH-WIDTH-40)/max_score, bottom)
        text = font.render(f"{max_count}", True, WHITE)
        screen.blit(text, (WIDTH + 30, 135))
        text = font.render(f"{max_score}", True, WHITE)
        screen.blit(text, (SCREEN_WIDTH - 30 - 10.4*int(np.log10(max_score if max_score != 0 else 1)), bottom + 10))

    if GRAPH_LOG:
        text = font.render("LOG", True, GREEN)
        screen.blit(text, (WIDTH + 30, 160))
    text = font.render("SAVE:", True, WHITE)
    screen.blit(text, (WIDTH + 20, HEIGHT - 40))
    if SAVE_MODE:
        text = font.render("ON", True, GREEN)
        screen.blit(text, (WIDTH + 75, HEIGHT - 40))
    else:
        text = font.render("OFF", True, RED)
        screen.blit(text, (WIDTH + 75, HEIGHT - 40))

species = []
population = reproduce([Player() for _ in range(POPULATION)])
gen, run, time, pause, speed_idx = 1, True, 0, False, 2
speed = [0.1, 1, 100]
best_score = [0]
best_avg_score = [0]
best_fitness = [0]
best_avg_fitness = [0]
score_list = [{} for _ in range(10000)]

coin_list = [np.random.randint(50, WIDTH-50, 2) for _ in range(100)]

while run:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            exit()
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_SPACE:
                pause = not pause
            if event.key == pg.K_LEFT:
                speed_idx = max(0, speed_idx - 1)
            if event.key == pg.K_RIGHT:
                speed_idx = min(2, speed_idx + 1)
            if event.key == pg.K_l:
                GRAPH_LOG = not GRAPH_LOG
            if event.key == pg.K_e:
                EYES_OPEN = not EYES_OPEN
            if event.key == pg.K_h:
                SHOW_TEXT = not SHOW_TEXT
            if event.key == pg.K_g:
                GRAPH_NUM = (GRAPH_NUM + 1) % 3
            if event.key == pg.K_s:
                SAVE_MODE = not SAVE_MODE
            if event.key == pg.K_b:
                BEST_DRAW = not BEST_DRAW
    
    if pause:
        pg.draw.rect(screen, WHITE, (10, 10, 10, 30))
        pg.draw.rect(screen, WHITE, (30, 10, 10, 30))
        pg.display.update()
        continue
    
    screen.fill(BLACK)
    
    if BEST_DRAW == False:
        for i in range(40):
            pg.draw.circle(screen, COLOR_LIST[i//5], coin_list[i], COIN_SIZE)
            pg.draw.circle(screen, WHITE, coin_list[i], COIN_SIZE, 2)
            text = font.render(f"{i+1}", True, WHITE)
            screen.blit(text, (coin_list[i][0] + 15, coin_list[i][1] - 13))
    else:
        best = max(population, key=lambda x: x.genome.avg_fitness)
        coin_pos1 = coin_list[best.genome.score]
        coin_pos2 = coin_list[min(best.genome.score + 1, len(coin_list) - 1)]
        pg.draw.circle(screen, YELLOW, (int(coin_pos1[0]), int(coin_pos1[1])), COIN_SIZE)
        pg.draw.circle(screen, (150, 150, 0), (int(coin_pos1[0]), int(coin_pos1[1])), COIN_SIZE, 2)
        pg.draw.circle(screen, (100, 100, 0), (int(coin_pos2[0]), int(coin_pos2[1])), COIN_SIZE)
        pg.draw.circle(screen, (50, 50, 0), (int(coin_pos2[0]), int(coin_pos2[1])), COIN_SIZE, 2)
        best.draw()

    for player in population:
        player.update()
        if BEST_DRAW == False:
            player.draw()

    pg.draw.rect(screen, BLACK, (WIDTH, 0, SCREEN_WIDTH - WIDTH, HEIGHT))
    pg.draw.rect(screen, WHITE, (WIDTH, 0, 3, HEIGHT))

    draw_stats()

    pg.display.update()

    time += 1

    if time == 1000:
        time = 0
        gen += 1
        best = max(population, key=lambda x: x.genome.fitness).genome
        for i, node in enumerate(best.nodes):
            print(f"Node {i}: {node.id}, {node.bias:.2f}, {node.activation}")
        for i, link in enumerate(best.links):
            print(f"Link {i}: {link.in_id} -> {link.out_id}, {link.weight:.2f}, {link.enabled}, {link.innov}")
        print()
        print([len(s) for s in species])
        print()
        if SAVE_MODE:
            with open(f"best_genome.txt", "w") as f:
                for node in best.nodes:
                    f.write(f"{node.id} {node.bias} {node.activation}\n")
                for link in best.links:
                    f.write(f"{link.in_id} {link.out_id} {link.weight} {link.enabled} {link.innov}\n")
        for player in population:
            score_list[gen-1][player.genome.score] = score_list[gen-1].get(player.genome.score, 0) + 1
            player.genome.avg_fitness = (player.genome.avg_fitness*3 + player.genome.fitness)/4
            player.genome.avg_score = (player.genome.avg_score*3 + player.genome.score)/4
        best_score.append(max(population, key=lambda x: x.genome.score).genome.score)
        best_avg_score.append(sum([player.genome.score for player in population])/len(population))
        best_fitness.append(max(population, key=lambda x: x.genome.fitness).genome.fitness)
        best_avg_fitness.append(sum([player.genome.fitness for player in population])/len(population))
        population = reproduce(population)
        population.sort(key=lambda x: x.genome.fitness)
        push_innov()

    if speed[speed_idx] != 100:
        clock.tick(FPS * speed[speed_idx])
