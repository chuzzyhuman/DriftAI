import pygame as pg
import numpy as np

WIDTH, HEIGHT = 600, 600
SCREEN_WIDTH = 1200
FPS = 60
PLAYER_SIZE, COIN_SIZE = 20, 10
PLAYER_X, PLAYER_Y = 300, 300
MAX_SPEED, ACCELERATION = 15, 1

SHOW_TEXT, GRAPH_LOG, GRAPH_NUM = True, False, 0
SAVE_MODE = False
BEST_DRAW = False

BLACK, WHITE, RED, GREEN, BLUE, YELLOW = (0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)
COLOR_LIST = [(i, 255, 0) for i in range(0, 256, 51)] + [(255, i, 0) for i in range(255, -1, -51)] + [(i, 0, 0) for i in range(255, -1, -51)] + [(0, 0, 0) for i in range(100)]

INPUTS, OUTPUTS = 6, 2
POPULATION = 1000

c1, c2, c3 = 1, 1, 0.4
MAX_WEIGHT = 5
DELTA_THRESHOLD = 0.4
HIDDEN_ACTIVATION, OUTPUT_ACTIVATION = 2, 2

pg.init()
screen = pg.display.set_mode((SCREEN_WIDTH, HEIGHT))
pg.display.set_caption("driftAI")
font = pg.font.Font("font.ttf", 24)
clock = pg.time.Clock()

def squash(x, n):
    if n == 0:
        return x
    if n == 1:
        return 1/(1 + np.exp(-x))
    if n == 2:
        return np.tanh(x)
    if n == 3:
        return np.maximum(0, x)
    
def log(x, n):
    if n:
        return np.log10(1+x)
    return x

class Genome:
    def __init__(self):
        self.nodes = [INPUTS, 3, 3, OUTPUTS]
        self.weights = np.zeros((len(self.nodes)-1, max(self.nodes)+1, max(self.nodes)+1))
        self.values = np.zeros((len(self.nodes), max(self.nodes)+1))
        self.fitness = 0
        self.avg_fitness = 0
        self.score = 0
        self.avg_score = 0
        self.species = 0
        self.reload()

    def clone(self):
        g = Genome()
        g.nodes = self.nodes.copy()
        g.weights = np.array([w.copy() for w in self.weights])
        g.values = np.zeros(self.values.shape)
        g.fitness = self.fitness
        g.avg_fitness = self.avg_fitness
        g.score = self.score
        g.avg_score = self.avg_score
        g.species = self.species
        return g

    def reload(self):
        self.values = np.zeros(self.values.shape)

    def distance(self, other):
        return np.sum((self.weights - other.weights) ** 2) / self.weights.size

    def feed_forward(self, inputs):
        self.values[0, :self.nodes[0]] = inputs
        self.values[0, self.nodes[0]] = 1
        for i in range(1, self.values.shape[0]-1):
            self.values[i, :self.nodes[i]] = squash(self.weights[i-1, :self.nodes[i-1]+1, :self.nodes[i]].T @ self.values[i-1, :self.nodes[i-1]+1], HIDDEN_ACTIVATION)
            self.values[i, self.nodes[i]] = 1
        self.values[-1, :self.nodes[-1]] = squash(self.weights[-1, :self.nodes[-2]+1, :self.nodes[-1]].T @ self.values[-2, :self.nodes[-2]+1], OUTPUT_ACTIVATION)
        return self.values[-1, :self.nodes[-1]]

    def mutate(self):
        r = np.random.rand()
        if r < 1/3:
            self.weights[np.random.randint(self.weights.shape[0]), np.random.randint(self.weights.shape[1]), np.random.randint(self.weights.shape[2])] = np.random.randn()
            self.weights = np.clip(self.weights, -MAX_WEIGHT, MAX_WEIGHT)
        else:
            self.weights[np.random.randint(self.weights.shape[0]), np.random.randint(self.weights.shape[1]), np.random.randint(self.weights.shape[2])] += np.random.randn()
            self.weights = np.clip(self.weights, -MAX_WEIGHT, MAX_WEIGHT)

    def crossover(self, other):
        child = self.clone()
        for i in range(self.weights.shape[0]):
            for j in range(self.weights.shape[1]):
                for k in range(self.weights.shape[2]):
                    if np.random.rand() < 0.5:
                        child.weights[i, j, k] = other.weights[i, j, k]
        return child
    
class Player:
    def __init__(self, genome=None):
        self.x = PLAYER_X + np.random.uniform(-10, 10)
        self.y = PLAYER_Y + np.random.uniform(-10, 10)
        self.vx = 0
        self.vy = 0
        self.ax = 0
        self.ay = 0
        self.penalty = 0
        self.increment = 0
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
        self.increment = 0
        self.genome.fitness = 0
        self.genome.score = 0

    def update(self):
        coin_pos1 = coin_list[player.genome.score]
        coin_pos2 = coin_list[min(player.genome.score + 1, len(coin_list) - 1)]
        if np.sqrt((player.x - coin_pos1[0])**2 + (player.y - coin_pos1[1])**2) < PLAYER_SIZE + COIN_SIZE:
            player.genome.score += 1

        prev_dist = np.sqrt((player.x - coin_pos1[0])**2 + (player.y - coin_pos1[1])**2)
        diff1 = np.array([(coin_pos1[0] - self.x)/WIDTH, (coin_pos1[1] - self.y)/HEIGHT])
        diff2 = np.array([(coin_pos2[0] - self.x)/WIDTH, (coin_pos2[1] - self.y)/HEIGHT])
        gap = 0
        diff1 += np.sign(diff1) * gap
        diff2 += np.sign(diff2) * gap
        self.input = [diff1[0], diff1[1], diff2[0], diff2[1], self.vx/MAX_SPEED, self.vy/MAX_SPEED]
        outputs = self.genome.feed_forward(self.input)
        self.ax, self.ay = outputs[0]*ACCELERATION, outputs[1]*ACCELERATION

        self.vx += self.ax
        self.vy += self.ay
        self.vx = np.clip(self.vx, -MAX_SPEED, MAX_SPEED)
        self.vy = np.clip(self.vy, -MAX_SPEED, MAX_SPEED)
        self.x += self.vx
        self.y += self.vy

        self.genome.fitness = max(self.genome.score + 1/(1 + (np.sqrt((self.x - coin_pos1[0])**2 + (self.y - coin_pos1[1])**2))/WIDTH) - self.penalty, 0)
        if prev_dist < np.sqrt((player.x - coin_pos1[0])**2 + (player.y - coin_pos1[1])**2):
            self.penalty += self.increment*min(self.genome.score, 10)/10
            self.increment = min(self.increment + 0.005, 0.1)
        else:
            self.increment = 0

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
            child.avg_fitness = s[j].avg_fitness
            child.avg_score = s[j].avg_score
            new_population.append(child)
    for s in species:
        for i in range(min(len(s), 30)):
            parent1 = np.random.choice(s[:min(len(s)//4+1, 5)])
            parent2 = np.random.choice(s[:min(len(s)//4+1, 5)])
            child = parent1.crossover(parent2)
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
    text = font.render(f"Species: {[len(s) for s in species]}", True, WHITE)
    screen.blit(text, (WIDTH + 20, 70))
    
    top, bottom = 110, 270
    w, h, g1, g2 = 90, 295, 80, 36
    min_value, max_value = -1, 1
    best = best_player.genome
    
    if SHOW_TEXT:
        for i, value in enumerate(best_player.input):
            text = font.render(f"{value:.2f}", True, WHITE)
            screen.blit(text, (WIDTH + 20, h - 12 + i*g2))
        for i, value in enumerate(best_player.genome.values[-1, :best_player.genome.nodes[-1]]):
            text = font.render(f"{value:.2f}", True, WHITE)
            screen.blit(text, (WIDTH + 100 + (best.values.shape[0]-1)*g1 + 20, h - 12 + i*g2))
    
    for i in range(best.weights.shape[0]):
        for j in range(best.nodes[i]+1):
            for k in range(best.nodes[i+1]):
                if best.weights[i, j, k] != 0:
                    pos1 = (WIDTH + w + i*g1, h + j*g2)
                    pos2 = (WIDTH + w + (i+1)*g1, h + k*g2)
                    pg.draw.line(screen, RED if best.weights[i, j, k] > 0 else BLUE if best.weights[i, j, k] < 0 else (50, 50, 50), pos1, pos2, abs(int(best.weights[i, j, k])) + 2)
    
    for i in range(best.values.shape[0]):
        for j in range(best.nodes[i] + 1 if i != best.values.shape[0] - 1 else best.nodes[i]):
            b = int(255 * (best.values[i, j] - min_value) / (max_value - min_value))
            b = max(0, min(255, b))
            pg.draw.circle(screen, (b, b, b), (WIDTH + w + i*g1, h + j*g2), 16)
            pg.draw.circle(screen, WHITE, (WIDTH + w + i*g1, h + j*g2), 16, 3)
    
    pg.draw.line(screen, WHITE, (WIDTH + 20, bottom), (SCREEN_WIDTH - 20, bottom), 5)
    pg.draw.line(screen, WHITE, (WIDTH + 20, top), (WIDTH + 20, bottom), 5)
    
    if GRAPH_NUM == 0:
        max_score = max(max(best_score + best_avg_score), 1)
        for i in range(len(best_score) - 1):
            pg.draw.line(screen, RED, (WIDTH + 20 + i*(SCREEN_WIDTH-WIDTH-40)/(len(best_score) - 1), bottom - log(best_score[i], GRAPH_LOG)*(bottom - top)/log(max_score, GRAPH_LOG)), (WIDTH + 20 + (i + 1)*(SCREEN_WIDTH-WIDTH-40)/(len(best_score) - 1), bottom - log(best_score[i + 1], GRAPH_LOG)*(bottom - top)/log(max_score, GRAPH_LOG)), 5)
        for i in range(len(best_avg_score) - 1):
            pg.draw.line(screen, BLUE, (WIDTH + 20 + i*(SCREEN_WIDTH-WIDTH-40)/(len(best_avg_score) - 1), bottom - log(best_avg_score[i], GRAPH_LOG)*(bottom - top)/log(max_score, GRAPH_LOG)), (WIDTH + 20 + (i + 1)*(SCREEN_WIDTH-WIDTH-40)/(len(best_avg_score) - 1), bottom - log(best_avg_score[i + 1], GRAPH_LOG)*(bottom - top)/log(max_score, GRAPH_LOG)), 5)
        text = font.render(f"{max_score}", True, WHITE)
        screen.blit(text, (WIDTH + 30, 105))
    elif GRAPH_NUM == 1:
        max_fitness = max(best_fitness + best_avg_fitness)
        for i in range(len(best_fitness) - 1):
            pg.draw.line(screen, (255, 255, 0), (WIDTH + 20 + i*(SCREEN_WIDTH-WIDTH-40)/(len(best_fitness) - 1), bottom - log(best_fitness[i], GRAPH_LOG)*(bottom - top)/log(max_fitness, GRAPH_LOG)), (WIDTH + 20 + (i + 1)*(SCREEN_WIDTH-WIDTH-40)/(len(best_fitness) - 1), bottom - log(best_fitness[i + 1], GRAPH_LOG)*(bottom - top)/log(max_fitness, GRAPH_LOG)), 5)
        for i in range(len(best_avg_fitness) - 1):
            pg.draw.line(screen, (255, 0, 255), (WIDTH + 20 + i*(SCREEN_WIDTH-WIDTH-40)/(len(best_avg_fitness) - 1), bottom - log(best_avg_fitness[i], GRAPH_LOG)*(bottom - top)/log(max_fitness, GRAPH_LOG)), (WIDTH + 20 + (i + 1)*(SCREEN_WIDTH-WIDTH-40)/(len(best_avg_fitness) - 1), bottom - log(best_avg_fitness[i + 1], GRAPH_LOG)*(bottom - top)/log(max_fitness, GRAPH_LOG)), 5)
        text = font.render(f"{max_fitness:.2f}", True, WHITE)
        screen.blit(text, (WIDTH + 30, 105))
    elif GRAPH_NUM == 2:
        max_score = 1 if len(score_list[gen-1]) == 0 else max(max(score_list[gen-1].keys()), 1) if gen == 1 else max(max(score_list[gen-1].keys()), max(score_list[gen-2].keys()), 1)
        max_count = 0 if len(score_list[gen-1]) == 0 else max(score_list[gen-1].values()) if gen == 1 else max(max(score_list[gen-1].values()), max(score_list[gen-2].values()))
        for j in range(0, max_score + 1):
            if j in score_list[gen-1]:
                pg.draw.line(screen, WHITE, (WIDTH + 20 + j*(SCREEN_WIDTH-WIDTH-40)/max_score, bottom), (WIDTH + 20 + j*(SCREEN_WIDTH-WIDTH-40)/max_score, bottom - log(score_list[gen-1][j], GRAPH_LOG)*(bottom - top)/log(max_count, GRAPH_LOG)), 7)
                if gen > 1 and j in score_list[gen-2]:
                    if score_list[gen-1][j] > score_list[gen-2][j]:
                        pg.draw.line(screen, GREEN, (WIDTH + 20 + j*(SCREEN_WIDTH-WIDTH-40)/max_score, bottom - log(score_list[gen-1][j], GRAPH_LOG)*(bottom - top)/log(max_count, GRAPH_LOG)), (WIDTH + 20 + j*(SCREEN_WIDTH-WIDTH-40)/max_score, bottom - log(score_list[gen-2][j], GRAPH_LOG)*(bottom - top)/log(max_count, GRAPH_LOG)), 7)
                    else:
                        pg.draw.line(screen, RED, (WIDTH + 20 + j*(SCREEN_WIDTH-WIDTH-40)/max_score, bottom - log(score_list[gen-1][j], GRAPH_LOG)*(bottom - top)/log(max_count, GRAPH_LOG)), (WIDTH + 20 + j*(SCREEN_WIDTH-WIDTH-40)/max_score, bottom - log(score_list[gen-2][j], GRAPH_LOG)*(bottom - top)/log(max_count, GRAPH_LOG)), 7)
                elif gen > 1:
                    pg.draw.line(screen, GREEN, (WIDTH + 20 + j*(SCREEN_WIDTH-WIDTH-40)/max_score, bottom - log(score_list[gen-1][j], GRAPH_LOG)*(bottom - top)/log(max_count, GRAPH_LOG)), (WIDTH + 20 + j*(SCREEN_WIDTH-WIDTH-40)/max_score, bottom), 7)
            elif gen > 1 and j in score_list[gen-2]:
                pg.draw.line(screen, RED, (WIDTH + 20 + j*(SCREEN_WIDTH-WIDTH-40)/max_score, bottom - log(score_list[gen-2][j], GRAPH_LOG)*(bottom - top)/log(max_count, GRAPH_LOG)), (WIDTH + 20 + j*(SCREEN_WIDTH-WIDTH-40)/max_score, bottom), 7)
        text = font.render(f"{max_count}", True, WHITE)
        screen.blit(text, (WIDTH + 30, 105))
        text = font.render(f"{max_score}", True, WHITE)
        screen.blit(text, (SCREEN_WIDTH - 30 - 10.4*int(np.log10(max_score if max_score != 0 else 1)), bottom + 10))

    if GRAPH_LOG:
        text = font.render("LOG", True, GREEN)
        screen.blit(text, (WIDTH + 30, 130))
    text = font.render("SAVE:", True, WHITE)
    screen.blit(text, (WIDTH + 20, HEIGHT - 33))
    if SAVE_MODE:
        text = font.render("ON", True, GREEN)
        screen.blit(text, (WIDTH + 75, HEIGHT - 33))
    else:
        text = font.render("OFF", True, RED)
        screen.blit(text, (WIDTH + 75, HEIGHT - 33))

species = []
population = reproduce([Player() for _ in range(POPULATION)])
gen, run, time, pause, speed_idx = 1, True, 0, False, 2
speed = [0.1, 1, 40]
best_player = max(population, key=lambda x: x.genome.avg_fitness)
best_score = [0]
best_avg_score = [0]
best_fitness = [0]
best_avg_fitness = [0]
score_list = [{} for _ in range(10000)]
score_list[0][0] = POPULATION

coin_list = [(i, j) for i in range(50, WIDTH-50, 50) for j in range(50, HEIGHT-50, 50)]
np.random.shuffle(coin_list)

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
            pg.draw.circle(screen, COLOR_LIST[i//3], coin_list[i], COIN_SIZE)
            pg.draw.circle(screen, WHITE, coin_list[i], COIN_SIZE, 2)
            text = font.render(f"{i+1}", True, WHITE)
            screen.blit(text, (coin_list[i][0] + 15, coin_list[i][1] - 13))
    else:
        coin_pos1 = coin_list[best_player.genome.score]
        coin_pos2 = coin_list[min(best_player.genome.score + 1, len(coin_list) - 1)]
        pg.draw.circle(screen, YELLOW, (int(coin_pos1[0]), int(coin_pos1[1])), COIN_SIZE)
        pg.draw.circle(screen, (150, 150, 0), (int(coin_pos1[0]), int(coin_pos1[1])), COIN_SIZE, 2)
        pg.draw.circle(screen, (80, 80, 0), (int(coin_pos2[0]), int(coin_pos2[1])), COIN_SIZE)
        pg.draw.circle(screen, (40, 40, 0), (int(coin_pos2[0]), int(coin_pos2[1])), COIN_SIZE, 2)
        best_player.draw()

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
        if SAVE_MODE:
            with open(f"best_genome.txt", "w") as f:
                for i in range(best.weights.shape[0]):
                    for j in range(best.weights.shape[1]):
                        for k in range(best.weights.shape[2]):
                            f.write(f"{best.weights[i, j, k]}\n")
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
        for player in population:
            player.genome.reload()
            player.reset()
        best_player = max(population, key=lambda x: x.genome.avg_fitness)

    if speed[speed_idx] != 40:
        clock.tick(FPS * speed[speed_idx])