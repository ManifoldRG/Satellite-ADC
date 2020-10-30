import poliastro as pol
from poliastro.twobody import Orbit
from poliastro.maneuver import Maneuver
from poliastro.bodies import Earth, Mars, Sun
from poliastro.plotting import OrbitPlotter2D
from poliastro.plotting import OrbitPlotter3D
import numpy as np
import scipy as sci
from astropy import units as u
import random
import operator

#The spacecraft class is declared here

class Spacecraft:

  def __init__(self, position, velocity):
    self.r_start =  position*u.km
    self.v_start =  velocity*u.km/u.s

    self.orbit = Orbit.from_vectors(Earth, self.r_start, self.v_start)

  def maneuver(self, dv):
    man = Maneuver.impulse(dv*u.km/u.s)
    self.orbit = self.orbit.apply_maneuver(man)

#These are the initial r and v for each spacecraft.
tests = [Spacecraft((-6045, -3490, 2500), (3.457, -6.618, 3.533)), Spacecraft((-5045, -4490, 2500), (-4.457, 5.618, 3.533)), Spacecraft((8000, 0, 0), (0, 8, 0)), Spacecraft((-5045, -4490, 2500), (4.457, -5.618, -3.533)), Spacecraft((-5045, -4490, 2500), (4.457, -5.618, 3.533))]
#This function generates a fitness score based on the eccentricity, inclination, and number of delta v maneuvers.
#Each parameter can be weighted according to how important it is.
def generate_fitness_score(w1,w2,w3,delt_v_size,test_orb,seq):
  num_vs = 0
  for man in seq:
    if man == '1':
      test_orb = test_orb.apply_maneuver(Maneuver.impulse((delt_v_size,0,0)*u.m/u.s))
      num_vs += 1
    if man == '2':
      test_orb = test_orb.apply_maneuver(Maneuver.impulse((0,delt_v_size,0)*u.m/u.s))
      num_vs += 1
    if man == '3':
      test_orb = test_orb.apply_maneuver(Maneuver.impulse((0,0,delt_v_size)*u.m/u.s))
      num_vs += 1
    if man == '4':
      test_orb = test_orb.apply_maneuver(Maneuver.impulse((-delt_v_size,0,0)*u.m/u.s))
      num_vs += 1
    if man == '5':
      test_orb = test_orb.apply_maneuver(Maneuver.impulse((0,-delt_v_size,0)*u.m/u.s))
      num_vs += 1
    if man == '6':
      test_orb = test_orb.apply_maneuver(Maneuver.impulse((0,0,-delt_v_size)*u.m/u.s))
      num_vs += 1
    test_orb = test_orb.propagate(20*u.min)
  return (np.abs(float(test_orb.inc/u.rad - 1.57)))*w1 + float(test_orb.ecc)*w2 + float(num_vs)*w3

#This function generates random maneuver strings.
#They are created in this encoded way for simplicity and efficiency
#A 0 means idle, a 1 means an x impulse, a 2 means a y impulse, and a 3 means a z impulse.
#4, 5 and 6 are impulses in the negative x, y and z directions respectively.
#The random generation can be seeded with a maneuver string of choice, so that
#the random population will be similar to the seed.
def generate_pop(max_delt_v, delt_v_size, pop_size, test_orb, seed = []):
  gen_pop = []
  for j in range(0,pop_size):
    seq = ""
    for i in range(0,max_delt_v):
      seed_chance = np.random.random()
      if len(seed)>0:
        if seed_chance <= 0.6:
          seq += seed[i]
        else:
          seq += str(np.random.randint(0,6))
      else:
        seq += str(np.random.randint(0,6))
    seq_and_fit = (seq, generate_fitness_score(1,1,0.05,delt_v_size,test_orb,seq))
    gen_pop.append(seq_and_fit)
  return gen_pop

#This function mates two maneuver strings together.
#Maneuvers that are the same between strings are kept in the offspring,
#ones that are different are randomly selected in the offspring.
#There is a chance of mutation, where even if the parents agree, the child will have
#a random maneuver in that section.

def mate_seq(fath,moth,delt_v_size,test_orb):
  chil = ""
  num_vs = 0
  mut = random.randint(0,10)
  for i in range(0,len(fath)):
    if fath[i] == moth[i] and mut != 1:
      chil += fath[i]
    else:
      chil += str(np.random.randint(0,6))
  chil_and_fit = (chil, generate_fitness_score(1,1,0.05,delt_v_size,test_orb,chil))
  return chil_and_fit


# In this cell, sequences will evolve. Initially, a set number of generations will be used.
# The fittest maneuver string after 200 generations will be stored as the best plan.

fittest = []

for test in tests:
  #Two populations are generated, and then interbred every 5 generations to break each other out of genetic
  #stagnancy, where all children will start looking the same.
  population_a = generate_pop(30, 500, 100, test_orb=test.orbit)
  population_b = generate_pop(30, 500, 100, test_orb=test.orbit)

  for g in range(0, 200):
    population_a.sort(key=operator.itemgetter(1))
    population_b.sort(key=operator.itemgetter(1))

    #The fittest individual survives as long as they remain the fittest.
    #This way, fit individuals are not lost and each generation only gets better.
    sorted_pop_a = population_a[:10]
    sorted_pop_b = population_b[:10]
    population_a = population_a[:1]
    population_b = population_b[:1]

    #Another measure to break stagnancy, if both populations stagnate, an immigrant population is brought in
    #to mate with population a, breaking the genetic stagnancy and introducing diversity again.
    immigrants = generate_pop(30, 500, 10, test_orb=test.orbit, seed=sorted_pop_a[0][0])

    for i in range(0, len(sorted_pop_a)):
      for j in range(0, 2):
        if g % 5 == 0:
          for k in range(i, len(sorted_pop_a)):
            population_a.append(mate_seq(sorted_pop_b[i][0], sorted_pop_a[k][0], test_orb=test.orbit, delt_v_size=500))
            population_b.append(mate_seq(sorted_pop_a[i][0], sorted_pop_b[k][0], test_orb=test.orbit, delt_v_size=500))
        elif sorted_pop_a[0][1] == sorted_pop_b[0][1]:
          for k in range(i, len(sorted_pop_a)):
            population_a.append(mate_seq(immigrants[i][0], sorted_pop_a[k][0], test_orb=test.orbit, delt_v_size=500))
            population_b.append(mate_seq(sorted_pop_b[i][0], sorted_pop_b[k][0], test_orb=test.orbit, delt_v_size=500))
        else:
          for k in range(i, len(sorted_pop_a)):
            population_a.append(mate_seq(sorted_pop_a[i][0], sorted_pop_a[k][0], test_orb=test.orbit, delt_v_size=500))
            population_b.append(mate_seq(sorted_pop_b[i][0], sorted_pop_b[k][0], test_orb=test.orbit, delt_v_size=500))

  population_a.sort(key=operator.itemgetter(1))
  population_b.sort(key=operator.itemgetter(1))
  if population_a[0][1] < population_b[0][1]:
    fittest.append(population_a[0])
    print(population_a[0])
  else:
    fittest.append(population_b[0])
    print(population_b[0])
#The fittest for each spacecraft are output here.
print(fittest)