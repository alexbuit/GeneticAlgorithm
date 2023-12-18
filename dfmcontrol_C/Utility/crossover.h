
#ifndef _Selection_
#define _Selection_

// 32 bit crossover functions
void single_point_crossover32(int *parent1, int *parent2, int *child1, int *child2, int genes);
void two_point_crossover32(int *parent1, int *parent2, int *child1, int *child2, int genes);
void uniform_crossover32(int *parent1, int *parent2, int *child1, int *child2, int genes);
void complete_crossover32(int *parent1, int *parent2, int *child1, int *child2, int genes);

// variable bit crossover functions
void single_point_crossover(int *parent1, int *parent2, int *child1, int *child2, int genes);
void two_point_crossover(int *parent1, int *parent2, int *child1, int *child2, int genes);
void uniform_crossover(int *parent1, int *parent2, int *child1, int *child2, int genes);
void complete_crossover(int *parent1, int *parent2, int *child1, int *child2, int genes);
