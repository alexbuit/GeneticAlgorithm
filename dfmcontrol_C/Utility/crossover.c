#include "stdio.h"
#include "stdlib.h"
#include "math.h"

#include "crossover.h"

// Path: Utility/crossover.c

void single_point_crossover(int *parent1, int *parent2, int *child1, int *child2, int size){
    // parent1 and parent2 are the parents to be crossed over and child1 and child2 are the children to be created all of size size
    // The function should fill child1 and child2 with the crossed over values

    // find a random point to cross over
    int point = rand() % size;

    // copy the first part of parent1 to child1 and the first part of parent2 to child2
    for(int i; i<point; i++){
        child1[i] = parent1[i];
        child2[i] = parent2[i];
    }
    // for the other half, copy the values from the other parent
    for(int i; i<size; i++){
        child1[i] = parent2[i];
        child2[i] = parent1[i];
    }
}

void two_point_crossover(int *parent1, int *parent2, int *child1, int *child2, int size){
    // parent1 and parent2 are the parents to be crossed over and child1 and child2 are the children to be created all of size size
    // point1 and point2 are the points to cross over at
    // The function should fill child1 and child2 with the crossed over values

    // find two random points to cross over
    int point1 = rand() % size - 1;
    int point2 = rand() % (size - point1) + point1;

    // make sure point1 is less than point2 and less than size
    if(point1 > point2 || point2 > size){
        int temp = point1;
        point1 = point2;
        point2 = temp;

        if(point2 > size){
            point2 = size - 1;
        }
    }

    // copy the first part of parent1 to child1 and the first part of parent2 to child2
    for(int i = 0; i<point1; i++){
        child1[i] = parent2[i];
        child2[i] = parent1[i];
    }
    // for the other half, copy the values from the other parent
    for(int i = point1; i<point2; i++){
        child1[i] = parent1[i];
        child2[i] = parent2[i];
    }
    // for the last part, copy the values from the other parent
    for(int i = point2; i<size; i++){
        child1[i] = parent2[i];
        child2[i] = parent1[i];
    }
}

void uniform_crossover(int *parent1, int *parent2, int *child1, int *child2, int size){
    // parent1 and parent2 are the parents to be crossed over and child1 and child2 are the children to be created all of size size
    // prob is the probability of a value being copied from the first parent
    // The function should fill child1 and child2 with the crossed over values

    // for each value, copy from the first parent with probability prob
    for(int i; i<size; i++){
        if(rand() % 100 < 50){
            child1[i] = parent1[i];
            child2[i] = parent2[i];
        }
        else{
            child1[i] = parent2[i];
            child2[i] = parent1[i];
        }
    }
}

