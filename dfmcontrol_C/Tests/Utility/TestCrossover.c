#include "stdio.h"
#include "stdlib.h"
#include "math.h"

#include "../../Utility/crossover.c"

int main(){

    srand(__TIME__);

    int length = 8 ;
    
    // test single point crossover
    int* parent1 = malloc(length * sizeof(int));
    int* parent2 = malloc(length * sizeof(int));

    for(int i=0; i<length; i++){
        parent1[i] = random_int();
        parent2[i] = random_int();
    }

    int* child1 = malloc(length * sizeof(int));
    int* child2 = malloc(length * sizeof(int));
    printf("int size %d\n", sizeof(int));
    single_point_crossover(parent1, parent2, child1, child2, length);

    // Check that the children are different
    int* same;
    same = malloc(length * sizeof(int));
    for(int i=0; i<length; i++){
        if(child1[i] == child2[i]){
            same[i] = 1;
        } 
        else if (child1[i] == parent1[i] && child2[i] == parent2[i]) {
            same[i] = 1;
        }
        else {
            same[i] = 0;
        }
    }

    int same_count = 0;
    for(int i=0; i<length; i++){
        if(same[i] == 1){
            same_count++;
        }
    }

    printf("Parent 1: ");
    for(int i=0; i<length; i++){
        printf("%#010x ", parent1[i]);
    }
    printf("\nChild 1:  ");
    for(int i=0; i<length; i++){
        printf("%#010x ", child1[i]);
    }
    printf("\nParent 2: ");
    for(int i=0; i<length; i++){
        printf("%#010x ", parent2[i]);
    }
    printf("\nChild 2:  ");
    for(int i=0; i<length; i++){
        printf("%#010x ", child2[i]);
    }

    printf("\n");

    if(same_count == length){
        printf("Single point crossover failed\n");
    } else {
        printf("Single point crossover passed\n");
    }

    free(parent1);
    free(parent2);
    free(child1);
    free(child2);
    free(same);

    // test two point crossover
    parent1 = malloc(length * sizeof(int));
    parent2 = malloc(length * sizeof(int));

    for(int i=0; i<length; i++){
        parent1[i] = random_int();
        parent2[i] = random_int();
    }

    child1 = malloc(length * sizeof(int));
    child2 = malloc(length * sizeof(int));

    two_point_crossover(parent1, parent2, child1, child2, length );

    // Check that the children are different
    same = malloc(length * sizeof(int));
    for(int i=0; i<length; i++){
        if(child1[i] == child2[i]){
            same[i] = 1;
        } 
        
        else if (child1[i] == parent1[i] && child2[i] == parent2[i]) {
            same[i] = 1;
        }

        else {
            same[i] = 0;
        }
    }

    same_count = 0;
    for(int i=0; i<length; i++){
        if(same[i] == 1){
            same_count++;
        }
    }

    printf("Parent 1: ");
    for(int i=0; i<length; i++){
        printf("%#010x ", parent1[i]);
    }
    printf("\nChild 1:  ");
    for(int i=0; i<length; i++){
        printf("%#010x ", child1[i]);
    }
    printf("\nParent 2: ");
    for(int i=0; i<length; i++){
        printf("%#010x ", parent2[i]);
    }
    printf("\nChild 2:  ");
    for(int i=0; i<length; i++){
        printf("%#010x ", child2[i]);
    }
    printf("\n");
    if(same_count == length){
        printf("Two point crossover failed\n");
    } else {
        printf("Two point crossover passed\n");
    }

    free(parent1);
    free(parent2);
    free(child1);
    free(child2);
    free(same);

    // test uniform crossover
    parent1 = malloc(length * sizeof(int));
    parent2 = malloc(length * sizeof(int));

    for(int i=0; i<length; i++){
        parent1[i] = random_int();
        parent2[i] = random_int();
    }

    child1 = malloc(length * sizeof(int));
    child2 = malloc(length * sizeof(int));

    uniform_crossover(parent1, parent2, child1, child2, length );

    // Check that the children are different
    same = malloc(length * sizeof(int));
    for(int i=0; i<length; i++){
        if(child1[i] == child2[i]){
            same[i] = 1;
        } 
        else if (child1[i] == parent1[i] && child2[i] == parent2[i])
        {
            same[i] = 1;
        } 
        else {
            same[i] = 0;
        }
    }

    same_count = 0;
    for(int i=0; i<length; i++){
        if(same[i] == 1){
            same_count++;
        }
    }

    printf("Parent 1: ");
    for(int i=0; i<length; i++){
        printf("%#010x ", parent1[i]);
    }
    printf("\nChild 1: ");
    for(int i=0; i<length; i++){
        printf("%#010x ", child1[i]);
    }
    printf("\nParent 2: ");
    for(int i=0; i<length; i++){
        printf("%#010x ", parent2[i]);
    }
    printf("\nChild 2: ");
    for(int i=0; i<length; i++){
        printf("%#010x ", child2[i]);
    }
    printf("\n");
    if(same_count == length){
        printf("Uniform crossover failed\n");
    } else {
        printf("Uniform crossover passed\n");
    }


    // test uniform crossover
    parent1 = malloc(length * sizeof(int));
    parent2 = malloc(length * sizeof(int));

    for(int i=0; i<length; i++){
        parent1[i] = random_int();
        parent2[i] = random_int();
    }

    child1 = malloc(length * sizeof(int));
    child2 = malloc(length * sizeof(int));

    complete_crossover(parent1, parent2, child1, child2, length );

    // Check that the children are different
    same = malloc(length * sizeof(int));
    for(int i=0; i<length; i++){
        if(child1[i] == child2[i]){
            same[i] = 1;
        } 
        else if (child1[i] == parent1[i] && child2[i] == parent2[i])
        {
            same[i] = 1;
        } 
        else {
            same[i] = 0;
        }
    }

    same_count = 0;
    for(int i=0; i<length; i++){
        if(same[i] == 1){
            same_count++;
        }
    }

    printf("Parent 1: ");
    for(int i=0; i<length; i++){
        printf("%#010x ", parent1[i]);
    }
    printf("\nChild 1: ");
    for(int i=0; i<length; i++){
        printf("%#010x ", child1[i]);
    }
    printf("\nParent 2: ");
    for(int i=0; i<length; i++){
        printf("%#010x ", parent2[i]);
    }
    printf("\nChild 2: ");
    for(int i=0; i<length; i++){
        printf("%#010x ", child2[i]);
    }
    printf("\n");
    if(same_count == length){
        printf("Complete crossover failed\n");
    } else {
        printf("Complete crossover passed\n");
    }

    free(parent1);
    free(parent2);
    free(child1);
    free(child2);
    free(same);


}