#include "stdio.h"
#include "stdlib.h"
#include "math.h"

#include "mutation.h"

void mutate(int *bit, int size, float mutate_coeff_rate){

    // mutate_coeff_rate is the amount of mutations over the bit;
    // check if mutate_coeff_rate is < size

    if (mutate_coeff_rate > size){
        printf("Error: mutate_coeff_rate is bigger than size\n");
        exit(1);
    }

    int *mutations = malloc(size * sizeof(int));
    // generate random mutations, that are not at the same position
    for (int i = 0; i < size; i++){
        mutations[i] = rand() % size;
        for (int j = 0; j < i; j++){
            if (mutations[i] == mutations[j]){
                i--;
                break;
            }
        }
    }
    
    // mutate the bit
    for (int i = 0; i < mutate_coeff_rate; i++){
        bit[mutations[i]] = !bit[mutations[i]];
    }

    free(mutations);

}