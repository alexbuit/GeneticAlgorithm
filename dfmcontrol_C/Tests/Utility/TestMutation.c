
#include "stdio.h"
#include "stdlib.h"
#include "math.h"

#include "../Utility/mutation.c"

int main(){

    // test mutation
    int length = 16;

    int *bit = malloc(length * sizeof(int));
    int *bitcopy = malloc(length * sizeof(int));

    for(int i=0; i<length; i++){
        if(rand() % 2 == 0){
            bit[i] = 1;
        } else {
            bit[i] = 0;
        }
    }

    for(int i=0; i<length; i++){
        bitcopy[i] = bit[i];
    }

    int amount_of_mutations = 10;
    int *mutate_coeff_rate = malloc(sizeof(int) * amount_of_mutations);
    for(int i=0; i<amount_of_mutations; i++){
        mutate_coeff_rate[i] = rand() % length;
    }

    printf("Bit before mutation: ");
    for(int i=0; i<length; i++){
        printf("%d ", bit[i]);
    }
    printf("\n");

    int mutations_passed = 0;

    // mutate the bit
    for(int i=0; i<amount_of_mutations; i++){
        mutate(bit, length, mutate_coeff_rate[i]);

        int amount_of_mutations = 0;
        for(int i=0; i<length; i++){
            if(bit[i] != bitcopy[i]){
                amount_of_mutations++;
            }
        }

        printf("expected amount of mutations / amount of mutations: %d / %d \n", mutate_coeff_rate[i], amount_of_mutations);
        if (mutate_coeff_rate[i] == amount_of_mutations){
            mutations_passed++;
        }

        printf("Bit after mutation %d: ", i);
        for(int i=0; i<length; i++){
            printf("%d ", bit[i]);
        }
        printf("\n");

        for(int i=0; i<length; i++){
            bit[i] = bitcopy[i];
        }
    }

    
    if (mutations_passed == amount_of_mutations){
        printf("Mutation passed\n");
    } else {
        printf("Mutation failed\n");
    }
    
    free(bit);
    free(bitcopy);
    free(mutate_coeff_rate);
}