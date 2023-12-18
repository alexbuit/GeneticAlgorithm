#include "stdio.h"
#include "stdlib.h"
#include "math.h"

#include "mutation.h"

void mutate32(int* individual, int genes, int mutate_coeff_rate){

    /*
    
    This function mutates a bitarray by flipping a random bit.

    :param bit: bitarray to mutate
    :type bit: int*

    :param size: size of the bitarray
    :type size: int

    :param mutate_coeff_rate: amount of mutations over the bitarray
    :type mutate_coeff_rate: int

    :param chaos_coeff: the signifigance of the bits impacted by the mutation (1 to 32) (1 for least significant bit, 32 for most significant bit)
    :type chaos_coeff: int

    :param allow_sign_flip: whether or not to allow the sign to flip, 1 for yes, 0 for no
    :type allow_sign_flip: int

    */

    // mutate_coeff_rate is the amount of mutations over the bit;
    // check if mutate_coeff_rate is < size

    if (mutate_coeff_rate > genes ){
        printf("Error: mutate_coeff_rate is bigger than possible flips\n");
        exit(1);
    }

    int* mutations = malloc(genes * sizeof(int));
    // generate random mutations, that are not at the same position
    for (int i = 0; i < genes; i++){
        mutations[i] = 0;
    }

    int mutation;
    int mutation_bit = 0;
    for (int i = 0; i < mutate_coeff_rate; i++){
        mutation = rand() % genes;
        mutation_bit = 1 << (rand() % sizeof(int)*8);

        if(mutations[mutation] & mutation_bit){
            i--;
            continue;
        }

        mutations[mutation] = mutations[mutation]  | mutation_bit;
    }

    for (int i = 0; i < genes; i++){
        if(mutations[i] != 0){
            individual[i] = individual[i] ^ mutations[i];
        }
    }

    free(mutations);

}

void mutate(int* individual, int genes, int mutate_coeff_rate){

    /*
    
    This function mutates a bitarray by flipping a random bit.

    :param bit: bitarray to mutate
    :type bit: int*

    :param size: size of the bitarray
    :type size: int

    :param mutate_coeff_rate: amount of mutations over the bitarray
    :type mutate_coeff_rate: int
    */

    // mutate_coeff_rate is the amount of mutations over the bit;
    // check if mutate_coeff_rate is < size

    if (mutate_coeff_rate > genes){
        printf("Error: mutate_coeff_rate is bigger than size\n");
        exit(1);
    }

    int *mutations = malloc(genes * sizeof(int));
    // generate random mutations, that are not at the same position
    for (int i = 0; i < genes; i++){
        mutations[i] = rand() % genes;
        for (int j = 0; j < i; j++){
            if (mutations[i] == mutations[j]){
                i--;
                break;
            }
        }
    }
    
    // mutate the bit
    for (int i = 0; i < mutate_coeff_rate; i++){
        individual[mutations[i]] = !individual[mutations[i]];
    }

    free(mutations);

}