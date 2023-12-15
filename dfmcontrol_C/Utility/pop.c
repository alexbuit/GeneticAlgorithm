#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#define PI   3.14159265358979323846264338327950288419716939937510f

#include "../Helper/Helper.c"
#include "pop.h"


void bitpop(int bitsize, int genes, int individuals, int** result){

    /*
    Fill a matrix with random bits.

    :param bitsize: The size of the integer in binary.
    :type bitsize: int

    :param genes: The number of genes in an individual.
    :type genes: int

    :param individuals: The number of individuals in the population.
    :type individuals: int

    :param result: The matrix to be filled with random bits.
                   shape = (individuals, genes * bitsize)
    :type result: int**

    */

   for(int i=0; i<individuals; i++){
       for(int j=0; j<genes * bitsize; j++){
           result[i][j] = rand() % 2;
       }
   }

}
void uniform_bit_pop(int bitsize, int genes, int individuals,
                     float factor, float bias, int** result){
    /*
    Fill a matrix with bits according to a uniform distribution.

    :param bitsize: The size of the bitstring.
    :type bitsize: int

    :param genes: The number of genes in the bitstring.
    :type genes: int

    :param individuals: The number of individuals in the bitstring.
    :type individuals: int

    :param factor: The factor by which the uniform distribution is scaled.
    :type factor: float

    :param bias: The bias of the uniform distribution.
    :type bias: float

    The factor and bias are used to calculate the upper and lower bounds of the uniform distribution 
    according to the following formula: [1]

    .. math::
        upper = round((bias + factor) * 2^{bitsize}) \\
        lower = round((bias - factor) * 2^{bitsize})

    Which results in the integer domain between round(lower * 2^{bitsize}) and round(upper * 2^{bitsize}).

    :param normalised: Whether the uniform distribution is normalised.
    :type normalised: int


    :param result: The matrix to be filled with bits according to a uniform distribution.
                   shape = (individuals, genes * bitsize)
    :type result: int**

    References
    ----------
    .. [1] https://stackoverflow.com/questions/11641629/generating-a-uniform-distribution-of-integers-in-c Lior Kogan (2012)

    */

    int upper = (int) roundf((bias + factor) * pow(2, bitsize));
    int lower = (int) roundf((bias - factor) * pow(2, bitsize));

    int** temp = malloc(sizeof(int) * individuals );

    for(int i=0; i<individuals; i++){
        temp[i] = (int*) malloc(sizeof(int) * genes * bitsize);
    }

    // now we have upper and lower bounds for the uniform distribution in the 
    // integer domain between round(lower * 2^bitsize) and round(upper * 2^bitsize)

    unsigned int nRange = (unsigned int)(upper - lower);
    unsigned int nRangeBits = (unsigned int) ceil(log2((double) (nRange)));

    // now we have the number of bits required to represent the range of the
    // uniform distribution in the integer domain

    unsigned int nRand; // random number

    for(int i=0; i<individuals; i++){
        for(int j=0; j<genes * bitsize; j++){
            do{
                nRand = 0;
                for(int k=0; k<nRangeBits; k++){
                    nRand = (nRand << 1) | (rand() & 1);
                }
            } while(nRand >= nRange);
            temp[i][j] = (int) (nRand + lower);
        }
    }

    // now we have a matrix of random integers between round(lower * 2^bitsize)
    // and round(upper * 2^bitsize) which we can convert to a matrix of bits

    intmat2binmat( temp, bitsize, genes, individuals, result);

    // free the memory
    for(int i=0; i<individuals; i++){
        free(temp[i]);
    }

    free(temp);
}
void normal_bit_pop_boxmuller(int bitsize, int genes, int individuals,
                    float factor, float bias,
                    float loc, float scale, int** result){
    /* 
    Fill a matrix with bits according to a normal distribution.
    using the following probability density function:

    .. math::
        f(x) = \\frac{1}{\\sigma \\sqrt{2 \\pi}} e^{-\\frac{1}{2} (\\frac{x - \\mu}{\\sigma})^2}

    Calculate them using a Box-Muller transform, where two random numbers are generated 
    according to a uniform distribution and then transformed to a normal distribution with
    the following formula:

    .. math::
        z_0 = \sqrt{-2 \ln U_1 } \cos{(2 \pi U_2)} \\
        z_1 = \sqrt{-2 \ln U_1 } \sin{(2 \pi U_2)}

    Where :math:`U_1` and :math:`U_2` are random numbers between 0 and 1.

    :param bitsize: The size of the bitstring.
    :type bitsize: int

    :param genes: The number of genes in the bitstring.
    :type genes: int

    :param individuals: The number of individuals in the bitstring.
    :type individuals: int

    :param factor: The factor by which the normal distribution is scaled.
    :type factor: float

    :param bias: The bias of the normal distribution.
    :type bias: float

    The factor and bias are used for the conversion to the integer domain according to the following formula:

    .. math::
        float = (int - bias) / factor

    :param normalised: Whether the normal distribution is normalised.
    :type normalised: int

    :param loc: The mean of the normal distribution.
    :type loc: float

    :param scale: The standard deviation of the normal distribution.
    :type scale: float

    :param result: The matrix to be filled with bits according to a normal distribution.
                   shape = (individuals, genes * bitsize)
    :type result: int**
    */

    int numbers_inuni = genes;

    // fill temp with uniform numbers
    // if the amount of genes is not div by 2 add one
    if(genes % 2 !=0 ){
        numbers_inuni ++;
    }

    int** temp = malloc(sizeof(int*) * individuals );

    for(int i=0; i<individuals; i++){
        temp[i] = malloc(sizeof(int) * numbers_inuni);
    }



    int lower_uniform = 0;
    int upper_uniform = pow(2, bitsize);

    uniform_random(individuals, numbers_inuni, lower_uniform, upper_uniform, temp);

    // convert to floats between 0 and 1

    float** normal_dist = malloc(sizeof(float*) * numbers_inuni );
    for(int i=0; i<individuals; i++){
        normal_dist[i] = malloc(sizeof(float) * numbers_inuni);
        for(int j =0; j< numbers_inuni; j++){
            normal_dist[i][j] = (float) (temp[i][j] / pow(2, bitsize));
        }
    }

    for(int i=0; i<individuals; i++){
        free(temp[i]);
    }

    free(temp);

    float Z0;
    float Z1;

    for(int i = 0; i<individuals; i++){
        for(int j = 0; j<(int) roundf(numbers_inuni/2); j+=2){
            
            Z0 = (sqrtf(-2 * logf(normal_dist[i][j])) * cosf(2 * PI * normal_dist[i][j + 1]) * scale) + loc;
            Z1 = (sqrtf(-2 * logf(normal_dist[i][j])) * sinf(2 * PI * normal_dist[i][j + 1]) * scale) + loc;

            normal_dist[i][j] = Z0;
            normal_dist[i][j + 1] = Z1;
        }
    }

    // Convert to binary matrix
    int2ndbit(normal_dist, bitsize, genes, individuals, factor, bias, result);


    
    // free the memory
    for(int i=0; i<individuals; i++){
        free(normal_dist[i]);
    }

    free(normal_dist);
}

void normal_bit_pop(int bitsize, int genes, int individuals,
                    float factor, float bias,
                    float loc, float scale, int** result){
    /*

    Produce a normal distributed set of values using the Gaussian distribution:

    .. math::
        f(x) = \frac{1}{\sigma \sqrt{2 \pi}} e^{-\frac{1}{2} (\frac{x - \mu}{\sigma})^2}

    Where x is linearly spaced between (-factor and factor) + bias.

    :param bitsize: The size of the bitstring.
    :type bitsize: int

    :param genes: The number of genes in the bitstring.
    :type genes: int

    :param individuals: The number of individuals in the bitstring.
    :type individuals: int

    :param factor: The factor by which the normal distribution is scaled.
    :type factor: float

    :param bias: The bias of the normal distribution.
    :type bias: float

    The factor and bias are used for the conversion to the integer domain according to the following formula:

    .. math::
        float = (int - bias) / factor

    :param normalised: Whether the normal distribution is normalised.
    :type normalised: int

    :param loc: The mean of the normal distribution.
    :type loc: float

    :param scale: The standard deviation of the normal distribution. Due to the use of floats the scale should be > 2.
    :type scale: float

    :param result: The matrix to be filled with bits according to a normal distribution.
                   shape = (individuals, genes * bitsize)

    */

   // Determine the steps between the values in the normal distribution
    float step = (2 * factor) / (genes *individuals);

    // Determine the lower and upper bounds of the normal distribution
    float lower = scale * (-factor) + loc;
    float upper = scale * factor + loc;

    // Determine the number of values in the normal distribution
    int numvalues = genes * individuals;

    // Fill the normal distribution with values using the formula
    double* normal_dist = malloc(sizeof(double) * numvalues);

    double* range = malloc(sizeof(double) * numvalues);
    for(int i = 0; i < numvalues; i++){
        range[i] = -factor + (i * step);
    }

    float sum = 0;
    for(int i = 0; i < numvalues; i++){
        normal_dist[i] = gaussian(range[i], loc, scale);
        sum += normal_dist[i];
    } 

    // normalise the normal distribution
    for(int i = 0; i < numvalues; i++){
        normal_dist[i] = normal_dist[i] / sum;
    }

    // use the probability density function to compute random numbers
    // according to the normal distribution



    // printf("probability density function: \n");
    // for(int i = 0; i < numvalues; i++){
    //     printf("p: %f; val: %f; idx: %d \n", normal_dist[i], range[i], i);
    // }

    // printf("\n");

    float** normal_distmat = malloc(sizeof(float*) * numvalues);
    int* indices = malloc(sizeof(int) * numvalues);
    roulette_wheel(normal_dist, numvalues, numvalues, indices);

    // printf("indices: \n");
    // for(int i = 0; i < numvalues; i++){
    //     printf("idx: %d ;p: %f ; val: %f; i: %d\n", indices[i], normal_dist[indices[i]], range[indices[i]], i);
    // }

    for(int i = 0; i < individuals; i++){  
        normal_distmat[i] = malloc(sizeof(float) * genes);    
        for(int j = 0; j < genes; j++){
            normal_distmat[i][j] = range[indices[(i * genes) + j]];
        }
    }


    free(normal_dist);
    free(range);

    // convert to binary matrix
    int2ndbit(normal_distmat, bitsize, genes, individuals, factor, bias, result);

    // free the memory
    for(int i=0; i<individuals; i++){
        free(normal_distmat[i]);
    }

    free(normal_distmat);

}


void cauchy_bit_pop(int bitsize, int genes, int individuals,
                    float factor, float bias,
                    float loc, float scale, int** result){
    /*
    
    Produce a normal distributed set of values using the Cauchy distribution:

    .. math::
        f(x) = \frac{1}{\pi \gamma [1 + (\frac{x - x_0}{\gamma})^2]}

    Where x is linearly spaced between (-factor and factor) + bias.

    :param bitsize: The size of the bitstring.
    :type bitsize: int

    :param genes: The number of genes in the bitstring.
    :type genes: int

    :param individuals: The number of individuals in the bitstring.
    :type individuals: int

    :param factor: The factor by which the normal distribution is scaled.
    :type factor: float

    :param bias: The bias of the normal distribution.
    :type bias: float

    :param normalised: Whether the normal distribution is normalised.
    :type normalised: int

    :param loc: The location of the peak of the distribution.
    :type loc: float

    :param scale: The width of the distribution.
    :type scale: float   

    :param result: The matrix to be filled with bits according to a normal distribution.
                   shape = (individuals, genes * bitsize)
    :type result: int** 

    */
// Determine the steps between the values in the normal distribution
    float step = (2 * factor) / (genes *individuals);

    // Determine the lower and upper bounds of the normal distribution
    float lower = scale * (-factor) + loc;
    float upper = scale * factor + loc;

    // print when bias is not in the range of the normal distribution
    if(bias < lower || bias > upper){
        printf("Warning: bias is not in the range of the normal distribution\n");
    }

    // Determine the number of values in the normal distribution
    int numvalues = genes * individuals;

    // Fill the normal distribution with values using the formula
    double* normal_dist = malloc(sizeof(double) * numvalues);

    double* range = malloc(sizeof(double) * numvalues);
    for(int i = 0; i < numvalues; i++){
        range[i] = -factor + (i * step);
    }

    float sum = 0;
    for(int i = 0; i < numvalues; i++){
        normal_dist[i] = cauchy(range[i], loc, scale);
        sum += normal_dist[i];
    } 

    // normalise the normal distribution
    for(int i = 0; i < numvalues; i++){
        normal_dist[i] = normal_dist[i] / sum;
    }

    // use the probability density function to compute random numbers
    // according to the normal distribution



    // printf("probability density function: \n");
    // for(int i = 0; i < numvalues; i++){
    //     printf("p: %f; val: %f; idx: %d \n", normal_dist[i], range[i], i);
    // }

    // printf("\n");

    float** normal_distmat = malloc(sizeof(float*) * numvalues);
    int* indices = malloc(sizeof(int) * numvalues);
    roulette_wheel(normal_dist, numvalues, numvalues, indices);

    // printf("indices: \n");
    // for(int i = 0; i < numvalues; i++){
    //     printf("idx: %d ;p: %f ; val: %f; i: %d\n", indices[i], normal_dist[indices[i]], range[indices[i]], i);
    // }

    for(int i = 0; i < individuals; i++){  
        normal_distmat[i] = malloc(sizeof(float) * genes);    
        for(int j = 0; j < genes; j++){
            normal_distmat[i][j] = range[indices[(i * genes) + j]];
        }
    }


    free(normal_dist);
    free(range);

    // convert to binary matrix
    int2ndbit(normal_distmat, bitsize, genes, individuals, factor, bias, result);

    // free the memory
    for(int i=0; i<individuals; i++){
        free(normal_distmat[i]);
    }

    free(normal_distmat);
}
