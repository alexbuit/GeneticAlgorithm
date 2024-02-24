
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "../../Utility/pop.h"

#include "TestPop.h"

int main(){
    // Testbitpop(32, 16, 16, 1);
    // Testuniformpop(32, 16, 16, 10.0f, 1.0f, 1, 1);
    double loc = 0.0f;
    double scale = 2.0f;
    double factor = 10.0f;
    double bias = 1.0f;
    Testnormalpop(32, 16, 16, loc, scale, factor, bias, 1);
    Testcauchypop(32, 16, 16, loc, scale, factor, bias, 1);
    // TestnormalpopBM(32, 64, 16, 5.0f, 1.0f, 10.0f, 0.0f, 1, 1);
}

void Testbitpop(int bitsize, int genes, int individuals, int writeresult){

    int** result = (int**)malloc(individuals * sizeof(int*));
    for (int i = 0; i < individuals; i++){
        result[i] = (int*)malloc(genes * bitsize * sizeof(int));
    }

    double** numresult = malloc(individuals * sizeof(double*));

    for (int i = 0; i < individuals; i++){
        numresult[i] = malloc(genes * sizeof(double));
    }

    bitpop(bitsize, genes, individuals, result);

    ndbit2int(result, bitsize, genes, individuals, 5.0f, 0.0f, numresult);

    printMatrix(result, individuals, genes * bitsize);
    printfMatrix(numresult, individuals, genes, 4);

    double factor = 0;
    double bias = 0;
    int normalised = 1;

    if (writeresult == 1){
        char filename[] = "TestBitPop.txt";
        write2file(bitsize, genes, individuals, factor, bias, filename, result, numresult);
    }   




    for (int i = 0; i < individuals; i++){
        free(numresult[i]);
        free(result[i]);
        
    }
    
    free(numresult);
    free(result);
    
}

void Testuniformpop(int bitsize, int genes, int individuals,
                     double factor, double bias, int writeresult){

        int** result = (int**)malloc(individuals * sizeof(int*));
    for (int i = 0; i < individuals; i++){
        result[i] = (int*)malloc(genes * bitsize * sizeof(int));
    }

    double** numresult = malloc(individuals * sizeof(double*));

    for (int i = 0; i < individuals; i++){
        numresult[i] = malloc(genes * sizeof(double));
    }

    bitpop(bitsize, genes, individuals, result);

    ndbit2int(result, bitsize, genes, individuals, factor, bias, numresult);

    printMatrix(result, individuals, genes * bitsize);
    printfMatrix(numresult, individuals, genes, 4);

    if (writeresult == 1){
        char filename[] = "Testbituniformpop.txt";
        write2file(bitsize, genes, individuals, factor, bias, filename, result, numresult);
    }   



    for (int i = 0; i < individuals; i++){
        free(numresult[i]);
        free(result[i]);
        
    }
    
    free(numresult);
    free(result);

}

void TestnormalpopBM(int bitsize, int genes, int individuals, double loc, double scale,
                     double factor, double bias, int writeresult){

    /*

    */
    int** result = (int**)malloc(individuals * sizeof(int*));
    for (int i = 0; i < individuals; i++){
        result[i] = (int*)malloc(genes * bitsize * sizeof(int));
    }

    double** numresult = malloc(individuals * sizeof(double*));

    for (int i = 0; i < individuals; i++){
        numresult[i] = malloc(genes * sizeof(double));
    }

    normal_bit_pop_boxmuller(bitsize, genes, individuals, factor, bias, loc, scale, result);

    ndbit2int(result, bitsize, genes, individuals, factor, bias, numresult);

    printMatrix(result, individuals, genes * bitsize);
    printfMatrix(numresult, individuals, genes, 8);

    if (writeresult == 1){
        char filename[] = "TestBitNormalPopBoxMuller.txt";
        write2file(bitsize, genes, individuals, factor, bias, filename, result, numresult);
    }   

    for (int i = 0; i < individuals; i++){
    free(numresult[i]);
    free(result[i]);
    
    }
    
    free(numresult);
    free(result);

}

void Testnormalpop(int bitsize, int genes, int individuals, double loc, double scale,
                     double factor, double bias, int writeresult){

    /*

    */
    int** result = (int**)malloc(individuals * sizeof(int*));
    for (int i = 0; i < individuals; i++){
        result[i] = (int*)malloc(genes * bitsize * sizeof(int));
    }

    double** numresult = malloc(individuals * sizeof(double*));

    for (int i = 0; i < individuals; i++){
        numresult[i] = malloc(genes * sizeof(double));
    }

    normal_bit_pop(bitsize, genes, individuals, factor, bias, loc, scale,result);

    ndbit2int(result, bitsize, genes, individuals, factor, bias, numresult);

    // printMatrix(result, individuals, genes * bitsize);
    // printfMatrix(numresult, individuals, genes, 8);

    if (writeresult == 1){
        char filename[] = "TestBitNormalPop.txt";
        write2file(bitsize, genes, individuals, factor, bias ,filename, result, numresult);
    }   

    for (int i = 0; i < individuals; i++){
    free(numresult[i]);
    free(result[i]);
    
    }
    
    free(numresult);
    free(result);

}

void Testcauchypop(int bitsize, int genes, int individuals, double loc, double scale,
                     double factor, double bias, int writeresult){

    /*

    */
    int** result = (int**)malloc(individuals * sizeof(int*));
    for (int i = 0; i < individuals; i++){
        result[i] = (int*)malloc(genes * bitsize * sizeof(int));
    }

    double** numresult = malloc(individuals * sizeof(double*));

    for (int i = 0; i < individuals; i++){
        numresult[i] = malloc(genes * sizeof(double));
    }

    cauchy_bit_pop(bitsize, genes, individuals, factor, bias, loc, scale,result);

    ndbit2int(result, bitsize, genes, individuals, factor, bias, numresult);

    // printMatrix(result, individuals, genes * bitsize);
    // printfMatrix(numresult, individuals, genes, 8);

    if (writeresult == 1){
        char filename[] = "TestBitCauchyPop.txt";
        write2file(bitsize, genes, individuals, factor, bias ,filename, result, numresult);
    }   

    for (int i = 0; i < individuals; i++){
    free(numresult[i]);
    free(result[i]);
    
    }
    
    free(numresult);
    free(result);

}

void write2file(int bitsize, int genes, int individuals, double factor, double bias, char* filename, int** result, double** numresult){

    /*

    */

    // printf("result: \n");
    // for(int i=0; i<individuals; i++){
    //     for(int j=0; j<genes; j++){
    //         for(uint32_t k = 1; k != 0x00000000; k = k << 1){
    //             printf("%c", (result[i][j]&k ? '1':'0'));
    //         }
    //     }
    //     printf("\n");
    // }

        // Write result to file
    FILE* fp;
    fp = fopen(filename, "w");
    fprintf(fp, "Binarystring\n");
    for (int i = 0; i < individuals; i++) {
        fprintf(fp, "");
        for (int j = 0; j < genes; j++) {
            for(uint32_t k = 1; k != 0x00000000; k = k << 1){
                fprintf(fp, "%c", (result[i][j]&k ? '1':'0'));
            }
        }
        fprintf(fp, "\n");
    }
    fprintf(fp, "\n");

    // print numerical values
    fprintf(fp, "inidviduals %d genes %d bitsize %d factor %.6f bias %.6f normalised %d\n", individuals, genes, bitsize, factor, bias);
    fprintf(fp, "x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15\n");

    for (int i = 0; i < individuals; i++) {
        for (int j = 0; j < genes; j++) {
            fprintf(fp, "%.8f", numresult[i][j]);
            if (j < genes - 1) {
                fprintf(fp, ",");
            }
        }
        fprintf(fp, "\n");
    }

    fprintf(fp, "\n");

    fclose(fp);
}