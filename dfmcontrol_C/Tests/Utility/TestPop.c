
#include <stdio.h>
#include <stdlib.h>

#include "../../Utility/pop.h"
#include "../../Utility/pop.c"

#include "TestPop.h"

int main(){
    // Testbitpop(32, 16, 16, 1);
    // Testuniformpop(32, 16, 16, 10.0f, 1.0f, 1, 1);
    float loc = 0.0f;
    float scale = 2.0f;
    float factor = 10.0f;
    float bias = 1.0f;
    Testnormalpop(32, 64, 64, loc, scale, factor, bias, 1, 1);
    // TestnormalpopBM(32, 64, 16, 5.0f, 1.0f, 10.0f, 0.0f, 1, 1);
}

void Testbitpop(int bitsize, int genes, int individuals, int writeresult){

    int** result = (int**)malloc(individuals * sizeof(int*));
    for (int i = 0; i < individuals; i++){
        result[i] = (int*)malloc(genes * bitsize * sizeof(int));
    }

    float** numresult = malloc(individuals * sizeof(float*));

    for (int i = 0; i < individuals; i++){
        numresult[i] = malloc(genes * sizeof(float));
    }

    bitpop(bitsize, genes, individuals, result);

    ndbit2int(result, bitsize, genes, individuals, 5.0f, 0.0f,(int) 1, numresult);

    printMatrix(result, individuals, genes * bitsize);
    printfMatrix(numresult, individuals, genes, 4);

    float factor = 0;
    float bias = 0;
    int normalised = 1;

    if (writeresult == 1){
        char filename[] = "TestBitPop.txt";
        write2file(bitsize, genes, individuals, factor, bias, normalised, filename, result, numresult);
    }   




    for (int i = 0; i < individuals; i++){
        free(numresult[i]);
        free(result[i]);
        
    }
    
    free(numresult);
    free(result);
    
}

void Testuniformpop(int bitsize, int genes, int individuals,
                     float factor, float bias, int normalised, int writeresult){

        int** result = (int**)malloc(individuals * sizeof(int*));
    for (int i = 0; i < individuals; i++){
        result[i] = (int*)malloc(genes * bitsize * sizeof(int));
    }

    float** numresult = malloc(individuals * sizeof(float*));

    for (int i = 0; i < individuals; i++){
        numresult[i] = malloc(genes * sizeof(float));
    }

    bitpop(bitsize, genes, individuals, result);

    ndbit2int(result, bitsize, genes, individuals, factor, bias, normalised, numresult);

    printMatrix(result, individuals, genes * bitsize);
    printfMatrix(numresult, individuals, genes, 4);

    if (writeresult == 1){
        char filename[] = "Testbituniformpop.txt";
        write2file(bitsize, genes, individuals, factor, bias, normalised, filename, result, numresult);
    }   



    for (int i = 0; i < individuals; i++){
        free(numresult[i]);
        free(result[i]);
        
    }
    
    free(numresult);
    free(result);

}

void TestnormalpopBM(int bitsize, int genes, int individuals, float loc, float scale,
                     float factor, float bias, int normalised, int writeresult){

    /*

    */
    int** result = (int**)malloc(individuals * sizeof(int*));
    for (int i = 0; i < individuals; i++){
        result[i] = (int*)malloc(genes * bitsize * sizeof(int));
    }

    float** numresult = malloc(individuals * sizeof(float*));

    for (int i = 0; i < individuals; i++){
        numresult[i] = malloc(genes * sizeof(float));
    }

    normal_bit_pop_boxmuller(bitsize, genes, individuals, factor, bias, normalised, loc, scale,result);

    ndbit2int(result, bitsize, genes, individuals, factor, bias,(int) 1, numresult);

    printMatrix(result, individuals, genes * bitsize);
    printfMatrix(numresult, individuals, genes, 8);

    if (writeresult == 1){
        char filename[] = "TestBitNormalPopBoxMuller.txt";
        write2file(bitsize, genes, individuals, factor, bias, normalised, filename, result, numresult);
    }   

    for (int i = 0; i < individuals; i++){
    free(numresult[i]);
    free(result[i]);
    
    }
    
    free(numresult);
    free(result);

}

void Testnormalpop(int bitsize, int genes, int individuals, float loc, float scale,
                     float factor, float bias, int normalised, int writeresult){

    /*

    */
    int** result = (int**)malloc(individuals * sizeof(int*));
    for (int i = 0; i < individuals; i++){
        result[i] = (int*)malloc(genes * bitsize * sizeof(int));
    }

    float** numresult = malloc(individuals * sizeof(float*));

    for (int i = 0; i < individuals; i++){
        numresult[i] = malloc(genes * sizeof(float));
    }

    normal_bit_pop(bitsize, genes, individuals, factor, bias, normalised, loc, scale,result);

    ndbit2int(result, bitsize, genes, individuals, factor, bias,(int) 1, numresult);

    // printMatrix(result, individuals, genes * bitsize);
    // printfMatrix(numresult, individuals, genes, 8);

    if (writeresult == 1){
        char filename[] = "TestBitNormalPop.txt";
        write2file(bitsize, genes, individuals, factor, bias, normalised ,filename, result, numresult);
    }   

    for (int i = 0; i < individuals; i++){
    free(numresult[i]);
    free(result[i]);
    
    }
    
    free(numresult);
    free(result);

}

void write2file(int bitsize, int genes, int individuals, float factor, float bias, int normalised, char* filename, int** result, float** numresult){

    /*

    */
           // Write result to file
        FILE* fp;
        fp = fopen(filename, "w");
        fprintf(fp, "Binarystring\n");
        for (int i = 0; i < individuals; i++) {
            fprintf(fp, "");
            for (int j = 0; j < genes * bitsize; j++) {
                fprintf(fp, "%d", result[i][j]);
            }
            fprintf(fp, "\n");
        }
        fprintf(fp, "\n");

        // print numerical values
        fprintf(fp, "inidviduals %d genes %d bitsize %d factor %.6f bias %.6f normalised %d\n", individuals, genes, bitsize, factor, bias, normalised);
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