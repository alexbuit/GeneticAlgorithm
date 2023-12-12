
#include <stdio.h>
#include <stdlib.h>

#include "../../Utility/pop.h"
#include "../../Utility/pop.c"

#include "TestPop.h"

int main(){
    Testbitpop(32, 16, 16, 1);
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
    printfMatrix(numresult, individuals, genes);

    if (writeresult == 1){
        // Write result to file
        FILE* fp;
        fp = fopen("TestBitPop.txt", "w");
        fprintf(fp, "cols: %d\n", genes * bitsize);
        fprintf(fp, "rows: %d\n", individuals);

        fprintf(fp, "[");
        for (int i = 0; i < individuals; i++) {
            fprintf(fp, "[");
            for (int j = 0; j < genes * bitsize; j++) {
                fprintf(fp, "%d", result[i][j]);
                if (j < genes * bitsize - 1) {
                    fprintf(fp, ", ");
                }
            }
            fprintf(fp, "]");
            if (i < individuals - 1) {
                fprintf(fp, ", \n");
            }
        }
        fprintf(fp, "]\n");

        // print numerical values
        fprintf(fp, "[");

        for (int i = 0; i < individuals; i++) {
            fprintf(fp, "[");
            for (int j = 0; j < genes; j++) {
                fprintf(fp, "%.4f", numresult[i][j]);
                if (j < genes - 1) {
                    fprintf(fp, ", ");
                }
            }
            fprintf(fp, "]");
            if (i < individuals - 1) {
                fprintf(fp, ", \n");
            }
        }

        fprintf(fp, "]\n");

        fclose(fp);
    }   



    for (int i = 0; i < individuals; i++){
        free(numresult[i]);
        free(result[i]);
        
    }
    
    free(numresult);
    free(result);
    
}

void Testuniformpop(int bitsize, int genes, int individuals, int writeresult){

    

}