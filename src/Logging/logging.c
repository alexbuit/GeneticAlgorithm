
#include "../helper/struct.h"
#include <stdio.h>
#include <stdlib.h>

FILE* fileptr = NULL;
FILE* fileptrcsv = NULL;

void open_file(gene_pool_t gene_pool)
{
	if(fopen_s (&fileptr, "c:/temp/param.bin", "wb") != 0 || fopen_s(&fileptrcsv, "c:/temp/param.csv", "w") != 0)
	{
		printf("Error opening file!\n");
		exit(1);
	}
	fprintf(fileptrcsv, "iteration, individual, result, ");
	for (int i = 0; i < gene_pool.genes; i++)
	{
		fprintf(fileptrcsv, "gene%d, ", i);
	}
	fprintf(fileptrcsv, "\n");
}

void close_file()
{
	fclose(fileptr);
	fclose(fileptrcsv);
}

void write_param(gene_pool_t gene_pool, int iteration)
{
	if (fileptr == NULL)
	{
		printf("Error opening file!\n");
		exit(1);
	}
	fwrite(&iteration, sizeof(int), 1, fileptr);
	fwrite(&gene_pool.pop_param_double, sizeof(gene_pool.pop_param_double), 1, fileptr);
	fwrite(&gene_pool.pop_result_set, sizeof(gene_pool.pop_result_set), 1, fileptr);

	for (int i = 0; i < gene_pool.individuals; i++)
	{
		fprintf(fileptrcsv, "%d, %d, %f, ", iteration, i, gene_pool.pop_result_set[gene_pool.sorted_indexes[i]]);
		for (int j = 0; j < gene_pool.genes; j++)
		{
			fprintf(fileptrcsv, "%f, ", gene_pool.pop_param_double[i][j]);
		}
		fprintf(fileptrcsv, "\n");
	}
}