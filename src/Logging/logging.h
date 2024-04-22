
#ifndef LOGGING_H
#define LOGGING_H 

#include "../helper/struct.h"

void write_param(gene_pool_t gene_pool, int iteration);
void open_file(gene_pool_t gene_pool);
void close_file();

#endif // LOGGING_H