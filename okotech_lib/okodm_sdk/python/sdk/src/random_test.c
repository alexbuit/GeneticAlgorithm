#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mirror_edac40.h"

int main (int argc, char *argv[])
{
    int i, del;

    srand((unsigned)time(NULL));
    if (!init_dac())
        exit(1);

    if (argc != 2)
        del = 2;
    else
        sscanf (argv[1], "%d", &del);

    do
    {
        for (i=0; i<TOTAL_NR_OF_CHANNELS; i++)
        {
            if ( (double) rand()/RAND_MAX  > 0.5)
                voltage[i]=4095;
            else
                voltage[i]=0;
        }
        set_mirror();
        Sleep(del);
    } while (1);

    close_dac();
}
