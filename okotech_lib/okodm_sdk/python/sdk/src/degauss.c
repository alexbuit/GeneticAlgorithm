/****************************************************
 * FileName: degauss.c
 * Description:
 *
 * (c) 2006, FlexibleOptical
 *           Gleb Vdovin
 ****************************************************
 */

#include <stdio.h>
#include "mirror_edac40.h"

int main (int argc, char *argv[])
{
    int i, ii,  j, jj, period=2500, n_steps, step;

    if (argc != 2)
        n_steps = 200;
    else
        sscanf (argv[1], "%d", &n_steps);

    step = 65535/n_steps;

    if (!init_dac())
        exit(1);

    printf ("\nThis program will set maximum voltage to  all  mirror channels  periodically \n");

    for (i=0; i<TOTAL_NR_OF_CHANNELS; i++) voltage[i] =0.;
    set_mirror();
    for (jj=0; jj<period; jj++);

    for (i=0; i<TOTAL_NR_OF_CHANNELS; i++)
    {
        printf("Channel =  %d \n", i+1);
        for (j=65535; j>=0; j-=step)
        {
            for(ii=1; ii<=1; ii++)
            {
                voltage[i] = 0;
                set_mirror();
                for (jj=0; jj<period; jj++);
                voltage[i] = j;
                set_mirror();
                for (jj=0; jj<period; jj++);
            }
        }
    }
    for (i=0; i<TOTAL_NR_OF_CHANNELS; i++) voltage[i] =0.;
    set_mirror();

    close_dac();

    return 0;
}
