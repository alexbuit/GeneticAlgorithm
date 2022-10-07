  /****************************************************
 * FileName: pairs.c
 * Description:
 *
 * (c) 2005, FlexibleOptical
 *           Gleb Vdovin     & Mikhail Loktev
 ****************************************************
 */

#include <stdio.h>
#include "mirror_edac40.h"


int main (int argc, char *argv[])
{
	int i, j;

	if (!init_dac())
		exit(1);

	for (j=0; j<TOTAL_NR_OF_CHANNELS/2; j++)
	{
		for (i=0; i<TOTAL_NR_OF_CHANNELS; i++)
			voltage[i] = 0;
		voltage[j] = MAX_AMPLITUDE;
		voltage[j + TOTAL_NR_OF_CHANNELS/2] = MAX_AMPLITUDE;
		set_mirror();
		printf("%d and %d\n", j+1, j+1+TOTAL_NR_OF_CHANNELS/2);
		Sleep(500);
	}

	for (j=0; j<TOTAL_NR_OF_CHANNELS/2; j++)
	{
		for (i=0; i<TOTAL_NR_OF_CHANNELS; i++)
			voltage[i] = MAX_AMPLITUDE;
		voltage[j] = 0;
		voltage[j + TOTAL_NR_OF_CHANNELS/2] = 0;
		set_mirror();
		printf("%d and %d\n", j+1, j+1+TOTAL_NR_OF_CHANNELS/2);
		Sleep(500);
	}

	for (i=0; i<TOTAL_NR_OF_CHANNELS; i++)
        voltage[i] = 0;
	for (j=0; j<TOTAL_NR_OF_CHANNELS/2; j+=2)
		voltage[j] = voltage[j+TOTAL_NR_OF_CHANNELS/2+1] = MAX_AMPLITUDE;
	set_mirror();
	Sleep(2500);

	for (i=0; i<TOTAL_NR_OF_CHANNELS; i++)
        voltage[i] = MAX_AMPLITUDE;
	for (j=0; j<TOTAL_NR_OF_CHANNELS/2; j+=2)
		voltage[j] = voltage[j+TOTAL_NR_OF_CHANNELS/2+1] = 0;
	set_mirror();
	Sleep(2500);

	for (i=0; i<TOTAL_NR_OF_CHANNELS; i++)
		voltage[i] = 0;
	set_mirror();

	close_dac();

	return 0;
}
