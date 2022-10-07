 /****************************************************
 * FileName: rotate.c
 * Description:
 *
 * (c) 2004, FlexibleOptical
 *           Gleb Vdovin & Mikhail Loktev & Oleg Soloviev
 ****************************************************
 */

#include <stdio.h>
#include "mirror_edac40.h"


int
main (int argc, char *argv[])
{
	int i, nc, j;
	char mess[255];

	if (argc != 2)
		nc = MAX_AMPLITUDE;
	else
		sscanf (argv[1], "%d", &nc);

	if (!init_dac())
		exit(1);

	sprintf(mess, "\nThis program will set %d to all channels of %d-ch OKO mirror, one by one\n", nc, TOTAL_NR_OF_CHANNELS);
	printf(mess);

	for (i = 0; i < TOTAL_NR_OF_CHANNELS; i++) voltage[i]=0;
	set_mirror();
	Sleep(500);

	for (j=0; j<TOTAL_NR_OF_CHANNELS; j++)
	{
		for (i = 0; i < TOTAL_NR_OF_CHANNELS; i++) voltage[i]=0;
		voltage[j]=nc;
		set_mirror();
		Sleep(500);
	}

	for (i = 0; i < TOTAL_NR_OF_CHANNELS; i++) voltage[i]=0;
	set_mirror();

	Sleep(500);

    #ifdef PIEZO

    for (i = 0; i < TOTAL_NR_OF_CHANNELS; i++) voltage[i]=nc;
	set_mirror();


	for (j=0; j<TOTAL_NR_OF_CHANNELS; j++)
	{
		for (i = 0; i < TOTAL_NR_OF_CHANNELS; i++) voltage[i]=nc;
		voltage[j]=0;
		set_mirror();
		Sleep(500);
	}

	for (i = 0; i < TOTAL_NR_OF_CHANNELS; i++) voltage[i]=0;
	set_mirror();

	#endif

	close_dac();

	return 0;
}
