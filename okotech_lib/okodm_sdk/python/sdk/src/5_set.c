/****************************************************
 * FileName: 5_set.c
 * Description:
 *
 * (c) 2005, FlexibleOptical
 *           Gleb Vdovin & Mikhail Loktev
 ****************************************************
 */

#include <stdio.h>
#include "mirror_edac40.h"

int main (int argc, char *argv[])
{
	int i, nc;
	char mess[255];

	if (argc != 2)
		nc = MAX_AMPLITUDE;
	else
		sscanf (argv[1], "%d", &nc);

	if (!init_dac())
		exit(1);

	if (TOTAL_NR_OF_CHANNELS!=19)
	{
		printf("\nThis is not a 19-ch mirror\n");
		exit(0);
	}

	sprintf(mess, "\nThis program will set %d to 6-17 channels of MMDM17TT.\n\n", nc);
	printf(mess);

	for (i=0; i<5; i++) voltage[i]=0;
	for (i=5; i<17; i++) voltage[i]=nc;
	voltage[17] = voltage[18] = 0;
	set_mirror();
	close_dac();
	return 0;
}
