/**************************************************** 
 * FileName: ring2_set.cpp
 * Description:  
 *
 * (c) 2004, FlexibleOptical 
 *           Gleb Vdovin & Mikhail Loktev
 ****************************************************
 */

#include "mirror.h"

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

	if (TOTAL_NR_OF_CHANNELS!=79)
	{
		printf("\nThis is not a 79-ch mirror\n");
		exit(0);
	}

	sprintf(mess, "\nThis program will set %d to 18-79 channels of OKO 79ch mirror.\n\n", nc);
	printf(mess);

	for (i=0; i<17; i++) voltage[i]=0;
	for (i=17; i<79; i++) voltage[i]=nc;
	set_mirror();
	close_dac();
	return 0;
}
 
