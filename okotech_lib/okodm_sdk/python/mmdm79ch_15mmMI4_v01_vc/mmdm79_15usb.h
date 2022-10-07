/****************************************************
 * FileName: mmdm79_15usb.h
 * Description:
 *
 * (c) 2020, FlexibleOptical
 *           Gleb Vdovin & Mikhail Loktev & Oleg Soloviev
 ****************************************************
 */

#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include "ftd2xx.h"

#define TOTAL_NR_OF_CHANNELS 79
// The total number of channels used

#define MAX_NR_OF_CHANNELS 80
// Dimension of the arrays "arr_chan_adr" and "arr_chan_adr"; should be >=TOTAL_NR_OF_CHANNELS+1

#define NR_OF_UNITS 2

#define MAX_AMPLITUDE 4095
// Maximum value that can be addressed

int voltage[TOTAL_NR_OF_CHANNELS];
// Array of voltages addressed to the mirror, values in the range [0..MAX_AMPLITUDE]

int arr_chan_adr[MAX_NR_OF_CHANNELS];

int arr_chan_brd[MAX_NR_OF_CHANNELS];

BYTE packet[130]; // Packet of DAC data
WORD buf[40];     // Buffer of DAC channels
FT_HANDLE *unit_handles = NULL;
char *arr_ser_num = NULL;

static BYTE DAC_CHANEL_TABLE[40] = // Table of DAC channels
    {
        /*DAC-> 0   1   2   3   4    |
        ---------------------+  OUTPUT    */
        7, 15, 23, 31, 39, //|  A
        6, 14, 22, 30, 38, //|  B
        5, 13, 21, 29, 37, //|  C
        4, 12, 20, 28, 36, //|  D
        3, 11, 19, 27, 35, //|  E
        2, 10, 18, 26, 34, //|  F
        1, 9, 17, 25, 33,  //|  G
        0, 8, 16, 24, 32   //|  H
};

void MakePacket(WORD *buf, BYTE *packet)
/*
   Forming of the packet of DAC data
   "buf" is an array consisting of 16-bit voltage values, which correspond to channels 1..40;
   "packet" is an array consisting of 129 bytes of data that provide setting of the voltage
   levels at the outputs of DAC-40-USB
*/
{
    BYTE *p = packet + 1;

    for (int i = 0, s = 0; i < 8; i++, s += 5)
    {
        // Forming of the address parts of the control data words to be addressed to five DAC chips
        *(p++) = 0;
        *(p++) = (i & 4) ? 0x1f : 0;
        *(p++) = (i & 2) ? 0x1f : 0;
        *(p++) = (i & 1) ? 0x1f : 0;

        // Forming of the voltage level codes from the buffer of DAC channels
        // using the table of DAC channels
        for (int j = 0, mask = 0x800; j < 12; j++, mask >>= 1)
            *(p++) =
                ((buf[DAC_CHANEL_TABLE[s + 0]] & mask) ? 0x01 : 0) |
                ((buf[DAC_CHANEL_TABLE[s + 1]] & mask) ? 0x02 : 0) |
                ((buf[DAC_CHANEL_TABLE[s + 2]] & mask) ? 0x04 : 0) |
                ((buf[DAC_CHANEL_TABLE[s + 3]] & mask) ? 0x08 : 0) |
                ((buf[DAC_CHANEL_TABLE[s + 4]] & mask) ? 0x10 : 0);
    }
    packet[0] = 0xff; // non-zero start byte
}

BOOL init_dac()
// Initialization of DAC-40-USB control units
{
    DWORD ndevs = 0;
    char sn[16];
    int i, j, k;
    int bInit;
    char strg1[256];

    arr_chan_adr[1] = 2;
    arr_chan_adr[2] = 16;
    arr_chan_adr[3] = 4;
    arr_chan_adr[4] = 1;
    arr_chan_adr[5] = 38;
    arr_chan_adr[6] = 30;
    arr_chan_adr[7] = 24;
    arr_chan_adr[8] = 21;
    arr_chan_adr[9] = 10;
    arr_chan_adr[10] = 22;
    arr_chan_adr[11] = 3;
    arr_chan_adr[12] = 32;
    arr_chan_adr[13] = 28;
    arr_chan_adr[14] = 18;
    arr_chan_adr[15] = 12;
    arr_chan_adr[16] = 8;
    arr_chan_adr[17] = 21;
    arr_chan_adr[18] = 19;
    arr_chan_adr[19] = 36;
    arr_chan_adr[20] = 5;
    arr_chan_adr[21] = 26;
    arr_chan_adr[22] = 22;
    arr_chan_adr[23] = 17;
    arr_chan_adr[24] = 14;
    arr_chan_adr[25] = 33;
    arr_chan_adr[26] = 6;
    arr_chan_adr[27] = 25;
    arr_chan_adr[28] = 0;
    arr_chan_adr[29] = 37;
    arr_chan_adr[30] = 39;
    arr_chan_adr[31] = 34;
    arr_chan_adr[32] = 7;
    arr_chan_adr[33] = 1;
    arr_chan_adr[34] = 39;
    arr_chan_adr[35] = 19;
    arr_chan_adr[36] = 4;
    arr_chan_adr[37] = 38;
    arr_chan_adr[38] = 31;
    arr_chan_adr[39] = 29;
    arr_chan_adr[40] = 11;
    arr_chan_adr[41] = 23;
    arr_chan_adr[42] = 20;
    arr_chan_adr[43] = 12;
    arr_chan_adr[44] = 35;
    arr_chan_adr[45] = 15;
    arr_chan_adr[46] = 11;
    arr_chan_adr[47] = 9;
    arr_chan_adr[48] = 23;
    arr_chan_adr[49] = 3;
    arr_chan_adr[50] = 37;
    arr_chan_adr[51] = 36;
    arr_chan_adr[52] = 34;
    arr_chan_adr[53] = 35;
    arr_chan_adr[54] = 30;
    arr_chan_adr[55] = 15;
    arr_chan_adr[56] = 27;
    arr_chan_adr[57] = 7;
    arr_chan_adr[58] = 26;
    arr_chan_adr[59] = 16;
    arr_chan_adr[60] = 18;
    arr_chan_adr[61] = 17;
    arr_chan_adr[62] = 13;
    arr_chan_adr[63] = 10;
    arr_chan_adr[64] = 27;
    arr_chan_adr[65] = 2;
    arr_chan_adr[66] = 8;
    arr_chan_adr[67] = 6;
    arr_chan_adr[68] = 32;
    arr_chan_adr[69] = 28;
    arr_chan_adr[70] = 13;
    arr_chan_adr[71] = 9;
    arr_chan_adr[72] = 5;
    arr_chan_adr[73] = 24;
    arr_chan_adr[74] = 14;
    arr_chan_adr[75] = 33;
    arr_chan_adr[76] = 31;
    arr_chan_adr[77] = 29;
    arr_chan_adr[78] = 25;
    arr_chan_adr[79] = 20;

    arr_chan_brd[1] = 1;
    arr_chan_brd[2] = 1;
    arr_chan_brd[3] = 1;
    arr_chan_brd[4] = 1;
    arr_chan_brd[5] = 0;
    arr_chan_brd[6] = 0;
    arr_chan_brd[7] = 0;
    arr_chan_brd[8] = 0;
    arr_chan_brd[9] = 1;
    arr_chan_brd[10] = 1;
    arr_chan_brd[11] = 1;
    arr_chan_brd[12] = 0;
    arr_chan_brd[13] = 0;
    arr_chan_brd[14] = 1;
    arr_chan_brd[15] = 1;
    arr_chan_brd[16] = 1;
    arr_chan_brd[17] = 1;
    arr_chan_brd[18] = 0;
    arr_chan_brd[19] = 0;
    arr_chan_brd[20] = 0;
    arr_chan_brd[21] = 0;
    arr_chan_brd[22] = 0;
    arr_chan_brd[23] = 1;
    arr_chan_brd[24] = 1;
    arr_chan_brd[25] = 1;
    arr_chan_brd[26] = 1;
    arr_chan_brd[27] = 1;
    arr_chan_brd[28] = 1;
    arr_chan_brd[29] = 0;
    arr_chan_brd[30] = 0;
    arr_chan_brd[31] = 0;
    arr_chan_brd[32] = 0;
    arr_chan_brd[33] = 0;
    arr_chan_brd[34] = 1;
    arr_chan_brd[35] = 1;
    arr_chan_brd[36] = 0;
    arr_chan_brd[37] = 1;
    arr_chan_brd[38] = 1;
    arr_chan_brd[39] = 1;
    arr_chan_brd[40] = 1;
    arr_chan_brd[41] = 1;
    arr_chan_brd[42] = 1;
    arr_chan_brd[43] = 0;
    arr_chan_brd[44] = 0;
    arr_chan_brd[45] = 0;
    arr_chan_brd[46] = 0;
    arr_chan_brd[47] = 0;
    arr_chan_brd[48] = 0;
    arr_chan_brd[49] = 0;
    arr_chan_brd[50] = 1;
    arr_chan_brd[51] = 1;
    arr_chan_brd[52] = 1;
    arr_chan_brd[53] = 1;
    arr_chan_brd[54] = 1;
    arr_chan_brd[55] = 1;
    arr_chan_brd[56] = 1;
    arr_chan_brd[57] = 1;
    arr_chan_brd[58] = 1;
    arr_chan_brd[59] = 0;
    arr_chan_brd[60] = 0;
    arr_chan_brd[61] = 0;
    arr_chan_brd[62] = 0;
    arr_chan_brd[63] = 0;
    arr_chan_brd[64] = 0;
    arr_chan_brd[65] = 0;
    arr_chan_brd[66] = 0;
    arr_chan_brd[67] = 0;
    arr_chan_brd[68] = 1;
    arr_chan_brd[69] = 1;
    arr_chan_brd[70] = 1;
    arr_chan_brd[71] = 1;
    arr_chan_brd[72] = 1;
    arr_chan_brd[73] = 1;
    arr_chan_brd[74] = 0;
    arr_chan_brd[75] = 0;
    arr_chan_brd[76] = 0;
    arr_chan_brd[77] = 0;
    arr_chan_brd[78] = 0;
    arr_chan_brd[79] = 0;

    for (i = 1; i <= TOTAL_NR_OF_CHANNELS; i++)
    {
        arr_chan_adr[i - 1] = arr_chan_adr[i];
        arr_chan_brd[i - 1] = arr_chan_brd[i];
    }

    unit_handles = (FT_HANDLE *)calloc(NR_OF_UNITS, sizeof(FT_HANDLE));
    arr_ser_num = (char *)calloc(NR_OF_UNITS * 256, sizeof(char));
    if (!unit_handles || !arr_ser_num)
    {
        printf("\nMemory allocation error\n");
        return FALSE;
    }

    FILE *f_ini;
    f_ini = fopen("sernum.ini", "rt");
    if (!f_ini)
    {
        printf("\nFile \"sernum.ini\" not found\n");
        return FALSE;
    }

    for (i = 0; i < NR_OF_UNITS; i++)
    {
        if (fgets(strg1, 256, f_ini) == NULL)
        {
            printf("\nInvalid file \"sernum.ini\"\n");
            return FALSE;
        }
        if (sscanf(strg1, "%s\n", arr_ser_num + i * 256) != 1)
        {
            printf("\nInvalid file \"sernum.ini\"\n");
            return FALSE;
        }
    }

    if ((FT_ListDevices(&ndevs, NULL, FT_LIST_NUMBER_ONLY) != FT_OK) || !ndevs)
    {
        printf("\nNo devices available\n");
        return FALSE;
    }

    for (i = 0; i < NR_OF_UNITS; i++)
    {
        for (j = 0; j < ndevs; j++)
        {
            // Get the serial number
            FT_ListDevices((PVOID)j, sn, FT_LIST_BY_INDEX | FT_OPEN_BY_SERIAL_NUMBER);
            bInit = 0;
            if (strcmp(sn, arr_ser_num + i * 256) == 0)
            {
                FT_STATUS fs = FT_Open(j, unit_handles + i);
                if (fs != FT_OK)
                { // Error handling
                    printf("\nError opening the device \"%s\"\n", arr_ser_num + i * 256);
                    for (k = 0; k < i; k++)
                        FT_Close(unit_handles[k]);
                    return FALSE;
                }
                memset(packet, 0, 130); // Fill the buffer with zeros
                unsigned long BR;
                FT_Write(unit_handles[i], packet, 130, &BR); // Send zeros to initialize the device
                bInit = 1;
                break;
            }
        }
        if (!bInit)
        { // Error handling
            for (k = 0; k < i; k++)
                FT_Close(unit_handles[k]);
            printf("\nDevice \"%s\" not found; please correct serial numbers in \"sernum.ini\".\n", arr_ser_num + i * 256);
            return FALSE;
        }
    }
    return TRUE;
}

void close_dac()
// Close DAC-40-USB control unit
{
    for (int i = 0; i < NR_OF_UNITS; i++)
        FT_Close(unit_handles[i]);
    free(unit_handles);
    free(arr_ser_num);
}

void set_mirror()
// Set voltages from the "voltage[]" array to the mirror
{
    int i, j;
    unsigned long BR = 0;

    for (i = 0; i < NR_OF_UNITS; i++)
    {
        memset(buf, 0, 40 * sizeof(WORD));
        for (j = 0; j < TOTAL_NR_OF_CHANNELS; j++)
        {
            if (arr_chan_brd[j] == i)
                buf[arr_chan_adr[j]] = voltage[j];
        }
        MakePacket(buf, packet);
        FT_Write(unit_handles[i], packet, 130, &BR);
    }
}
