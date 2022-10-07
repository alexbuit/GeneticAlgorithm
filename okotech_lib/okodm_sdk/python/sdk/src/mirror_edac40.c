/***************************************************
 File: edac40mirror.c
 Description: Mirror control functions

 (c) 2004-2013 FlexibleOptical
          Gleb Vdovin,
          Mikhail Loktev,
          Oleg Soloviev,
          Seva Patlan
****************************************************/

#include "mirror_edac40.h" // defines TOTAL_NR_OF_CHANNELS which is visible to the main program
#include "edac40.h"
#include <stdio.h>
#include <string.h>

#define MIRROR_EDAC40_MAXN 50 // maximum expected number of EDAC40 units within the network
#define MIRROR_EDAC40_USE_TCP 0 // 0 -- for datagrams (UPD), 1 -- for stream (TCP)

// default initial values, different values can be given for different mirror configurations
#define MIRROR_EDAC40_DEFAULT_GLOBAL_OFFSET 0       // 0V, (negative, 14bits)
#define MIRROR_EDAC40_DEFAULT_OFFSET        0x8000  // -2^15 gives 0V
#define MIRROR_EDAC40_DEFAULT_GAIN          0xFFFF  // full span=12V
#define MIRROR_EDAC40_DEFAULT_VALUE         0       // initial value=0

// Number of units utilized for mirror control
#if (defined(MMDM_LIN19CH) || defined(MMDM_37CH) || defined(MMDM_39CH_30MM) || defined(PIEZO_LIN20CH) || \
     defined(PIEZO_19CH) || defined(PIEZO_37CH) || defined(PIEZO_37CH_TRIHEX) || defined(PIEZO_37CH_2005) || defined(PIEZO_37CH_50MM) || \
     defined(PIEZO_37CH_50MM_2008) || (defined(PIEZO_LO18CH)) || defined(PIEZO_19LO_30) || defined(MMDM_17TT))
int unit[TOTAL_NR_OF_CHANNELS];
#define NR_OF_UNITS 1
#endif

#if (defined(MMDM_79CH_30MM) || defined(MMDM_79CH_40MM) || defined(MMDM_79CH_50MM) || defined(PIEZO_79CH_50MM))
#define NR_OF_UNITS 2
#endif

// Table of correspondence between the order numbers of the mirror's actuators and
// the order numbers of outputs of EDAC40 unit
// Defined according to the mirror type.
// Multi-unit configurations contain unit indices as well.

#ifdef PIEZO_LO18CH
int addr[TOTAL_NR_OF_CHANNELS]={19,13,17,16,8,4,7,2,14,20,11,15,18,6,5,9,3,12};
#define NR_OF_UNITS 1
#endif

#ifdef MMDM_LIN19CH
int addr[TOTAL_NR_OF_CHANNELS]={1,3,2,5,4,7,6,9,8,10,11,12,13,14,15,16,17,18,19};
#endif

#ifdef MMDM_17TT
int addr[TOTAL_NR_OF_CHANNELS]={1,12,5,13,19,10,6,4,8,3,7,9,11,15,17,16,14,2,18};
#endif

#ifdef MMDM_37CH
int addr[TOTAL_NR_OF_CHANNELS]={10,18,13,9,5,6,14,19,17,12,11,1,3,2,4,7,8,15,16,
                                34,36,37,38,35,23,25,21,22,24,26,27,28,29,30,31,32,33};
#endif

#ifdef MMDM_39CH_30MM
int addr[TOTAL_NR_OF_CHANNELS]={33, 3, 2, 6,27,26,31,37, 7,11,13,20,25,30,34,38, 5, 8,12,17,
                                21,23,29,35,1,4,9,10,15,14,16,19,18,22,24,28,32,36,39};
#endif

#ifdef MMDM_79CH_30MM
int addr[TOTAL_NR_OF_CHANNELS]={38,17,39,5,24,18,14,10,31,33,36,3,6,9,29,25,20,19,13,9,
                                6,26,28,35,1,4,11,13,32,31,26,23,12,8,4,5,25,27,30,37,
                                2,8,12,14,37,34,30,27,21,16,11,7,2,3,21,22,24,29,32,0,
                                7,10,17,16,19,39,36,35,28,22,1,20,23,34,15,18,38,33,15};

int unit[TOTAL_NR_OF_CHANNELS]={1,0,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,
                                0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,
                                1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,
                                1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,0,0,0};
#endif

#ifdef MMDM_79CH_40MM
int addr[TOTAL_NR_OF_CHANNELS]={0,36,21,6,11,27,33,28,1,3,7,16,22,30,14,15,6,38,34,29,
                                25,2,8,13,17,21,29,32,37,18,10,4,35,30,26,20,5,12,15,19,
                                24,31,35,36,16,13,9,5,3,37,32,27,23,22,4,10,14,18,20,25,
                                26,33,34,39,19,12,11,7,2,1,39,31,24,9,23,28,38,17,8};

int unit[TOTAL_NR_OF_CHANNELS]={1,0,0,1,1,1,0,0,1,1,1,1,1,1,0,0,0,0,0,0,
                               0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,
                               1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,
                               1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0};
#endif

#ifdef MMDM_79CH_50MM
int addr[TOTAL_NR_OF_CHANNELS]={7,9,24,32,31,2,3,10,15,26,33,36,19,15,23,28,37,5,36,17,
                                21,28,34,3,5,10,13,17,22,26,33,38,6,13,16,23,24,31,35,39,
                                1,9,7,12,16,20,27,34,39,4,8,14,18,20,25,27,29,30,37,0,
                                2,4,6,11,14,21,25,30,35,11,12,19,22,32,38,8,18,29,1};

int unit[TOTAL_NR_OF_CHANNELS]={0,0,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,0,1,0,
                               0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,
                               1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,
                               1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,0};
#endif

#ifdef PIEZO_LIN20CH
int addr[TOTAL_NR_OF_CHANNELS]={0,3,5,7, 9,11,13,15,17,19,1,2,4,6, 8,10,12,14,16,18};
//	                            6,4,2,0,14,12,10, 8,22,20,7,5,3,1,15,13,11, 9,23,21};
#endif

#ifdef PIEZO_19CH
int addr[TOTAL_NR_OF_CHANNELS]={7,12,4,3,9,11,18,10,8,6,2,1,5,13,15,17,19,16,14};
#endif

#ifdef PIEZO_19LO_30
int addr[TOTAL_NR_OF_CHANNELS]={8,19,13,17,14,6,2,7,1,12,18,11,15,16,4,5,9,3,10};
#endif

#ifdef PIEZO_37CH
int addr[TOTAL_NR_OF_CHANNELS]={10,13,14,16,8,5,6,11,15,9,18,19,17,12,4,1,3,2,7,
                                33,34,35,36,37,38,27,26,25,24,23,21,22,28,29,30,32,31};
#endif

#ifdef PIEZO_37CH_2005
int addr[TOTAL_NR_OF_CHANNELS]={10,4,12,36,29,23,3,6,14,19,17,34,38,31,25,22,7,5,1,8,
                                16,18,15,13,11,30,32,37,33,27,21,24,26,28,9,35,2};
#endif

#ifdef PIEZO_37CH_TRIHEX
int addr[TOTAL_NR_OF_CHANNELS]={14,16,38,23,26,1,8,10,13,37,18,36,25,28,22,7,5,2,6,
								19,17,32,15,11,35,34,39,33,31,21,27,29,24,9,3,4,12};
#endif


#ifdef PIEZO_37CH_50MM
int addr[TOTAL_NR_OF_CHANNELS]={1,4,7,6,3,2,5,17,16,19,18,9,8,11,10,13,12,15,14,
								35,34,37,36,39,21,23,22,25,24,27,26,29,28,31,30,33,32};
#endif

#ifdef PIEZO_37CH_50MM_2008
int addr[TOTAL_NR_OF_CHANNELS]={11,7,13,14,12,6,4,9,16,17,18,19,15,10,8,3,1,5,2,
                                28,30,33,31,35,36,37,38,34,32,21,23,27,25,22,29,24,26};
#endif

#ifdef PIEZO_79CH_50MM
int addr[TOTAL_NR_OF_CHANNELS]={19,12,24,23,27,15,17,10,6,28,31,20,25,29,31,9,10,17,19,13,
                                11,5,30,32,33,25,23,21,26,32,33,34,7,6,12,14,16,14,8,9,
                                2,1,38,37,34,29,26,22,20,24,30,35,38,36,0,1,5,8,11,16,
                                15,18,7,4,3,39,35,36,27,21,22,28,37,39,2,3,4,13,18};

int unit[TOTAL_NR_OF_CHANNELS]={1,0,0,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,
                               0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,
                               0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,
                               0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1};
#endif

#ifdef PIEZO_109CH_50MM
#define NR_OF_UNITS 3
int addr[TOTAL_NR_OF_CHANNELS]={8,15,34,26,24,7,10,16,36,1,5,32,31,22,1,39,38,6,9,18,
                                38,31,35,8,4,2,33,30,21,2,4,8,32,30,36,2,11,19,39,28,
                                24,22,13,15,10,6,39,35,28,25,3,9,13,11,28,27,26,33,37,4,
                                13,17,33,37,26,21,27,25,12,17,14,11,3,37,36,29,27,5,7,12,
                                15,14,16,25,21,23,31,35,3,5,14,34,32,30,23,18,16,9,7,38,
                                23,6,10,17,19,22,29,34,1};

int unit[TOTAL_NR_OF_CHANNELS]={2,2,0,0,0,2,2,2,1,1,1,0,0,0,0,2,2,2,2,2,
                                1,1,1,1,1,1,0,0,0,0,0,0,2,2,2,2,2,2,1,1,
                                1,1,1,1,1,1,0,0,0,0,0,0,0,0,2,2,2,2,2,2,
                                2,2,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,
                                0,0,0,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,0,
                                0,0,0,0,0,2,2,2,2};
#endif

// Array of voltages addressed to the mirror, values in the range [0..MAX_AMPLITUDE]
// Global variable used in the main program
unsigned int voltage[TOTAL_NR_OF_CHANNELS];

// One TCP/UDP socket per each EDAC40 unit
SOCKET edac40_socket[NR_OF_UNITS];

char *sernum_mac[NR_OF_UNITS]; // MAC addresses of EDAC40 units to be used.
//Will be read in from the sernum.ini file

char *sernum_ip[NR_OF_UNITS]; // IP addresses corresponding to sernums

int init_dac()
{
   int i,j;
   edac40_list_node edac40_list[MIRROR_EDAC40_MAXN];
   FILE* f_ini;
   char strg1[256];
   int units_num; // total number of units found
   int units_sernums; // number of devices read from sernum file
   // Allocate memory for storing of serial numbers
   for(i=0; i<NR_OF_UNITS; i++)
         {
             sernum_mac[i]=(char *)malloc(20);
             sernum_ip[i]=(char *)malloc(20);
         }
   // get the list of EDAC40 MAC-address from sernum.ini text file.
   // One MAC address in the form of XX-XX-XX-XX-XX-XX per line.
   units_sernums=0;
   f_ini = fopen("sernum.ini", "r");
    if (!f_ini)
     {
        printf("\nFile \"sernum.ini\" not found\n");
        if(NR_OF_UNITS>1) return 0;
              else printf("Warning: first found DAC unit will be used.\n");
        // configuration file is mandatory in the case of multi-unit mirror configurations
     }
    else
    {
        for (i=0; i<NR_OF_UNITS; i++)
        {
            if (fgets(strg1, 256, f_ini)==NULL)
            {
               printf("\nInvalid file \"sernum.ini\"\n");
               if(NR_OF_UNITS>1) return 0;
            }
            //printf("\"%s\"\n",strg1);
            if (sscanf(strg1, "%s\n", sernum_mac[i])!=1)
            {
               printf("\nInvalid file \"sernum.ini\"\n");
               if(NR_OF_UNITS>1) return 0;
            }
        }
        units_sernums=i;
        fclose(f_ini);
    } // reading sernum file
   // for(i=0; i<NR_OF_UNITS; i++) printf("\"%s\"\n",sernum_mac[i]);
   // initialize WSock machinery...
   edac40_init();
   // Look for available EDAC40 devices...
   if((units_num=edac40_list_devices(edac40_list, MIRROR_EDAC40_MAXN, 1, 100))>=NR_OF_UNITS)
        {  // yes, there are enough devices present in the network
           if((NR_OF_UNITS==1) && (units_sernums<1)) // no sernum file for single-module configuration found
              {
                 // if only one device required grab the first available
                 // as there is no sernum file in this case
                 strcpy(sernum_ip[0],edac40_list[0].IPAddress);
                 // all logical channels belongs to the same 0 unit
                 for(i=0; i<TOTAL_NR_OF_CHANNELS; i++) unit[i]=0;
              }
            else
              {
                 // loop on MACs loaded from sernum file and try to find them among those really present
                 for(i=0; i<NR_OF_UNITS; i++)
                    {
                       for(j=0; j<units_num; j++)
                         {
                           // MAC address contains six bytes, represented in the form XX-XX-XX-XX-XX-XX,
                           // total 17 characters
                           // printf("\n%s<->%s\n",edac40_list[j].MACAddress,sernum_mac[i]);
                           if(strncmp(edac40_list[j].MACAddress,sernum_mac[i], 17)==0) // exact match, take it...
                              {
                                 strcpy(sernum_ip[i],edac40_list[j].IPAddress);
                                 break;
                              }
                         }
                       if(j==units_num) // end of list reached
                            {
                               printf("\nCoudn't find the device with MAC address:%s.\n",sernum_mac[i]);
                               return 0;
                            }
                    }

              }

          for(i=0; i<NR_OF_UNITS; i++)
             {
               // connect to the unit
               if((edac40_socket[i]=edac40_open(sernum_ip[i],MIRROR_EDAC40_USE_TCP))<0)
                  {  // invalid (negative) socket descriptor meaning some problem
                     printf("\nError connecting to the device %s (%s).\n",sernum_ip[i],sernum_mac[i]);
                     return 0;
                  }
               // set TCP time-out for blocking operations to 1s
               edac40_set_timeout(edac40_socket[i],1000);
               /* _DO_NOT_ set defaults any more, they are loaded from NVRAM!
               // set offset and gain so the output range would be 0..12V
               edac40_set(edac40_socket[i],EDAC40_SET_OFFSET_DACS,0,MIRROR_EDAC40_DEFAULT_GLOBAL_OFFSET);
               // set defaults to all channels, not only used in the current configuration
               for(j=0; j<40; j++)
                 {
                    // restore some reasonable defaults
                    edac40_set(edac40_socket[i],EDAC40_SET_GAIN,j,MIRROR_EDAC40_DEFAULT_GAIN);
                    //edac40_set(edac40_socket[i],EDAC40_SET_OFFSET,j,MIRROR_EDAC40_DEFAULT_OFFSET); // FIXME ????
  				    edac40_set(edac40_socket[i],EDAC40_SET_VALUE,j,MIRROR_EDAC40_DEFAULT_VALUE);
                 }
               */
             } // NR_OF_UNITS
          }  // Enough devices
      else
        {
           printf("\nNot enough device(s):");
           printf(" %d required, %d found.\n", NR_OF_UNITS, units_num);
           return 0;
        }
   // Don't need them any more...
   for(i=0; i<NR_OF_UNITS; i++)
         {
             free(sernum_mac[i]);
             free(sernum_ip[i]);
         }
   return 1;
}

void close_dac()
{
    int i;
    // close socket(s)
    for(i=0; i<NR_OF_UNITS; i++) edac40_close(edac40_socket[i]);
    // terminate WinSock mechanism
    edac40_finish();
}

void set_mirror()
{
    int i, j, buf_len;
    edac40_channel_value dac_data[NR_OF_UNITS][40];
    int tail[NR_OF_UNITS];
    char *buf;
    // prefill dac_data array with default value, to keep unused channels at bay
    // this is not really necessary
    for(i=0; i<NR_OF_UNITS; i++)
       {
         for(j=0; j<40; j++)
           {
             dac_data[i][j].channel=j;
             dac_data[i][j].value=MIRROR_EDAC40_DEFAULT_VALUE;
           }
         tail[i]=0;
       }
    for(i=0; i<TOTAL_NR_OF_CHANNELS; i++)
       {  // distribute the data according to unit and channel addresses
           (dac_data[unit[i]][tail[unit[i]]]).channel=addr[i];
           (dac_data[unit[i]][tail[unit[i]]]).value=voltage[i];
           tail[unit[i]]++;
       }
    for(i=0; i<NR_OF_UNITS; i++)
       {


           // prepare packet and send data belonging to that particular unit
		  // only those that were filled in for this unit
           buf_len=edac40_prepare_packet(dac_data[i],tail[i],&buf); // Second arg is tail[i], not 40 as it was. 9-11-11 O.S.
          edac40_send_packet(edac40_socket[i],buf,buf_len);
       }
    free(buf);
}

void print_pinout()
{
    int unit_idx,pin_idx,conn_idx,i;
    char chan_id1[256],chan_id2[256];
    for(unit_idx=0; unit_idx<NR_OF_UNITS; unit_idx++)
      {
         printf("--- Unit %d ---\n",unit_idx+1);
         for(conn_idx=0; conn_idx<2; conn_idx++)
            {
              printf("Connector %d:\n",conn_idx+1);
              for(pin_idx=0; pin_idx<10; pin_idx++)
                 {
                    strcpy(chan_id1," NC");
                    strcpy(chan_id2," NC");
                    for(i=0; i<TOTAL_NR_OF_CHANNELS; i++)
                      {
                         if(unit_idx==unit[i] && pin_idx*2+conn_idx*20==addr[i]) sprintf(chan_id1,"%3d",i+1);
                         if(unit_idx==unit[i] && pin_idx*2+1+conn_idx*20==addr[i]) sprintf(chan_id2,"%3d",i+1);
                      }
                    printf("%7s %2s %2s\n",(pin_idx==0)?"pin 1->":" ", chan_id1, chan_id2);
                 }
            }
      }
}
