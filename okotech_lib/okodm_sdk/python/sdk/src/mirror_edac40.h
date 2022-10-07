#ifndef _MIRROR_EDAC40_H_
#define _MIRROR_EDAC40_H_

#include "edac40.h"

#define MAX_AMPLITUDE 0xFFFF // Maximum value that can be addressed

// Number of channels depending on defice type
#if (defined(MMDM_17TT) || defined(MMDM_LIN19CH) || defined(PIEZO_19CH) || defined(PIEZO_19LO_30) )
#define TOTAL_NR_OF_CHANNELS 19
#endif


#ifdef PIEZO_LO18CH
#define TOTAL_NR_OF_CHANNELS 18
#endif

#ifdef PIEZO_LIN20CH
#define TOTAL_NR_OF_CHANNELS 20
#endif

#if (defined(MMDM_37CH) || defined(PIEZO_37CH) || defined(PIEZO_37CH_2005) ||  defined(PIEZO_37CH_TRIHEX) ||\
     defined(PIEZO_37CH_50MM) || defined(PIEZO_37CH_50MM_2008))
#define TOTAL_NR_OF_CHANNELS 37
#endif

#ifdef MMDM_39CH_30MM
#define TOTAL_NR_OF_CHANNELS 39
#endif

#if (defined(MMDM_79CH_30MM) || defined(MMDM_79CH_40MM) || defined(MMDM_79CH_50MM) || defined(PIEZO_79CH_50MM))
#define TOTAL_NR_OF_CHANNELS 79
#endif

#ifdef PIEZO_109CH_50MM
#define TOTAL_NR_OF_CHANNELS 109
#endif

#if (defined(PIEZO_19CH) || defined(PIEZO_19LO_30) || defined(PIEZO_LO18CH) || defined(PIEZO_LIN20CH) || \
     defined(PIEZO_37CH) || defined(PIEZO_37CH_2005) ||  defined(PIEZO_37CH_TRIHEX) || defined(PIEZO_37CH_50MM) || defined(PIEZO_37CH_50MM_2008) || \
     defined(PIEZO_79CH_50MM) || PIEZO_109CH_50MM)
#define PIEZO 1
#else
#define MMDM 1
#endif

// Array of voltages addressed to the mirror, values in the range [0..MAX_AMPLITUDE]
extern unsigned int voltage[TOTAL_NR_OF_CHANNELS];

int init_dac();
void close_dac();
void set_mirror();
void print_pinout();
#endif // _MIRROR_EDAC40_H_
