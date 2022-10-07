ALL : am_set rotate set_channel ring1_set ring2_set ring3_set ring4_set ring5_set ring6_set ring7_set


CLEAN :
	-@erase am_set.exe
	-@erase rotate.exe
	-@erase set_channel.exe
    -@erase ring1_set.exe
    -@erase ring2_set.exe
    -@erase ring3_set.exe
    -@erase ring4_set.exe
    -@erase ring5_set.exe
    -@erase ring6_set.exe
    -@erase ring7_set.exe

am_set :
	cl.exe -o am_set.exe am_set.cpp ftd2xx.lib
        -@erase am_set.obj

rotate :
	cl.exe -o rotate.exe rotate.cpp ftd2xx.lib
        -@erase rotate.obj

set_channel :
	cl.exe -o set_channel.exe set_channel.cpp ftd2xx.lib
        -@erase set_channel.obj

ring1_set :
	cl.exe -o ring1_set.exe ring1_set.cpp ftd2xx.lib
        -@erase ring1_set.obj
        
ring2_set :
	cl.exe -o ring2_set.exe ring2_set.cpp ftd2xx.lib
        -@erase ring2_set.obj        
ring3_set :
	cl.exe -o ring3_set.exe ring3_set.cpp ftd2xx.lib
        -@erase ring3_set.obj        
ring4_set :
	cl.exe -o ring4_set.exe ring4_set.cpp ftd2xx.lib
        -@erase ring4_set.obj        
ring5_set :
	cl.exe -o ring5_set.exe ring5_set.cpp ftd2xx.lib
        -@erase ring5_set.obj       
ring6_set :
	cl.exe -o ring6_set.exe ring6_set.cpp ftd2xx.lib
        -@erase ring6_set.obj       
ring7_set :
	cl.exe -o ring7_set.exe ring7_set.cpp ftd2xx.lib
        -@erase ring7_set.obj