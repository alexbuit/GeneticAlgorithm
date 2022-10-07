ALL : am_set rotate set_channel ring1_set


CLEAN :
	-@erase am_set.exe
	-@erase rotate.exe
	-@erase set_channel.exe
    -@erase ring1_set.exe
    -@erase ring2_set.exe

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
