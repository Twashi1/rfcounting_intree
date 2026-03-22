TARGET = mcpat
SHELL = /bin/sh
.PHONY: all depend clean
.SUFFIXES: .c .cc .o
DEPS = const.h 

ifndef NTHREADS
  NTHREADS = 4
endif


LIBS = 
INCS = -lm

ifeq ($(TAG),dbg)
  DBG = -Wall 
  OPT = -ggdb -g -O0 -DNTHREADS=1 -Icacti -Iorion
else
  DBG = 
  OPT = -O3 -msse2 -mfpmath=sse -DNTHREADS=$(NTHREADS) -Icacti -Iorion
  #OPT = -O0 -DNTHREADS=$(NTHREADS)
endif

#CXXFLAGS = -Wall -Wno-unknown-pragmas -Winline $(DBG) $(OPT) 
CXXFLAGS = -Wno-unknown-pragmas $(DBG) $(OPT) 
CXX = g++ -m64
CC  = gcc -m64

CFLAGS = -Iorion -g 

VPATH = cacti orion 

SRCS  = \
  Ucache.cc \
  XML_Parse.cc \
  arbiter.cc \
  area.cc \
  array.cc \
  bank.cc \
  basic_circuit.cc \
  basic_components.cc \
  cacti_interface.cc \
  component.cc \
  core.cc \
  crossbar.cc \
  decoder.cc \
  htree2.cc \
  interconnect.cc \
  io.cc \
  iocontrollers.cc \
  logic.cc \
  main.cc \
  mat.cc \
  memoryctrl.cc \
  noc.cc \
  nuca.cc \
  parameter.cc \
  processor.cc \
  router.cc \
  sharedcache.cc \
  subarray.cc \
  technology.cc \
  uca.cc \
  wire.cc \
  xmlParser.cc \
  powergating.cc \


OSRCS = \
  SIM_router.c \
  SIM_arbiter.c \
  SIM_crossbar.c \
  SIM_router_power.c \
  SIM_link.c \
  SIM_clock.c \
  SIM_router_area.c \
  SIM_array_l.c \
  SIM_array_m.c \
  SIM_cam.c \
  SIM_ALU.c \
  SIM_misc.c \
  SIM_permu.c \
  SIM_static.c \
  SIM_util.c \
  SIM_time.c \
  #orion_router_power.c \
  #orion_router_area.c \
  #orion_link.c \

OBJS = $(patsubst %.cc,obj_$(TAG)/%.o,$(SRCS))

OBJSORION = $(patsubst %.c,obj_$(TAG)/%.o,$(OSRCS)) 

#OBJS = $(OBJSPRE) $(OBJSORION)

all: obj_$(TAG)/$(TARGET)
	cp -f obj_$(TAG)/$(TARGET) $(TARGET)

obj_$(TAG)/$(TARGET) : $(OBJS) $(OBJSORION)
	$(CXX) $(OBJS) $(OBJSORION) -o $@ $(INCS) $(CXXFLAGS) $(LIBS) -pthread 
	#$(CC) $(OBJSORION) -o $@ $(INCS) $(CXXFLAGS) $(LIBS) -pthread 
	
obj_orion : $(OBJSORION) 	
	$(CC) $(OBJSORION) -o $@ $(INCS) $(CXXFLAGS) $(LIBS) -pthread 
	
#obj_$(TAG)/%.o : %.cc
#	$(CXX) -c $(CXXFLAGS) $(INCS) -o $@ $<

obj_$(TAG)/%.o : %.cc $(DEPS)
	$(CXX) $(CXXFLAGS) -c $< -o $@
	
obj_$(TAG)/%.o : %.c
	$(CC) $(CXXFLAGS) -c $< -o $@

clean:
	-rm -f *.o $(TARGET)


