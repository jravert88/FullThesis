# Last Modified: 021211
# 021211 Change: Moved to using dll_source and compiling with Windows DLL and Linux static library

LIB = .
APPINC= ./_AppSource
#MASSINC = AppDll.o
#GLITCH_MASSINC = AppDll.o

# load_svf, test_checksum are not commonly rebuilt, not included in make all
# acquire memtest  da digosc
#BUILDALL = acquire memtest digosc TestChecksum da
BUILDALL = acquire glitchtest_main memtest digosc calibrate

all: $(BUILDALL)



acquire : 	acquire.o
		g++ acquire.o  $(APPINC)/AcqSynth.a -o acquire -lrt
		# acquire built

acquire.o:      $(APPINC)/acquire.cpp 
		g++ -c $(APPINC)/acquire.cpp



memtest : 	memtest.o $(MASSINC)
		g++ memtest.o $(MASSINC) $(APPINC)/AcqSynth.a -o memtest -lrt
		# memtest built

memtest.o:      $(APPINC)/memtest.cpp
		g++ -c $(APPINC)/memtest.cpp


#TestChecksum : 	TestChecksum.o $(MASSINC)
#		g++ TestChecksum.o $(MASSINC) $(APPINC)/AcqSynth.a -o TestChecksum
#		# TestChecksum built

#TestChecksum.o:	$(APPINC)/TestChecksum.cpp
#		g++ -c $(APPINC)/TestChecksum.cpp

glitchTest.o:	$(APPINC)/glitchTest.cpp
		g++ $(CFLAGS) -c $(APPINC)/glitchTest.cpp
		# glitchTest built

glitchtest_main : glitchtest_main.o glitchTest.o
		g++ glitchtest_main.o glitchTest.o $(APPINC)/AcqSynth.a -o glitchTest -lrt
		# glitchtest built

glitchtest_main.o:	$(APPINC)/glitchtest_main.cpp
		g++ -c $(APPINC)/glitchtest_main.cpp

#calibrate : 	calibrate.o calcStats.o
#		g++ calibrate.o calcStats.o $(APPINC)/AcqSynth.a -o calibrate -lrt
		# calibrate built

#calibrate.o:	$(APPINC)/calibrate.cpp
#		g++ -c $(APPINC)/calibrate.cpp

#calcStats.o:	$(APPINC)/calcStats.cpp
#		g++ -c $(APPINC)/calcStats.cpp



#digosc7 : 	digosc7.o $(MASSINC)
#		g++ $(CFLAGS) digosc7.o $(MASSINC) $(APPINC)/AcqSynth.a `pkg-config --libs gtk+-2.0` -o digosc7
		# digosc7 built

#digosc7.o:      $(APPINC)/digosc7.c
#		g++ $(CFLAGS) -c  `pkg-config --cflags gtk+-2.0` $(APPINC)/digosc7.c


# Stuff below here is unfinished

#da : 		da.o $(MASSINC) 
#		g++ da.o $(MASSINC) $(APPINC)/AcqSynth.a -o da		
#		# da built

load_svf : 	load_svf.o 
		g++ load_svf.o $(LIB)/libuvdma_user.a -lm -o load_svf



#board_cal : 	board_cal.o $(MASSINC)
#		g++ board_cal.o $(MASSINC) $(LIB)/libuvdma_user.a -o board_cal
		# Board Cal Built


#da.o:      $(APPINC)/da.cpp 
#		g++ -c $(APPINC)/da.cpp 






clean: 
	rm -rf $(BUILDALL) *.o

cleandat: 
	rm -rf underover.dat checksum.dat uvdma.dat uvdma1.dat uvdma2.dat uvdma3.dat


