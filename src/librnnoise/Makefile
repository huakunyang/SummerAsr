##TARGET := libvad.so
##RNNOISE := librnnoise.o
##LIBVAD := vad_api.o
#CC := gcc
##CFLAG := -WALL -fPIC -shared
##INCLUDE := rnnoise.h
##SRC = vad_api.c
##RNNOISE_SRC = librnnoise/celt_lpc.c \
##	librnnoise/denoise.c \
##	librnnoise/rnn.c \
##	librnnoise/rnn_data.c \
##	librnnoise/pitch.c \
##	librnnoise/kiss_fft.c
##RNNOISE_OBJ = celt_lpc.o \
##	denoise.o \
##	rnn.o \
##	rnn_data.o \
##	pitch.o \
##	kiss_fft.o
##
#RNNOISE_HEADERS = arch.h  \
#	celt_lpc.h  \
#	common.h  \
#	_kiss_fft_guts.h  \
#	kiss_fft.h  \
#	opus_types.h  \
#	pitch.h  \
#	rnn_data.h  \
#	rnn.h  \
#	tansig_table.h\
#	rnnoise.h
#RNNOISE_OBJ = celt_lpc.o \
#	denoise.o \
#	rnn.o \
#	rnn_data.o \
#	pitch.o \
#	kiss_fft.o
#	
#celt_lpc.o:
#	$(CC) -c  celt_lpc.c -o celt_lpc.o
#denoise.o:
#	$(CC) -c  denoise.c -o denoise.o	
#rnn.o:
#	$(CC) -c  rnn.c -o rnn.o		
#rnn_data.o:
#	$(CC) -c  rnn_data.c -o rnn_data.o		
#pitch.o:
#	$(CC) -c  pitch.c -o pitch.o		
#kiss_fft.o:
#	$(CC) -c  kiss_fft.c -o kiss_fft.o	
#				
#all:$(RNNOISE_OBJ)
#	ld -r -o librnnoise.o $(RNNOISE_OBJ)


	
SRCS = $(wildcard *.c)
OBJS = $(SRCS:.c=.o)

all:$(OBJS)
	ld -r -o librnnoise.o  $(OBJS) 
%.o:%.c
	$(CC)  -o $@ -c $<
#	
#
#$(RNNOISE_OBJ):%.o:%.c
#	$(CC) -c  $< -o $@
#all:
#		ld -r out.o $(RNNOISE_OBJ)
#  
#clean:
#	rm *.o
