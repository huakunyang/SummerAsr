#include <stdio.h>
#include <stdlib.h>
#include "librnnoise/rnnoise.h"

typedef enum _audio_format{
	SR16K_16BITS,
	SR48K_16BITS,
}AUDIO_FORMAT;

typedef enum _detect_mode{
	DETECT_LOW,
	DETECT_MIDDLE, 
	DETECT_HIGH,
}DETECT_MODE;

typedef enum _voice_mode{
	VOICE_VALID,
	VOICE_INVALID, 
	VOICE_ERROR,
}VOICE_MODE;

typedef enum _frame_size{
	SIZE_160B = 160,  //just support this mode when the sample rate is 16KHz
	SIZE_480B = 480,  //just support this mode when the sample rate is 48KHz
}FRAME_SIZE;

typedef struct _BtmVadHandle{
	DenoiseState *st;
	AUDIO_FORMAT audio_format;
	DETECT_MODE mode;
	float *SampRteExchage;
	float check_value;
}BtmVadHandle;

BtmVadHandle *vad_create(AUDIO_FORMAT format,DETECT_MODE mode);
VOICE_MODE vad_detect(short *buffer,BtmVadHandle *handle);
void vad_destory(BtmVadHandle *handle);

static void voice_samplerate_change(short *buffer,FRAME_SIZE frame_size,AUDIO_FORMAT audio_format,BtmVadHandle *handle);

BtmVadHandle *vad_create(AUDIO_FORMAT format,DETECT_MODE mode)
{
	BtmVadHandle *handle;
	

	handle = (BtmVadHandle*)malloc(sizeof(BtmVadHandle));
	handle->SampRteExchage = NULL;
	handle->st = NULL;
	handle->st = rnnoise_create();
	if(handle->st == NULL)
	{
		printf("init is error \n");
		free(handle);
		return 0;
	}
	handle->audio_format = format;
	handle->mode = mode;
	
	switch(handle->mode)
	{
		case DETECT_LOW:handle->check_value = 0.75;
				 break;
		case DETECT_MIDDLE:handle->check_value = 0.85;
			   break;
		case DETECT_HIGH:handle->check_value = 0.90;
				 break;
		default:
				 break;
	}
	if(handle->audio_format ==  SR16K_16BITS)
	{
		handle->SampRteExchage = malloc(480*sizeof(float));
	}
	return handle;
}
VOICE_MODE vad_detect(short *buffer,BtmVadHandle *handle)
{
	float vad_prob = 0;
	float check_value = 0.8;
	short *x = NULL;
	
	if(handle->audio_format ==  SR16K_16BITS)
	{		
		voice_samplerate_change(buffer,SIZE_160B,handle->audio_format,handle);
		x = handle->SampRteExchage;
		vad_prob = rnnoise_process_frame(handle->st, x, x);
	}
	else if(handle->audio_format ==  SR48K_16BITS)
	{
		x = buffer;
		vad_prob = rnnoise_process_frame(handle->st, x, x);
	}
	else
		return VOICE_ERROR;
	
	if(vad_prob>check_value)
		return VOICE_VALID;
	else
		return VOICE_INVALID;
}
void vad_destory(BtmVadHandle *handle)
{
	free(handle->st);
	free(handle->SampRteExchage);
	free(handle);
}

static void voice_samplerate_change(short *buffer,FRAME_SIZE frame_size,AUDIO_FORMAT audio_format,BtmVadHandle *handle)
{
	int i = 0;
	float *x = handle->SampRteExchage;
	short *y = buffer;
	for(i=0;i<frame_size;i++)
	{
		*x++ = *y;
		*x++ = *y;
		*x++ = *y++;
	}
}
