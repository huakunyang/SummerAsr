#ifdef __cplusplus
extern "C" {
#endif

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

typedef struct _VadHandle VadHandle;

VadHandle *vad_create(AUDIO_FORMAT format,DETECT_MODE mode);
VOICE_MODE vad_detect(short *buffer,VadHandle *handle);
void vad_destory(VadHandle *handle);

#ifdef __cplusplus
}
#endif
