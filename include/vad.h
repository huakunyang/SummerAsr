#ifndef VAD_SDK_H
#define VAD_SDK_H

typedef enum VAD_RESULT_t
{
    VAD_OFF = 0,
    VAD_ON  = 1,
    VAD_ERROR
}VAD_RESULT_t;

void * vad_sdk_init();
VAD_RESULT_t vad_sdk_run(void * vadPrivData, char * audioData, int len, int * outPos);
void vad_sdk_destroy(void *vadPrivData);

#endif
