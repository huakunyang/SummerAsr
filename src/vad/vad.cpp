#include "vad.h"
#include "vad_internal_api.h"
#include "stdlib.h"
#include "string.h"
 
#define VAD_START_SMOOTH_COUNT  10
#define VAD_END_SMOOTH_COUNT    30
#define VAD_FRAME_SIZE          (160*2)
#define VAD_PAD_LEN              2

typedef enum VAD_INTERNAL_STATE_t
{
    VAD_INTERNAL_OFF = 0,
    VAD_INTERNAL_ON
}VAD_INTERNAL_STATE_t;

typedef enum VAD_STATUS_t
{
    VAD_STATUS_OFF_DETECTED = 0,
    VAD_STATUS_ON_DETECTED,
    VAD_STATUS_TMP_ON_DETECTED,
    VAD_STATUS_TMP_OFF_DETECTED
}VAD_STATUS_t;

typedef struct VAD_SDK_PRIV_DATA_t
{
    VadHandle * Handle_;
    VAD_INTERNAL_STATE_t lastVadStatus_;
    VAD_STATUS_t curVadStatus_;
    VAD_RESULT_t vadResult_;
    int vadTmpPos_;
    int vadTmpSmoothCount_;
    int curAudioPos_;
}VAD_SDK_PRIV_DATA_t;

void * vad_sdk_init()
{
    VAD_SDK_PRIV_DATA_t * vadPrivData = (VAD_SDK_PRIV_DATA_t *)malloc(sizeof(VAD_SDK_PRIV_DATA_t));
    memset(vadPrivData, 0, sizeof(VAD_SDK_PRIV_DATA_t));
    
    vadPrivData->Handle_ = vad_create(SR16K_16BITS,DETECT_MIDDLE);
    if(vadPrivData->Handle_ == NULL)
    {
        free((void*)vadPrivData);
        return NULL;
    }
    
    vadPrivData->curVadStatus_ =  VAD_STATUS_OFF_DETECTED;
    vadPrivData->lastVadStatus_ = VAD_INTERNAL_OFF;
    vadPrivData->vadResult_ = VAD_OFF;
    return (void *)vadPrivData;
}

VAD_RESULT_t vad_sdk_run(void * vadData, char * audioData, int len, int * outPos)
{
    VAD_SDK_PRIV_DATA_t * vadPrivData = (VAD_SDK_PRIV_DATA_t *)vadData;

    if(vadPrivData == NULL)
    {
        return VAD_ERROR;
    }
    
    *outPos = -1; 

    int frameNum = len/VAD_FRAME_SIZE;
    for(int i = 0; i< frameNum; i++)
    {

        VOICE_MODE voiceMode = vad_detect((short*)(audioData + i*VAD_FRAME_SIZE), vadPrivData->Handle_);

        switch(vadPrivData->curVadStatus_)
        {
            case VAD_STATUS_OFF_DETECTED:
                if(voiceMode == VOICE_VALID)
                {
                    vadPrivData->curVadStatus_ = VAD_STATUS_TMP_ON_DETECTED;
                    vadPrivData->vadTmpSmoothCount_ += 1;
                }
                break; 

            case VAD_STATUS_ON_DETECTED:
                if(voiceMode == VOICE_INVALID)
                {
                    vadPrivData->curVadStatus_ = VAD_STATUS_TMP_OFF_DETECTED;
                    vadPrivData->vadTmpSmoothCount_ += 1;
                }
                break;

            case VAD_STATUS_TMP_ON_DETECTED:
                if(voiceMode == VOICE_VALID)
                {
                    vadPrivData->vadTmpSmoothCount_ += 1;
                    if(vadPrivData->vadTmpSmoothCount_ == VAD_START_SMOOTH_COUNT)
                    {
                        vadPrivData->curVadStatus_ = VAD_STATUS_ON_DETECTED;
                        vadPrivData->vadResult_ = VAD_ON;
                        *outPos = VAD_FRAME_SIZE*(vadPrivData->curAudioPos_ - (VAD_START_SMOOTH_COUNT+4*VAD_PAD_LEN));
                        if(*outPos < 0)
                        {
                           outPos = 0;
                        }
                        vadPrivData->vadTmpSmoothCount_ = 0;
                    }
                    
                }
                else if(voiceMode == VOICE_INVALID)
                {
                    vadPrivData->vadTmpSmoothCount_ = 0;
                    vadPrivData->curVadStatus_ = VAD_STATUS_OFF_DETECTED;
                    vadPrivData->vadResult_ = VAD_OFF;
                }
                break;

            case VAD_STATUS_TMP_OFF_DETECTED:
                if(voiceMode == VOICE_INVALID)
                {
                    vadPrivData->vadTmpSmoothCount_ += 1;
                    if(vadPrivData->vadTmpSmoothCount_ == VAD_END_SMOOTH_COUNT)
                    {
                        vadPrivData->curVadStatus_ = VAD_STATUS_OFF_DETECTED;
                        vadPrivData->vadResult_ = VAD_OFF;
                        *outPos = VAD_FRAME_SIZE*(vadPrivData->curAudioPos_ - (VAD_END_SMOOTH_COUNT-4*VAD_PAD_LEN));     
                        vadPrivData->vadTmpSmoothCount_ = 0;
                    }
                    
                }
                else if(voiceMode == VOICE_VALID)
                {
                    vadPrivData->vadTmpSmoothCount_ = 0;
                    vadPrivData->curVadStatus_ = VAD_STATUS_ON_DETECTED;
                    vadPrivData->vadResult_ = VAD_ON;
                }
                break;
        }
        vadPrivData->curAudioPos_++;
    }
    return vadPrivData->vadResult_;
}

void vad_sdk_destroy(void *vadData)
{
    VAD_SDK_PRIV_DATA_t * vadPrivData = (VAD_SDK_PRIV_DATA_t *)vadData;

    if(vadPrivData == NULL)
    {
        return ;
    }
    
    vad_destory(vadPrivData->Handle_);
    
    free(vadData);
}

