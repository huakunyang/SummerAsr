#include "stdio.h"
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include <string.h>
#include <dirent.h>
#include "stdlib.h"
#include <ctime>
#include "extract_feat.h"
#include "stdint.h"
#include "asr.h"

int main(int argc, char * argv[])
{

    struct stat st;
    if(-1 == stat(argv[1], &st))
    {
        return -1 ;
    }

    FILE *fp = fopen(argv[1], "rb");
    if(!fp)
    {
        printf("Fail to open wav file:%s!\n", argv[4]);
        return -1;
    }
   
    fseek(fp, 44, SEEK_SET);

    float wavDur = (float)((st.st_size-44)/32000.0);

    int16_t * wavBuf = (int16_t *)malloc(st.st_size-44);
    fread(wavBuf,st.st_size-44, 1, fp);
    fclose(fp);

	struct timeval start, end;
    float timeuse,t;

    //omp_set_num_threads(6);

	gettimeofday( &start, NULL );

    void * asrData = asrInit("../model/am.model","../model/char.txt","../model/lm.model", false);

	gettimeofday( &end, NULL );
	timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
	std::cout<<"Model loading time costs:"<<std::setprecision(6)<<timeuse/1000000<<'s'<<"\n";

	gettimeofday( &start, NULL );

    std::string asrResult =  asrRun_without_vad(asrData, wavBuf, (((st.st_size)-44)/2));

	gettimeofday( &end, NULL );

    printf("Asr Result: %s\n",asrResult.c_str());

	timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
	std::cout<<"Wav duration: "<<std::setprecision(6)<<wavDur<< "s, Asr Decoding time costs: "<<std::setprecision(6)<<timeuse/1000000<<"s, RTF: "<<std::setprecision(6)<<(timeuse/1000000)/wavDur<<"\n";
}

