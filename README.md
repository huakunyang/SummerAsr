# SummerAsr 用于纪念2023年即将到来和终将逝去的夏天

# 说明
- SummerAsr 是一个独立编译的大范围连续语音识别系统(ASR)。
- SummerAsr 的底层计算库使用Eigen，Eigen是一套模板定义的函数，大部分情况下，只需要包含头文件即可，所以本项目基本上没有其他依赖，在C++环境下可以独立编译和运行。
- 本项目使用Eigen提供的矩阵库实现了神经网络的算子，不需要依赖其他NN运行环境，例如pytorch，tensorflow 等。
- 本项目在 Ubuntu 上编译运行通过，其他类Linux平台，如Android，树莓派等，也应该没啥大问题，在Window上没有测试过，可能需要少许改动。

# 使用说明
- 将本项目的代码克隆到本地，最好是Ubuntu Linux 环境
- 从以下的百度网盘地址下载模型，放入本项目的model目录中：  
  链接: https://pan.baidu.com/s/13KgAaD79Pd3XsWI6k6VViw?pwd=y4rd 提取码: y4rd

  目录结构和内容如下：  
   model  
   ├── am.model  
   ├── char.txt  
   └── lm.model  
	  
- 进入Build 目录，执行以下命令：
  cmake ..  
  make 

- 编译完成后，会在Build 目录中生成 asr_test 执行程序
- 运行下列命令，测试语音识别：  
  ./asr_test ../wavSamples/test.wav
- 正常情况下，可得到如下结果

  Model loading time costs:0.236086s  
  Asr Result: 今天是二零二二年五月四号现在是晚上十一点五十三分  
  Wav duration: 9.25s, Asr Decoding time costs: 3.35407s, RTF: 0.362602  

# 后续开发
- 后续将开放模型训练和转化脚本

# 联系作者
- 有进一步的问题或需要可以发邮件到 120365182@qq.com , 或添加微信: hwang_2011, 本人尽量回复。

# 感谢
本项目在源代码和算法方面使用了下列方案，在此表示感谢, 若可能引发任何法律问题，请及时联系我协调解决
- Eigen 
- rnnoise (https://github.com/xiph/rnnoise)
- MASR(https://github.com/yeyupiaoling/MASR)
- KenLM (https://github.com/kpu/kenlm)
- CTC beam search decoder(https://github.com/Sundy1219/ctc_beam_search_lm)

