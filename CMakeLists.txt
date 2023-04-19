cmake_minimum_required(VERSION 3.5)

project(asr_test)

#set(CMAKE_CXX_FLAGS " -O3 -w -std=c++11 -fopenmp  -DNDEBUG -DKENLM_MAX_ORDER=6 ")
#set(CMAKE_C_FLAGS " -O3 -w -std=c++11 -fopenmp -DNDEBUG -DKENLM_MAX_ORDER=6 ")

set(CMAKE_CXX_FLAGS " -O3 -w -std=c++11  -DNDEBUG -DKENLM_MAX_ORDER=6 ")
set(CMAKE_C_FLAGS " -O3 -w -std=c++11 -DNDEBUG -DKENLM_MAX_ORDER=6 ")

add_executable(asr_test
               ./src/am/am.cpp
               ./src/asr/asr.cpp
               ./src/decoder/ctc_beam_search_decoder.cpp
               ./src/feat/extract_feat.cpp
               ./src/feat/hanning.cpp
               ./src/nn/nn.cpp
               ./src/decoder/scorer.cpp
               ./src/vad/vad.cpp
               ./src/vad/vad_internal_api.c
               ./src/lm/bhiksha.cc
               ./src/lm/binary_format.cc
               ./src/lm/config.cc
               ./src/lm/lm_exception.cc
               ./src/lm/model.cc
               ./src/lm/quantize.cc
               ./src/lm/read_arpa.cc
               ./src/lm/search_hashed.cc
               ./src/lm/search_trie.cc
               ./src/lm/sizes.cc
               ./src/lm/trie.cc
               ./src/lm/trie_sort.cc
               ./src/lm/value_build.cc
               ./src/lm/virtual_interface.cc
               ./src/lm/vocab.cc
               ./src/util/bit_packing.cc
               ./src/util/ersatz_progress.cc
               ./src/util/exception.cc
               ./src/util/file.cc
               ./src/util/file_piece.cc
               ./src/util/float_to_string.cc
               ./src/util/integer_to_string.cc
               ./src/util/mmap.cc
               ./src/util/murmur_hash.cc
               ./src/util/parallel_read.cc
               ./src/util/pool.cc
               ./src/util/read_compressed.cc
               ./src/util/scoped.cc
               ./src/util/spaces.cc
               ./src/util/string_piece.cc
               ./src/util/usage.cc
               ./src/util/double-conversion/bignum.cc
               ./src/util/double-conversion/bignum-dtoa.cc
               ./src/util/double-conversion/cached-powers.cc
               ./src/util/double-conversion/diy-fp.cc
               ./src/util/double-conversion/double-conversion.cc
               ./src/util/double-conversion/fast-dtoa.cc
               ./src/util/double-conversion/fixed-dtoa.cc
               ./src/util/double-conversion/strtod.cc
               ./src/librnnoise/celt_lpc.c
               ./src/librnnoise/denoise.c
               ./src/librnnoise/kiss_fft.c
               ./src/librnnoise/pitch.c
               ./src/librnnoise/rnn.c
               ./src/librnnoise/rnn_data.c
               ./test/main.cpp)

target_include_directories(asr_test PUBLIC
                           ./include
                           ./src
                           ./src/util
                           ./eigen-3.4.0)


               
               
