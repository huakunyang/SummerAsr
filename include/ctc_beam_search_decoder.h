//#ifndef CTC_BEAM_SEARCH_DECODER_H_
//#define CTC_BEAM_SEARCH_DECODER_H_

#include <vector>
#include <string>
#include <utility>
#include "scorer.h"

std::vector<std::pair<float, std::string> > 
           ctc_beam_search_decoder(std::vector<std::vector<float> > probs_seq,
					 												int beam_size,
																	std::vector<std::string> vocabulary,
																	int blank_id=0,
																	float cutoff_prob=1.0,
																	Scorer *ext_scorer=NULL
																	);
