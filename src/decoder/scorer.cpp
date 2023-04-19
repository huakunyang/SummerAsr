#include <iostream>

#include "scorer.h"
#include "lm/model.hh"
//#include "util/tokenize_piece.hh"
//#include "util/string_piece.hh"
#include <string.h>

using namespace lm::ngram;


Scorer::Scorer(float alpha, float beta, std::string lm_model_path) {
    this->_alpha = alpha;
    this->_beta = beta;
    this->_language_model = new Model(lm_model_path.c_str());
}

Scorer::~Scorer(){
   delete (Model *)this->_language_model;
}


inline void strip(std::string &str, char ch=' ') {
    if (str.size() == 0) return;
    int start  = 0;
    int end = str.size()-1;
    for (int i=0; i<str.size(); i++){
        if (str[i] == ch) {
            start ++;
        } else {
            break;
        }
    }
    for (int i=str.size()-1; i>=0; i--) {
        if  (str[i] == ch) {
            end --;
        } else {
            break;
        }
    }
    
    if (start == 0 && end == str.size()-1) return;
    if (start > end) {
        std::string emp_str;
        str = emp_str;
    } else {
        str = str.substr(start, end-start+1);
    }
}


int Scorer::word_count(std::string sentence) {
    strip(sentence);
    int cnt = 0;
    for (int i=0; i<sentence.size(); i++) {
        if (sentence[i] == ' ' && sentence[i-1] != ' ') {
            cnt ++;
        }
    }
    if (cnt > 0) cnt ++;
		if (cnt==0) cnt++;
    return cnt;
}


inline void split(const std::string &s,char delimiter,std::vector<std::string> &v)
{
	std::string::size_type i = 0;
	std::string::size_type j = s.find(delimiter);
	while(j != std::string::npos)
	{
		v.push_back(s.substr(i,j-i));
		i = ++j;
		j = s.find(delimiter,i);
	}
	if(j == std::string::npos)
		v.push_back(s.substr(i,s.length()));
}


float Scorer::language_model_score(std::string sentence) {
    Model *model = (Model *)this->_language_model;
    State state, out_state;
    lm::FullScoreReturn ret;
    state = model->BeginSentenceState();
		
		std::vector<std::string> word;
		split(sentence,' ',word);
		for(std::vector<std::string>::iterator it=word.begin();it!=word.end();it++)
		{
        lm::WordIndex vocab = model->GetVocabulary().Index(*it);
        ret = model->FullScore(state, vocab, out_state);
        state = out_state;
    }
    float score = ret.prob; 
    
    return  pow(10, score);
}


float Scorer::get_score(std::string sentence) {
    float lm_score = language_model_score(sentence);
    int word_cnt = word_count(sentence);
		float final_score;
    //if (log)
		final_score = _alpha*std::log(lm_score) + _beta*std::log(word_cnt);
		//else
//    final_score = pow(lm_score, _alpha) * pow(word_cnt, _beta);
    
		return final_score;
}


/*int main(){
   Scorer *p_score=new Scorer(1.0,1.5,"train.binary");
   //std::string s="金 林 饭 店";
   std::string s="一 九";
	 bool is_log = true;
   float lm_score=p_score->language_model_score(s);
   float final_score=p_score->get_score(s);
   std::cout<<"The score of last word of sentence is "<<lm_score<<"\n";
   std::cout<<"The final score of last word of sentence is "<<final_score<<"\n";
}*/

