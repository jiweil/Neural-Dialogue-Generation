#/data/users/$USER/fbsource/fbcode/buck-out/gen/deeplearning/torch/cuth.lex run.lua -TopResponseFile ../extract_top/top_response_index.txt -TrainingData ../Encoder/t_given_s_dialogue_length2_6_train.txt1 -OutputFile Glove_Left_1 -save_score
#/data/users/$USER/fbsource/fbcode/buck-out/gen/deeplearning/torch/cuth.lex run.lua -TopResponseFile ../extract_top/top_response_index_iter1.txt -TrainingData ../Encoder/t_given_s_dialogue_length2_6_train.txt1 -OutputFile Glove_Left_four_1 -loadscore -total_lines 45288384 -distill_four_gram

#python NumToString1.py ../extract_top/movie_25000 Glove_remove_index.txt Glove_remove_text.txt
#python NumToString1.py ../extract_top/movie_25000 Glove_reserve_index.txt Glove_reserve_text.txt


#/data/users/$USER/fbsource/fbcode/buck-out/gen/deeplearning/torch/cuth.lex run.lua -TopResponseFile ../extract_top/top_response_index_iter2.txt -TrainingData /mnt/vol/gfsai-east/ai-group/users/jiwei/Generation/conversation/data/Glove_Left_1 -OutputFile Glove_Left_2 -save_score
#/data/users/$USER/fbsource/fbcode/buck-out/gen/deeplearning/torch/cuth.lex run.lua -TopResponseFile ../extract_top/top_response_index_iter2.txt -TrainingData /mnt/vol/gfsai-east/ai-group/users/jiwei/Generation/conversation/data/Glove_Left_1 -OutputFile Glove_Left_2 -loadscore -total_lines 40671162 -distill_four_gram 

#python NumToString1.py ../extract_top/movie_25000 Glove_remove_index.txt Glove_remove_text.txt
#python NumToString1.py ../extract_top/movie_25000 Glove_reserve_index.txt Glove_reserve_text.txt


#/data/users/$USER/fbsource/fbcode/buck-out/gen/deeplearning/torch/cuth.lex run.lua -TopResponseFile ../extract_top/top_response_four_index_iter2.txt -TrainingData /data/users/jiwel/fbsource/fbcode/experimental/deeplearning/jiwei/babi/rlbabi2/jiwei/distill/Glove/Glove_Left_2 -OutputFile Glove_Left_3 -save_score -save_score_file relevance_score_2
/data/users/$USER/fbsource/fbcode/buck-out/gen/deeplearning/torch/cuth.lex run.lua -TopResponseFile ../extract_top/top_response_four_index_iter2.txt -TrainingData /data/users/jiwel/fbsource/fbcode/experimental/deeplearning/jiwei/babi/rlbabi2/jiwei/distill/Glove/Glove_Left_2 -OutputFile Glove_Left_3 -loadscore -total_lines 35701138 -distill_four_gram -save_score_file relevance_score_2
