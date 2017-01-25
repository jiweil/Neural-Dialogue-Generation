cd ../decode
th decode.lua -params_file hyperparameterFile_pretrained_seq2seq -model_file modelFile_pretrained_seq2seq -batch_size 640 -InputFile yourInputFileToDecode -OutputFile yourOutputFile -batch_size -max_decoded_num 1000000
echo "decode done"

cd ../distill/extract_top
sh select_top_decoded.sh yourDecodingOutputFile yourFileToStoreTopResponses
echo "extract top done"

cd ../Glove 
th run.lua -TopResponseFile yourFileToStoreTopResponses -TrainingData yourTrainingData -OutputFile FileForRemainingData -save_score -save_score_file relevance_score
echo "relevance score done
"
th run.lua -TopResponseFile yourFileToStoreTopResponses -TrainingData yourTrainingData -OutputFile FileForRemainingData -total_lines "number of lines in yourTrainingData" -save_score_file relevance_score

echo "data distillation done"

