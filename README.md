#Neural Dialogue Generation

This project contains the code or part of the code for the dialogue generation part in the following papers:
* [1] J.Li, M.Galley, C.Brockett, J.Gao and B.Dolan. "[A Diversity-Promoting Objective Function for Neural Conversation Models](https://arxiv.org/pdf/1510.03055.pdf)". NAACL2016.
* [2] J.Li, M.Galley, C.Brockett, J.Gao and B.Dolan. "[A persona-based neural conversation model](https://arxiv.org/pdf/1603.06155.pdf)". ACL2016.
* [3] J.Li, W.Monroe, T.Shi, A.Ritter, D.Jurafsky. "[Adversarial Learning for Neural Dialogue Generation
](https://arxiv.org/pdf/1701.06547.pdf)" arxiv
* [4] J.Li, W.Monroe, D.Jurafsky. "[Learning to Decode for Future Success](https://arxiv.org/pdf/1701.06549.pdf)" arxiv
* [5] J.Li, W.Monroe, D.Jurafsky. "[A Simple, Fast Diverse Decoding Algorithm for Neural Generation](https://arxiv.org/pdf/1611.08562.pdf)" arxiv
* [6] J.Li, W.Monroe, D.Jurafsky. "Data Distillation for Controlling Specificity in Dialogue Generation (to appear on arxiv)"

This project is maintained by [Jiwei Li](http://www.stanford.edu/~jiweil/). Feel free to contact jiweil@stanford.edu for any relevant issue. This repo will continue to be updated. Thanks to all the collaborators: [Will Monroe](http://stanford.edu/~wmonroe4/), [Michel Galley](https://www.microsoft.com/en-us/research/people/mgalley/), [Alan Ritter](http://aritter.github.io/), [TianLin Shi](http://www.timshi.xyz/home/index.html), [Jianfeng Gao](http://research.microsoft.com/en-us/um/people/jfgao/), [Chris Brockett](https://www.microsoft.com/en-us/research/people/chrisbkt/), [Bill Dolan](https://www.microsoft.com/en-us/research/people/billdol/) and [Dan Jurafsky](https://web.stanford.edu/~jurafsky/).

#Setup

This code requires Torch7 and the following luarocks packages 
* [fbtorch](https://github.com/facebook/fbtorch)
* [cutorch](https://github.com/torch/cutorch)
* [cunn](https://github.com/torch/cunn)
* [nngraph](https://github.com/torch/nngraph)
* [torchx](https://github.com/nicholas-leonard/torchx)
* [tds](https://github.com/torch/tds)

#Download Data
Processed traning datasets can be downloaded at [link](http://nlp.stanford.edu/data/OpenSubData.tar) (unpacks to 8.9GB). All tokens have been transformed to indexes (dictionary file found at data/movie_2500)

    t_given_s_dialogue_length2_3.txt: dialogue length 2, minimum utterance length 3, sources and targets separated by "|"
    s_given_t_dialogue_length2_3.txt: dialogue length 2, minimum utterance length 3, targets and sources separated by "|"
    t_given_s_dialogue_length2_6.txt: dialogue length 2, minimum utterance length 6, sources and targets separated by "|"
    s_given_t_dialogue_length2_6.txt: dialogue length 2, minimum utterance length 6, targets and sources separated by "|"
    t_given_s_dialogue_length3_6.txt: dialogue length 3, minimum utterance length 6, contexts (consisting of 2 utterances) and targets separated by "|"



#Atten
Training a vanilla attention encoder-decoder model. 

Available options include:

    -batch_size     (default 128,batch size)
    -dimension      (default 512, vector dimensionality)
    -dropout        (default 0.2, dropout rate)
    -train_file     (default ../data/t_given_s_train.txt)
    -dev_file       (default ../data/t_given_s_dev.txt)
    -test_file      (default ../data/t_given_s_test.txt)
    -init_weight    (default 0.1)
    -alpha          (default 1, initial learning rate)
    -start_halve    (default 6, when the model starts halving its learning rate)
    -thres          (default 5, threshold for gradient clipping)
    -source_max_length  (default 50, max length of source sentences)
    -target_max_length  (default 50, max length of target sentences)
    -layers         (default 2, number of lstm layers)
    -saveFolder     (default "save",the folder to save models and parameters)
    -reverse        (default false, whether to reverse the sources)
    -gpu_index      (default 1, the index of the GPU you want to train your model on)
    -saveModel      (default true, whether to save the trained model)
    -dictPath       (default ../data/movie_25000, dictionary file)

training/dev/testing data: each line corresponds a source-target pair (in t_given_s*.txt) or target-source pair (in s_given_t*.txt) separated by "|". 

to train the forward p(t|s) model, run
        
    th train_atten.lua -train_file ../data/t_given_s_train.txt -dev_file ../data/t_given_s_dev.txt -test_file ../data/t_given_s_test.txt -saveFolder save_t_given_s

After training, the trained models will be saved in save_t_given_s/model*. input parameters will be stored in save_t_given_s/params

to train the backward model p(s|t), run

    th train_atten.lua -train_file ../data/s_given_t_train.txt -dev_file ../data/s_given_t_dev.txt -test_file ../data/s_given_t_test.txt -saveFolder save_s_given_t

the trained models will be stored in save_s_given_t/model*. input parameters will be stored insave_s_given_t/params

#Decode

Decoding given a pre-trained generative model. The pre-trained model doesn't have to be a vanila Seq2Seq model (for example, it can be a trained model from adversarial learning).   

Available options include:

    -beam_size      (default 7, beam size)
    -batch_size     (default 128, decoding batch size)
    -params_file    (default "../Atten/save_t_given_s/params", input parameter files for a pre-trained Seq2Seq model.)
    -model_file     (default "../Atten/save_s_given_t/model1",path for loading a pre-trained Seq2Seq model)
    -setting        (default "BS", the setting for decoding, taking values of "sampling": sampling tokens from the distribution; "BS" standard beam search; "DiverseBS", the diverse beam search described in [5]; "StochasticGreedy", the StochasticGreedy model described in [6])
    -DiverseRate    (default 0. The diverse-decoding rate for penalizing intra-sibling hypotheses in the diverse decoding model described in [5])
    -InputFile      (default "../data/t_given_s_test.txt", the input file for decoding)
    -OutputFile     (default "output.txt", the output file to store the generated responses)
    -max_length     (default 20, the maximum length of a decoded response)
    -min_length     (default 1, the minimum length of a decoded response)
    -NBest          (default false, whether to output a decoded N-best list (if true) or only  output the candidate with the greatest score(if false))
    -gpu_index      (default 1, the index of GPU to use for decoding)
    -allowUNK       (default false, whether to allow to generate UNK tokens)
    -MMI            (default false, whether to perform the mutual information reranking after decoding as in [1])
    -MMI_params_file    (default "../Atten/save_s_given_t/params", the input parameter file for training the backward model p(s|t))
    -MMI_model_file     (default "../Atten/save_s_given_t/model1", path for loading the backward model p(s|t))
    -max_decoded_num    (default 0. the maximum number of instances to decode. decode the entire input set if the value is set to 0.)
    -output_source_target_side_by_side  (default true, output input sources and decoded targets side by side)
    -dictPath       (default ../data/movie_25000, dictionary file)

to run the model
    
    th decode.lua [params]

to run the mutual information reranking model in [1],  -MMI_params_file and -MMI_model_file need to be pre-specified

#Persona_Addressee

the persona_addressee model described in [2]

Additional options include:

    -train_file     (default ../data/speaker_addresseet_train.txt)
    -dev_file       (default ../data/speaker_addressee_dev.txt)
    -test_file      (default ../data/speaker_addressee_test.txt)
    -SpeakerNum     (default 10000, number of distinct speakers)
    -AddresseeNum   (default 10000, number of distinct addressees)
    -speakerSetting (taking values of "speaker" or "speaker_addressee". For "speaker", only the user who speaks is modeled. For "speaker_addressee" both the speaker and the addressee are modeled)

data: the first token of a source line is the index of the Addressee and the first token in the target line is the index of the speaker. For example: 2 45 43 6|1 123 45 means that the index of the addressee is 2 and the index of the speaker is 1

to train the model

    th train.lua [params]

#Adversarial 

the adversarial-reinforcement learning model and the adversarial-evaluation model described in [3]

##discriminative 

adversarial-evaluation: to train a binary evaluator (a hierarchical neural net) to label dialogues as machine-generated (negative) or human-generated (positive)

Available options are:

    -batch_size     (default 128, batch size)
    -dimension      (default 512, vector dimensionality)
    -dropout        (default 0.2, dropout rate)
    -pos_train_file     (default "../../data/t_given_s_train.txt", human generated training examples)
    -neg_train_file     (default "../../data/decoded_train.txt", machine generated training examples)
    -pos_dev_file       (default "../../data/t_given_s_dev.txt", human generated dev examples)
    -neg_dev_file       (default "../../data/decoded_dev.txt", machine generated dev examples)
    -pos_test_file      (default "../../data/t_given_s_test.txt", human generated test examples)
    -neg_test_file      (default "../../data/decoded_test.txt", machine generated test examples)
    -source_max_length      (default 50, maximum sequence length)
    -dialogue_length        (default 2, the number of turns for a dialogue. the model supports multi-turn dialgoue classification)
    -save_model_path        (default "save", path for saving a trained discriminative model)
    -save_params_file       (default "save/params", path for saving input hyperparameters)
    -saveModel              (default true, whether to save the model)

##Reinforce

to train the adversarial-reinforcement learning model in [3]

Available options include:

    -disc_params        (default "../discriminative/save/params", hyperparameters for the pre-trained discriminative model)
    -disc_model         (default "../discriminative/save/iter1", path for loading a pre-trained discriminative model)
    -generate_params    (default "../../Atten/save_t_given_s/params", hyperparameters for the pre-trained generative model)
    -generate_model     (default ../../Atten/save_t_given_s/model1, path for loading a pre-trained generative model)
    -trainData      (default "../../data/t_given_s_train.txt", path for the training set)
    -devData        (default "../../data/t_given_s_train.txt", path for the dev set)
    -testData       (default "../../data/t_given_s_train.txt", path for the test set)
    -saveFolder     (default "save", path for data saving)
    -vanillaReinforce       (default false, whether to use vanilla Reinforce or Monte Carlo)
    -MonteCarloExample_N    (default 5, number of tries for Monte Carlo search to approximnate the expectation)
    -baseline       (default true, whether to use baseline or not)
    -baselineType   (default "critic", taking value of either "aver" or "critic". If set to "critic", another neural model is trained to estimate the reward, the role of which is similar to the critic in the actor-critic RL model; If set to "aver", just use the average reward for earlier examples as a baseline")
    -baseline_lr    (default 0.0005, learning rate for updating the critic)
    -logFreq        (default 2000, how often to print the log and save the model)
    -Timeslr        (default 0.5, increasing the learning rate)
    -gSteps         (default 1, how often to update the generative model)
    -dSteps         (default 5, how often to update the discriminative model)
    -TeacherForce   (default true, whether to run the teacher forcing model)

To run the adversarial-reinforcement learning model, a pretrained generative model and a pretrained discriminative model are needed. Trained models will be saved and can be later re-loaded for decoding using different decoding strategies in the folder "decode".

to train the model

    th train.lua [params]

Note: if you encounter the following error "bad argument #2 to '?' (out of range) in function model_backward" after training the model for tens of hours, this means the model has exploded (see the teacher forcing part in Section 3.2 of the paper). The reason why the error appears as "bad argument #2 to '?'" is because of the sampling algorithm in Torch. If you encounter this issue, shrink the value of the variable -Timeslr.

#Future_Prediction 

the future prediction (Soothsayer) models described in [4]

##train_length 

to train the Soothsayer Model for Length Prediction

Available options include:

    -dimension          (default 512, vector dimensionality. The value should be the same as that of the pretrained Seq2Seq model. Otherwise, an error will be reported)
    -params_file        (default "../../Atten/save_t_given_s/params", load hyperparameters for a pre-trained generative model)
    -generate_model     (default ../../Atten/save_t_given_s/model1, path for loading the pre-trained generative model)
    -save_model_path    (default "save", path for saving the model)
    -train_file         (default "../../data/t_given_s_train.txt", path for the training set)
    -dev_file           (default "../../data/t_given_s_dev.txt", path for the training set)
    -test_file          (default "../../data/t_given_s_test.txt", path for the training set)
    -alpha              (default 0.0001, learning rate)
    -readSequenceModel  (default true, whether to read a pretrained seq2seq model. this variable has to be set to true when training the model)
    -readFutureModel    (default false, whether to load a pretrained Soothsayer Model. this variable has to be set to false when training the model)
    -FuturePredictorModelFile   (path for load a pretrained Soothsayer Model. does not need it at model training time)

to train the model (a pretrained Seq2Seq model is required)

    th train_length.lua [params]

##train_backward

train the Soothsayer Model to predict the backward probability p(s|t) of the mutual information model

Available options include:

    -dimension              (default 512, vector dimensionality. This value should be the same as that of the pretrained Seq2Seq model. Otherwise, an error will be reported)
    -batch_size             (default 128, batch_size)
    -save_model_path        (default "save")
    -train_file             (default "../../data/t_given_s_train.txt", path for the training set)
    -dev_file               (default "../../data/t_given_s_dev.txt", path for the training set)
    -test_file              (default "../../data/t_given_s_test.txt", path for the training set)
    -alpha                  (default 0.01, learning rate)
    -forward_params_file(default "../../Atten/save_t_given_s/params",input parameter files for a pre-trained Seq2Seqmodel p(t|s))
    -forward_model_file     (default "../../Atten/save_s_given_t/model1", path for loading the pre-trained Seq2Seq model p(t|s))
    -backward_params_file   (default "../../Atten/save_s_given_t/params",input parameter files for a pre-trained backward Seq2Seq model p(s|t))
    -backward_model_file    (default "../../Atten/save_s_given_t/model1" path for loading the pre-trained backward Seq2Seq model p(s|t))
    -readSequenceModel      (default true, whether to read a pretrained seq2seq model. this variable has to be set to true when during the model training period)
    -readFutureModel        (default false, whether to load a pretrained Soothsayer Model. this variable has to be set to false during the model training period)
    -PredictorFile          (path for load a pretrained Soothsayer Model. does not need it at model training time)


to train the model (a pretrained forward Seq2Seq model p(t|s) and a backward model p(s|t) are both required)

    th train.lua [params]


##decode
    
decoding by combining a pre-trained Seq2Seq model and a Soothsayer future prediction model
    
Other than the input parameters of the standard decoding model in the Folder "decode", additional options include:

    -Task                   (the future prediction task, taking values of "length" or "backward")
    -target_length          (default 0, forcing the model to generate sequences of a pre-specific length. 0 if there is no such a constraint. If your task is "length", a value for -target_length is required)
    -FuturePredictorModelFile   (path for loading a pre-trained Soothsayer future prediction model. If "Task" takes a value of "length", the value of FuturePredictorModelFile should be a model saved from training length prediction model in folder train_length. If "Task" takes a value of "backward", the model is a model saved from training the backward probability model in the folder train_backward)
    -PredictorWeight        (default 0, the weight for the Soothsayer model)
    
To run the decoder with a pre-trained Soothsayer model of length:
    
    th decode.lua -params_file hyperparameterFile_pretrained_seq2seq -model_file modelFile_pretrained_seq2seq -InputFile yourInputFileToDecode -OutputFile yourOutputFile -FuturePredictorModelFile modelFile_Soothsayer_length -PredictorWeight 1 -Task length -target_length 15
    
To run the decoder with a pre-trained Soothsayer model of backward probability

    th decode.lua -params_file hyperparameterFile_pretrained_seq2seq -model_file modelFile_pretrained_seq2seq -InputFile yourInputFileToDecode -OutputFile yourOutputFile -FuturePredictorModelFile modelFile_Soothsayer_backward -PredictorWeight 1 -Task backward
    
If you want to perform MMI reranking at the end,  -MMI_params_file and -MMI_model_file have to be pre-specified

#Distill

This folder contains the code for the data distillation method described in [6].

to run the model:

    sh pipeline.sh

* First, decode a large input set (more than 1 million) using a pre-trained Seq2Seq model

    cd ../decode

    th decode.lua -params_file hyperparameterFile_pretrained_seq2seq -model_file modelFile_pretrained_seq2seq -batch_size 640 -InputFile yourTrainingData -OutputFile yourDecodingOutputFile -batch_size -max_decoded_num 1000000

* Second, extract top frequent responses

    cd ../distill/extract_top 

    sh select_top_decoded.sh yourDecodingOutputFile yourFileToStoreTopResponses

* Third, compute relevance scores for the entire training set and then distill the training set. The code provides two different ways to compute the scores: using a pre-trained Seq2Seq model or averaging Glove embeddings

    cd ../Glove or cd ../Encoder

##Glove

options include
    
    -TrainingData       (path for your training data to distill)
    -TopResponseFile    (path for your extracted top frequent responses)
    -batch_size         (default 1280, batch size)
    -save_score_file    (default "relevance_score", path for saving relevance_score for each instance in the training set)
    -distill_rate       (default 0.08, the proportion of training data to distill in this round)
    -distill_four_gram  (default true, whether to remove all training instances that share four-grams with any one of the top frequent responses)
    -loadscore          (default false, whether to load already-computed relevance scores)
    -save_score         (default false, wehther to save relevance scores)

Compute relevance scores: 
    
    th run.lua -TopResponseFile yourFileToStoreTopResponses -TrainingData yourTrainingData -OutputFile FileForRemainingData -save_score -save_score_file relevance_score
    
Distill the Data: 
    
    th run.lua -TopResponseFile yourFileToStoreTopResponses -TrainingData yourTrainingData -OutputFile FileForRemainingData -total_lines "number of lines in yourTrainingData" -save_score_file relevance_score 

The remaining data after this round of data distillation will be stored in FileForRemainingData, on which a new Seq2Seq model will be trained.
        
##Encoder
use a pre-trained Seq2Seq model for data distillation. Other than input parameters in Glove, the path for a pre-trained Seq2Seq model needs to be pre-specified

        -params_file        (default "../../Atten/save_t_given_s/params", hyperparameters for the pre-trained generative model)
        -model_file     (default ../../Atten/save_t_given_s/model1, path for loading a pre-trained generative model)
    
to run the model:

    th distill_encode.lua -TopResponseFile yourFileToStoreTopResponses -TrainingData yourTrainingData -OutputFile FileForRemainingData -params_file Seq2SeqParamsFile -model_file Seq2SeqModelFile -batch_size 6400


## Acknowledgments
[Yoon Kim](http://people.fas.harvard.edu/~yoonkim)'s [MT repo](https://github.com/harvardnlp/seq2seq-attn)

LantaoYu's [SeqGAN Repo](https://github.com/LantaoYu/SeqGAN)

### Licence
MIT
