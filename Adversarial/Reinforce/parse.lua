local stringx = require('pl.stringx')
local cmd = torch.CmdLine()
cmd:option("-disc_params","../discriminative/save/params","")
cmd:option("-disc_model","../discriminative/save/iter1","")
cmd:option("-generate_params","../../Atten/save_t_given_s/params","")
cmd:option("-generate_model","../../Atten/save_t_given_s/model1","")
cmd:option("-trainData","../../data/t_given_s_train.txt","")
cmd:option("-devData","../../data/t_given_s_dev.txt","")
cmd:option("-testData","../../data/t_given_s_test.txt","")
cmd:option("-saveFolder","save","")
cmd:option("-gpu_index",1,"")
cmd:option("-lr",1,"")
cmd:option("-dimension",512,"")
cmd:option("-batch_size",64,"")
cmd:option("-sample",true,"")
cmd:option("-vanillaReinforce",false,"true is using vanilla reinforce model, false doing intermediate step monte carlo for reward estimation")
cmd:option("-MonteCarloExample_N",5,"number of instances for Monte carlo search")
cmd:option("-baseline",true,"")
cmd:option("-baselineType","critic","how to compute the baseline, taking values of critic or aver. for critic, training another neural model to estimate the reward, the role of which is similar to the critic in the actor-critic RL model; for aver, just use the average reward for earlier examples as a baseline")
cmd:option("-baseline_lr",0.0005,"learning rate for updating the critic")
cmd:option("-logFreq",2000,"")
cmd:option("-Timeslr",0.5,"")
cmd:option("-gSteps",1,"")
cmd:option("-dSteps",5,"")
cmd:option("-TeacherForce",true,"whether to run the teacher forcing model")

local params= cmd:parse(arg)
params.saveFolder=params.saveFolder.."/"..(params.vanillaReinforce and "MC_n_" or "MC_y_")..(params.TeacherForce and "Teacher_y_" or "Teacher_n_")..(params.baseline and "Base_y" or "Base_n").."_lr"..params.Timeslr
paths.mkdir(params.saveFolder)

params.save_prefix=params.saveFolder.."/"
params.save_prefix_dis=params.saveFolder.."/dis_model"
params.save_prefix_generate=params.saveFolder.."/generate_model"
params.save_params_file=params.saveFolder.."/params"
params.output_file=params.saveFolder.."/log"

print(params)
return params;
