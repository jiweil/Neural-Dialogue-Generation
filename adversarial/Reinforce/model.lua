require "cutorch"
require "nn"
require 'cunn'
require "nngraph"
local stringx=require("pl.stringx")

local model={};
function model:Initial(params)
    self.params=params
    self.generate_model=torch.reload("./generative")
    local file=torch.DiskFile(self.params.generate_params,"r"):binary()
    generate_params=file:readObject()
    generate_params.sample=params.sample
    generate_params.alpha=generate_params.alpha*self.params.lr
    generate_params.model_file=self.params.generate_model
    generate_params.batch_size=self.params.batch_size
    generate_params.vanillaReinforce=self.params.vanillaReinforce
    generate_params.MonteCarloExample_N=self.params.MonteCarloExample_N
    generate_params.max_length=40
    generate_params.min_length=0
    generate_params.dictPath="../../data/movie_25000"
    generate_params.output_source_target_side_by_side=true
    generate_params.PrintOutIllustrationSample=true
    generate_params.beam_size=1
    generate_params.target_length=0
    generate_params.NBest=false
    generate_params.train_file=self.params.trainData
    generate_params.dev_file=self.params.devData
    generate_params.test_file=self.params.testData
    print("generative_model")
    print(generate_params)
    file:close()
    self.generate_model:Initial(generate_params)
    self.generate_model:readModel()
    self.disc_model=torch.reload("../discriminative/dis_model")
    local file=torch.DiskFile(self.params.disc_params,"r"):binary()
    local disc_params=file:readObject()
    disc_params.model_file=self.params.disc_model
    disc_params.source_max_length=100
    disc_params.batch_size=self.params.batch_size
    disc_params.trainData=self.params.trainData
    disc_params.devData=self.params.devData
    disc_params.testData=self.params.testData
    self.disc_model:Initial(disc_params)
    self.disc_model:readModel()
    self.disc_model.mode="train"
    print("disc_model")
    print(disc_params)
    self.base_line_sum=0;
    self.base_line_n=0;
    self.log_=io.open(self.params.output_file,"w")
    if self.params.baseline and self.params.baselineType=="critic" then
        self.baseline = nn.Linear(self.params.dimension, 1)
        self.baseline.weight:fill(0)
        self.baseline.bias:fill(0)
        self.baseline:cuda()
        self.baseline_mse = nn.MSECriterion()
        self.baseline_mse:cuda()
    end
end

function model:save(batch_n)
    local params={};
    for i=1,#self.generate_model.Modules do
        params[i],_=self.generate_model.Modules[i]:parameters()
    end
    local file
    if batch_n~=nil then
        file=torch.DiskFile(self.params.save_prefix.."generate_iter"..self.iter.."_batch"..batch_n,"w"):binary();
    else
        file=torch.DiskFile(self.params.save_prefix.."generate_iter"..self.iter,"w"):binary();
    end
    file:writeObject(params);
    file:close()

    local params={};
    for i=1,#self.disc_model.Modules do
        params[i],_=self.disc_model.Modules[i]:parameters()
    end
    local file
    if batch_n~=nil then
        file=torch.DiskFile(self.params.save_prefix.."discriminate_iter"..self.iter.."_batch"..batch_n,"w"):binary();
    else
        file=torch.DiskFile(self.params.save_prefix.."discriminate_iter"..self.iter,"w"):binary();
    end
    file:writeObject(params);
    file:close()
end

function model:decodeSample()
    self.log_:write("Sampling ...\n")
    self.generate_model.params.setting="sampling"
    self.generate_model:DecodeIllustrationSample(self.log_)
    self.log_:write("Beam Search ...\n")
    self.generate_model.params.setting="BS"
    self.generate_model:DecodeIllustrationSample(self.log_)
    self.generate_model.params.setting="sampling"
end

function model:TrainD(open_train_file)
    local End=0;
    self.generate_model:clear()
    End,self.generate_model.Word_s,self.generate_model.Word_t,self.generate_model.Mask_s,self.generate_model.Mask_t,self.generate_model.Left_s,self.generate_model.Left_t,self.generate_model.Padding_s,self.generate_model.Padding_t,self.generate_model.Source,self.generate_model.Target=self.generate_model.Data:read_train(open_train_file)

    if End==1 then
        return End
    end
    self.generate_model.mode="decoding"
    self.generate_model:GenerateSample()
    
    local Source={};
    Source[1]={};Source[2]={}
    self.disc_model.labels=torch.Tensor(#self.generate_model.Source*2):cuda()
    self.disc_model.labels:sub(1,#self.generate_model.Source):fill(1)
    self.disc_model.labels:sub(1+#self.generate_model.Source,2*#self.generate_model.Source):fill(2)
    self.disc_model.labels=self.disc_model.labels:cuda()
        
    for i=1,#self.generate_model.Source do
        Source[1][#Source[1]+1]=self.generate_model.Source[i];
        Source[2][#Source[2]+1]=self.generate_model.Target[i]:sub(1,1,2,self.generate_model.Target[i]:size(2)-1)
    end
    for i=1,#self.generate_model.Source do
        Source[1][#Source[1]+1]=self.generate_model.Source[i];
        Source[2][#Source[2]+1]=self.generate_model.sample_target[i]:sub(1,1,2,self.generate_model.sample_target[i]:size(2)-1)
    end
    self.disc_model.Word_s,self.disc_model.Mask_s,self.disc_model.Left_t,self.disc_model.Padding_s=self.disc_model.Data:get_batch(Source)
    for i,v in pairs(self.disc_model.Word_s)do  
        v=v:cuda()
    end
    self.disc_model:clear()
    self.disc_model:model_forward()
    local dis_pred=self.disc_model:model_backward()
    if self.disc_model.mode=="train" then
        self.disc_model:update()
    end
    return End,dis_pred
end

function model:train()
    self.iter=0
    while true do 
        self.iter=self.iter+1
        local open_train_file=io.open(self.params.trainData,"r")
        local batch_n=0;
        local End=0
        while End==0 do
            local timer=torch.Timer();
            local time1=timer:time().real;
            batch_n=batch_n+1;
            if batch_n%(5*self.params.logFreq)==0 then
                print(self.params)
            end
            if batch_n%self.params.logFreq==0 then
                self.log_:write("batch_n  "..batch_n.."\n")
                --self:decodeSample()
                self.generate_model.mode="test"
                --self.generate_model:test()
                self:save(batch_n)
            end
            for i=1,self.params.dSteps do
                self.disc_model.mode="train"
                End=self:TrainD(open_train_file)
                if End==1 then break end
            end
            if End==0 then
                for i=1,self.params.gSteps do
                    self.disc_model.mode="test"
                    local End,dis_pred=self:TrainD(open_train_file)
                    if End==1 then break end
                    dis_pred=torch.exp(dis_pred)
                    local reward=dis_pred:sub(1+self.generate_model.params.batch_size,2*self.generate_model.params.batch_size,1,1);
                    self.generate_model:Integrate(true)
                    if not self.params.vanillaReinforce then
                        --run monte carlo
                        self:MonteCarloReward()
                    end
                    self:Reinforce(reward)
                end
                if self.params.TeacherForce then
                    self.generate_model:Integrate(false)
                    self.generate_model:clear()
                    self.generate_model.mode="train"
                    self.generate_model:model_forward()
                    self.generate_model:model_backward(true)
                    self.generate_model:update()
                end
            end
            local time2=timer:time().real;
        end
    end
end


function model:Reinforce(reward)
    reward=torch.reshape(reward,reward:size(1))
    self.base_line_sum=self.base_line_sum+reward:sum();
    self.base_line_n=self.base_line_n+reward:size(1)
    local baseline
    if self.params.baseline and self.params.baselineType=="critic" then
        local baseline_input=self.generate_model.SourceVector
        baseline = self.baseline:forward(baseline_input)
        baseline_error = self.baseline_mse:forward(baseline,reward)
        self.baseline:zeroGradParameters()
        self.baseline_mse:backward(baseline,reward);
        self.baseline:backward(baseline_input, self.baseline_mse.gradInput)
        self.baseline:updateParameters(self.params.baseline_lr)
    elseif self.params.baseline and self.params.baselineType=="aver" then
        baseline=self.base_line_sum/self.base_line_n
    end
    if self.params.baseline then
        self.generate_model.last_reward=self.params.Timeslr*(-reward+baseline)/self.params.batch_size
        if not self.params.vanillaReinforce then
            baseline=torch.repeatTensor(baseline,1,self.generate_model.MonteCarloReward:size(2))
            self.generate_model.MonteCarloReward=self.params.Timeslr*(-self.generate_model.MonteCarloReward+baseline)/self.params.batch_size
        end
    else
        self.generate_model.last_reward=-self.params.Timeslr*reward/self.params.batch_size
        if not self.params.vanillaReinforce then
            self.generate_model.MonteCarloReward=-self.params.Timeslr*self.generate_model.MonteCarloReward/self.params.batch_size
        end
    end
    self.generate_model.mode="train"
    self.generate_model:model_forward()
    self.generate_model:model_backward(false)
    self.generate_model:update(false)
end

function model:MonteCarloReward()
    local Source={};
    Source[1]={};Source[2]={}
    for j=1,#self.generate_model.Source do
        for i=1,self.params.MonteCarloExample_N do
            Source[1][#Source[1]+1]=self.generate_model.Source[j];
        end
    end
    for t=1,self.generate_model.Word_t:size(2)-2 do
        self.generate_model:ComputeMonteCarloReward(t) 
        Source[2]={}
        for j=1,#self.generate_model.partial_history do
            Source[2][#Source[2]+1]=self.generate_model.partial_history[j]
        end
        self.disc_model.Word_s,self.disc_model.Mask_s,self.disc_model.Left_t,self.disc_model.Padding_s=self.disc_model.Data:get_batch(Source)
        for i,v in pairs(self.disc_model.Word_s)do  
            v=v:cuda()
        end
        self.disc_model.mode="test"
        self.disc_model:model_forward()
        self.disc_model.labels=torch.Tensor(self.disc_model.Word_s[1]:size(1)):fill(1):cuda()
        local dis_pred=self.disc_model:model_backward()
        dis_pred=torch.exp(dis_pred)
        local reward_monte_carlo=dis_pred:select(2,1)
        reward_monte_carlo=torch.reshape(reward_monte_carlo,#self.generate_model.Source,self.params.MonteCarloExample_N)
        reward_monte_carlo=reward_monte_carlo:mean(2)
        if t==1 then
            self.generate_model.MonteCarloReward=reward_monte_carlo;
        else
            self.generate_model.MonteCarloReward=torch.cat(self.generate_model.MonteCarloReward,reward_monte_carlo,2);
        end
    end
end


return model
