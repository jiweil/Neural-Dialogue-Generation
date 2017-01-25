local model={}

function model:Initial(params)
    self.params=params;
    self.map=nn.Sequential()
    self.map:add(nn.Linear(self.params.dimension,self.params.dimension))
    self.map:add(nn.Tanh())
    self.map:add(nn.Linear(self.params.dimension,self.params.dimension))
    self.map:add(nn.Tanh())
    self.map:add(nn.Linear(self.params.dimension,1))
    self.map=self.map:cuda()
    self.mse = nn.MSECriterion()
    self.mse=self.mse:cuda()
    
    if self.params.readSequenceModel then
        self.forward_model=torch.reload("../../Decode/decode_model")
        local open_=torch.DiskFile(params.forward_params_file,"r"):binary();
        self.forward_model.params=open_:readObject()
        self.forward_model.params.dictPath="../../data/movie_25000"
        --self.forward_model.params.batch_size=self.params.batch_size
        self.forward_model.params.model_file=params.forward_model_file
        self.forward_model:DecoderInitial(self.forward_model.params_file)
        self.forward_model.mode="test"
        
        self.backward_model=torch.reload("../../Decode/decode_model")
        local open_=torch.DiskFile(params.backward_params_file,"r"):binary();
        self.backward_model.params=open_:readObject()
        --self.backward_model.params.batch_size=self.params.batch_size
        self.backward_model.params.dictPath="../../data/movie_25000"
        self.backward_model.params.model_file=params.backward_model_file
        self.backward_model:DecoderInitial()
        self.backward_model.mode="test"
    end
    
    if self.params.readFutureModel then
        local parameter,_=self.map:parameters()
        local file=torch.DiskFile(self.params.FuturePredictorModelFile,"r"):binary();
        local read_params=file:readObject();
        for j=1,#parameter do
            parameter[j]:copy(read_params[j]);
        end
    end
    print("read data done")
end

function model:model_forward()
    local total_Error=0;
    local instance=0;
    for t=1,self.forward_model.Word_t:size(2)-1 do
        local representation_left=self.forward_model.store_t[t][2*self.backward_model.params.layers-1]:index(1,self.forward_model.Left_t[t])
        local predict_left=self.backward_score:index(1,self.forward_model.Left_t[t])
        self.map:zeroGradParameters()
        local pred=self.map:forward(representation_left)
        local Error=self.mse:forward(pred,predict_left)
        if self.mode=="train" then
            self.mse:backward(pred,predict_left)
            self.map:backward(representation_left,self.mse.gradInput)
            self.map:updateParameters(self.params.alpha)
        end
        total_Error=total_Error+Error*representation_left:size(1)
        instance=instance+representation_left:size(1)
    end
    return total_Error,instance
end


function model:test()
    local open_train_file=io.open(self.params.test_file,"r")
    local End,Word_s,Word_t,Mask_s,Mask_t;
    local End=0;
    local batch_n=1;
    local total_Error=0;
    local n_instance=0;
    local aver_ppl=0;
    local sen_num=0
    while End==0 do
        batch_n=batch_n+1;
        End,self.forward_model.Word_s,self.forward_model.Word_t,self.forward_model.Mask_s,self.forward_model.Mask_t,self.forward_model.Left_s,self.forward_model.Left_t,self.forward_model.Padding_s,self.forward_model.Padding_t,self.forward_model.Source,self.forward_model.Target=self.forward_model.Data:read_train(open_train_file)
        if End==1 then
            break;
        end
        self.forward_model:model_forward()
        self.backward_model.Source={}
        self.backward_model.Target={}
        for i=1,#self.forward_model.Source do
            self.backward_model.Source[i]=self.forward_model.Target[i]:sub(1,-1,2,self.forward_model.Target[i]:size(2)-1);
            self.backward_model.Target[i]=torch.cat(torch.Tensor({{self.backward_model.Data.EOS}}),torch.cat(self.forward_model.Source[i],torch.Tensor({self.backward_model.Data.EOT})))
        end
        self.backward_model.Word_s,self.backward_model.Mask_s,self.backward_model.Left_s,self.backward_model.Padding_s=self.backward_model.Data:get_batch(self.backward_model.Source,true)
        self.backward_model.Word_t,self.backward_model.Mask_t,self.backward_model.Left_t,self.backward_model.Padding_t=self.backward_model.Data:get_batch(self.backward_model.Target,false)
        
        self.backward_model:model_forward()
        self.backward_score=self.backward_model:SentencePpl()
        aver_ppl=aver_ppl+self.backward_score:sum();
        sen_num=sen_num+self.backward_score:size(1)
        self.mode="test"
        Batch_error,Batch_instance=self:model_forward()
        total_Error=total_Error+Batch_error
        n_instance=n_instance+Batch_instance
    end
    print(total_Error/n_instance)
end

function model:save(batch_n)
    local params,_=self.map:parameters()
    local file=torch.DiskFile(self.params.save_model_path.."/"..batch_n,"w"):binary();
    file:writeObject(params);
    file:close()
end

function model:train()
    local timer=torch.Timer();
    local End,Word_s,Word_t,Mask_s,Mask_t;
    local End=0;
    local batch_n=1;
    self.iter=0;
    self:test()
    while true do
        End=0
        self.iter=self.iter+1
        local open_train_file=io.open(self.params.train_file,"r")
        while End==0 do
            batch_n=batch_n+1;
            if batch_n%5000==0 then
                print("batch_n  "..batch_n)
                self:test()
                self:save(batch_n)
            end
            local time1=timer:time().real;
            End,self.forward_model.Word_s,self.forward_model.Word_t,self.forward_model.Mask_s,self.forward_model.Mask_t,self.forward_model.Left_s,self.forward_model.Left_t,self.forward_model.Padding_s,self.forward_model.Padding_t,self.forward_model.Source,self.forward_model.Target=self.forward_model.Data:read_train(open_train_file)
            if End==1 then
                break;
            end
            self.forward_model:model_forward()
            self.backward_model.Source={}
            self.backward_model.Target={}
            for i=1,#self.forward_model.Source do
                self.backward_model.Source[i]=self.forward_model.Target[i]:sub(1,-1,2,self.forward_model.Target[i]:size(2)-1);
                self.backward_model.Target[i]=torch.cat(torch.Tensor({{self.backward_model.Data.EOS}}),torch.cat(self.forward_model.Source[i],torch.Tensor({self.backward_model.Data.EOT})))
            end
            self.backward_model.Word_s,self.backward_model.Mask_s,self.backward_model.Left_s,self.backward_model.Padding_s=self.backward_model.Data:get_batch(self.backward_model.Source,true)
            self.backward_model.Word_t,self.backward_model.Mask_t,self.backward_model.Left_t,self.backward_model.Padding_t=self.backward_model.Data:get_batch(self.backward_model.Target,false)
            
            self.backward_model:model_forward()
            self.backward_score=self.backward_model:SentencePpl()
            self.mode="train"
            self:model_forward()
            local time2=timer:time().real;
        end
        self:save(batch_n)
        self:test()
        --break
    end
end

return model
