local model={}
local base_model=torch.reload("../../Atten/atten")
setmetatable(model,{ __index = base_model })

function model:Initial(params)
    local params_file=torch.DiskFile(params.params_file,"r"):binary();
    local model_params=params_file:readObject()
    params_file:close();
    for i,v in pairs(model_params)do 
        if params[i]==nil or params[i]=="" then
            params[i]=v;
        end
    end
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
        self.Data:Initial(params)
        self.lstm_source =self:lstm_source_()
        self.lstm_target =self:lstm_target_()
        self.softmax =self:softmax_()
        self.Modules={}
        self.Modules[#self.Modules+1]=self.lstm_source
        self.Modules[#self.Modules+1]=self.lstm_target;
        self.Modules[#self.Modules+1]=self.softmax;
        self.lstms_s={}
        self.lstms_t={}
        self.lstms_s[1]=self.lstm_source
        self.lstms_t[1]=self.lstm_target
        self.store_s={}
        self:readModel()
    end
    if self.params.readFutureModel then
        local parameter,_=self.map:parameters()
        local file=torch.DiskFile(self.params.FuturePredictorModelFile,"r"):binary();
        local read_params=file:readObject();
        for j=1,#parameter do
            parameter[j]:copy(read_params[j]);
        end
        print("length predictor intialization done")
    end
end

function model:model_forward()
    local total_Error=0;
    local instance=0;
    self.context=torch.Tensor(self.Word_s:size(1),self.Word_s:size(2),self.params.dimension):cuda();
    for t=1,self.Word_s:size(2) do
        local input={};
        if t==1 then
            for ll=1,self.params.layers do
                table.insert(input,torch.zeros(self.Word_s:size(1),self.params.dimension):cuda());
                table.insert(input,torch.zeros(self.Word_s:size(1),self.params.dimension):cuda());
            end
        else
            input=self:clone_(output);
        end
        table.insert(input,self.Word_s:select(2,t));
        self.lstms_s[1]:evaluate()
        output=self.lstms_s[1]:forward(input);
        if self.Mask_s[t]:nDimension()~=0 then
            for i=1,#output do
                output[i]:indexCopy(1,self.Mask_s[t],torch.zeros(self.Mask_s[t]:size(1),self.params.dimension):cuda());
            end
        end
        self.context[{{},t}]:copy(output[2*self.params.layers-1]);
    end
    for t=1,self.Word_t:size(2)-1 do
        local lstm_input={};
        if t==1 then
            lstm_input=output
        else
            lstm_input={}
            for i=1,2*self.params.layers do
                lstm_input[i]=output[i];
            end
        end
        table.insert(lstm_input,self.context);
        table.insert(lstm_input,self.Word_t:select(2,t));
        table.insert(lstm_input,self.Padding_s);
        output=self.lstms_t[1]:forward(lstm_input);
        self.Length=self.Length-1;
        local left_index=(torch.range(1,#self.Source)[self.Length:gt(0)]):long()
        local representation_left=output[2*self.params.layers-1]:index(1,left_index);
        local length_left=self.Length:index(1,left_index);
        length_left=torch.reshape(length_left,length_left:size(1),1):cuda()

        self.map:zeroGradParameters()
        local pred=self.map:forward(representation_left)
        local Error=self.mse:forward(pred,length_left)
        if self.mode=="train" then
            self.mse:backward(pred,length_left)
            self.map:backward(representation_left,self.mse.gradInput)
            self.map:updateParameters(self.params.alpha)
        end
        total_Error=total_Error+Error*length_left:size(1)
        instance=instance+length_left:size(1)
        if self.Mask_t[t]:nDimension()~=0 then
            for i=1,#output do
                output[i]:indexCopy(1,self.Mask_t[t],torch.zeros(self.Mask_t[t]:size(1),self.params.dimension):cuda());
            end
        end
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
    while End==0 do
        batch_n=batch_n+1;
        self:clear()
        End,self.Word_s,self.Word_t,self.Mask_s,self.Mask_t,self.Left_s,self.Left_t,self.Padding_s,self.Padding_t,self.Source,self.Target=self.Data:read_train(open_train_file)
        if End==1 then
            break;
        end
        self.Length=torch.Tensor(#self.Target):fill(0);
        for i=1,#self.Target do
            self.Length[i]=self.Target[i]:size(2);
        end
        self.mode="test"
        self.Word_s=self.Word_s:cuda();
        self.Word_t=self.Word_t:cuda();
        self.Padding_s=self.Padding_s:cuda();
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
    while true do
        End=0
        self.iter=self.iter+1
        local open_train_file=io.open(self.params.train_file,"r")
        while End==0 do
            batch_n=batch_n+1;
            if batch_n%10000==0 then
                print(batch_n)
                self:test()
                self:save(batch_n)
            end
            self:clear()
            local time1=timer:time().real;
            End,self.Word_s,self.Word_t,self.Mask_s,self.Mask_t,self.Left_s,self.Left_t,self.Padding_s,self.Padding_t,self.Source,self.Target=self.Data:read_train(open_train_file)
            if End==1 then
                break;
            end
            self.Length=torch.Tensor(#self.Target):fill(0);
            for i=1,#self.Target do
                self.Length[i]=self.Target[i]:size(2);
            end
            self.mode="train"
            self.Word_s=self.Word_s:cuda();
            self.Word_t=self.Word_t:cuda();
            self.Padding_s=self.Padding_s:cuda();
            self:model_forward()
            local time2=timer:time().real;
        end
        self:save(batch_n)
        self:test()
        break
    end
end

return model
