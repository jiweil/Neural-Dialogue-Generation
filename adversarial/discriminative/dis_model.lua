require "cutorch"
require "nn"
require 'cunn'
require "nngraph"
local model={};
model.Data=torch.reload("./data")

function model:Initial(params)
    self.Data:Initial(params)
    self.params=params
    self.lstm_word=self:lstm_(true)
    self.lstm_sen=self:lstm_(false)
    self.lstms_word={}
    self.store_word={};
    for i=1,self.params.dialogue_length do
        self.lstms_word[i]=self:g_cloneManyTimes(self.lstm_word,self.params.source_max_length)
        self.store_word[i]={}
    end
    self.lstms_sen=self:g_cloneManyTimes(self.lstm_sen,5);
    self.store_sen={}
    self.softmax =self:softmax_()
    self.Modules={};
    self.Modules[#self.Modules+1]=self.lstm_word
    self.Modules[#self.Modules+1]=self.lstm_sen
    self.Modules[#self.Modules+1]=self.softmax;
    assert(self.params.output_file~=nil)
    if self.params.output_file~="" then
        self.output=io.open(self.params.output_file,"w")
    end
end

local function copy(A)
    local B;
    if A:nDimension()==1 then
        B=torch.Tensor(A:size(1)):cuda();
    end
    if A:nDimension()==2 then
        B=torch.Tensor(A:size(1),A:size(2)):cuda();
    end
    if A:nDimension()==3 then
        B=torch.Tensor(A:size(1),A:size(2),A:size(3)):cuda();
    end
    B:copy(A);
    return B;
end

function model:clone_(A)
    local B={};
    for i=1,#A do
        if A[i]:nDimension()==2 then
            B[i]=torch.Tensor(A[i]:size(1),A[i]:size(2)):cuda();
        else
            B[i]=torch.Tensor(A[i]:size(1)):cuda();
        end
        B[i]:copy(A[i]);
    end
    return B;
end


function model:readModel()
    local file = torch.DiskFile(self.params.model_file,"r"):binary()
    local model_params=file:readObject();
    file:close();
    for i=1,#self.Modules do
        local parameter,_=self.Modules[i]:parameters()
        for j=1,#parameter do
            parameter[j]:copy(model_params[i][j]);
        end
    end
    print("read model done")
end

function model:save()
    local params={};
    for i=1,#self.Modules do
        params[i],_=self.Modules[i]:parameters()
    end
    local file=torch.DiskFile(self.params.save_model_path..'/iter'..self.iter,"w"):binary();
    file:writeObject(params);
    file:close()
end

function model:saveParams()
    local file=torch.DiskFile(self.params.save_params_file,"w"):binary();
    file:writeObject(self.params)
    file:close()
end

function model:model_forward()
    self.last={}
    for i=1,#self.Word_s do
        for t=1,self.Word_s[i]:size(2)do
            local input={};
            if t==1 then
                for ll=1,self.params.layers do
                    table.insert(input,torch.zeros(self.Word_s[i]:size(1),self.params.dimension):cuda());
                    table.insert(input,torch.zeros(self.Word_s[i]:size(1),self.params.dimension):cuda());
                end
            else
                if self.mode=="train" then
                    input=self:clone_(self.store_word[i][t-1]);
                else input=self:clone_(output);
                end
            end
            table.insert(input,self.Word_s[i]:select(2,t));
            if self.mode=="train" then
                self.lstms_word[i][t]:training()
                output=self.lstms_word[i][t]:forward(input);
            else
                self.lstms_word[i][1]:evaluate()
                output=self.lstms_word[i][1]:forward(input);
            end
            if self.Mask_s[i][t]:nDimension()~=0 then
                for k=1,#output do
                    output[k]:indexCopy(1,self.Mask_s[i][t],torch.zeros(self.Mask_s[i][t]:size(1),self.params.dimension):cuda());
                end
            end
            if self.mode=="train" then
                self.store_word[i][t]=self:clone_(output);
            end
            self.last[i]=output[2*self.params.layers-1]
        end
    end
    for t=1,#self.Word_s do
        local input={};
        if t==1 then
            for ll=1,self.params.layers do
                table.insert(input,torch.zeros(self.Word_s[1]:size(1),self.params.dimension):cuda())
                table.insert(input,torch.zeros(self.Word_s[1]:size(1),self.params.dimension):cuda())
            end
        else
            if self.mode=="train" then
                input=self:clone_(self.store_sen[t-1]);
            else input=self:clone_(output);
            end
        end
        table.insert(input,self.last[t]);
        if self.mode=="train" then
            self.lstms_sen[t]:training()
            output=self.lstms_sen[t]:forward(input);
        else
            self.lstms_sen[1]:evaluate()
            output=self.lstms_sen[1]:forward(input);
        end
        if t==#self.Word_s then
            self.softmax_h=output[2*self.params.layers-1]   
        end
        self.store_sen[t]=self:clone_(output);
    end
end

function model:model_backward()
    local softmax_output=self.softmax:forward({self.softmax_h,self.labels});
    --print(1/math.exp(-softmax_output[1]))
    if self.mode=="train" then
        local dh=self.softmax:backward({self.softmax_h,self.labels},{torch.Tensor({1}):cuda(),torch.Tensor(softmax_output[2]:size()):fill(0):cuda()})
        local d_words={};
        local d_output={};
        for ll=1,self.params.layers do
            table.insert(d_output,torch.zeros(self.Word_s[1]:size(1),self.params.dimension):cuda());
            table.insert(d_output,torch.zeros(self.Word_s[1]:size(1),self.params.dimension):cuda());
        end
        for t=#self.Word_s,1,-1 do
            local input={};
            if t==1 then
                for ll=1,self.params.layers do
                    table.insert(input,torch.zeros(self.Word_s[1]:size(1),self.params.dimension):cuda())
                    table.insert(input,torch.zeros(self.Word_s[1]:size(1),self.params.dimension):cuda())
                end
            else
                input=self:clone_(self.store_sen[t-1]);
            end
            table.insert(input,self.last[t]);
            local d_store_t=self:clone_(d_output);
            if t==#self.Word_s then
                d_store_t[2*self.params.layers-1]:add(dh[1]);
            end
            d_input=self.lstms_sen[t]:backward(input,d_store_t)
            for i=1,2*self.params.layers do
                d_output[i]=copy(d_input[i]);
            end
            d_words[t]=copy(d_input[2*self.params.layers+1]);
        end
        for i=1,#self.Word_s do
            local d_output={};
            for ll=1,self.params.layers do
                table.insert(d_output,torch.zeros(self.Word_s[1]:size(1),self.params.dimension):cuda());
                table.insert(d_output,torch.zeros(self.Word_s[1]:size(1),self.params.dimension):cuda());
            end
            for t=self.Word_s[i]:size(2),1,-1 do
                local input={};
                if t==1 then
                    for ll=1,self.params.layers do
                        table.insert(input,torch.zeros(self.Word_s[1]:size(1),self.params.dimension):cuda())
                        table.insert(input,torch.zeros(self.Word_s[1]:size(1),self.params.dimension):cuda())
                    end
                else
                    input=self:clone_(self.store_word[i][t-1]);
                end
                table.insert(input,self.Word_s[i]:select(2,t));
                local d_store_t=self:clone_(d_output);
                if t==self.Word_s[i]:size(2) then
                    d_store_t[2*self.params.layers-1]:add(d_words[i]);
                end
                d_input=self.lstms_word[i][t]:backward(input,d_store_t)
                for i=1,2*self.params.layers do
                    d_output[i]=copy(d_input[i]);
                end
            end
        end
    end
    return softmax_output[2],1/math.exp(-softmax_output[1][1])
end

function model:softmax_()
    local y=nn.Identity()();
    local h=nn.Identity()();
    local h2y= nn.Linear(self.params.dimension,2)(h)
    local pred= nn.LogSoftMax()(h2y)
    local Criterion=nn.ClassNLLCriterion()
    local err= Criterion({pred, y})
    local module= nn.gModule({h,y},{err,pred});
    module:getParameters():uniform(-self.params.init_weight, self.params.init_weight)
    return module:cuda()
end

function model:g_cloneManyTimes(net, T)
    local clones = {}
    for t=1,T do
        clones[t]=net:clone('weight','bias','gradWeight', 'gradBias');
    end
    return clones
end

function model:lstm_(isWordLevel)
    local inputs = {}
    local outputs = {}
    for ll=1,self.params.layers do
        local h_ll=nn.Identity()()
        table.insert(inputs,h_ll)
        local c_ll=nn.Identity()()
        table.insert(inputs,c_ll)
    end
    for ll=1,self.params.layers do
        local prev_h=inputs[ll*2-1];
        local prev_c=inputs[ll*2];
        local x
        if ll==1 then 
            assert(isWordLevel~=nil)
            if isWordLevel then
                local x_=nn.Identity()()
                table.insert(inputs,x_)
                x=nn.LookupTable(self.params.vocab_size,self.params.dimension)(x_);
            else  x=nn.Identity()()
                table.insert(inputs,x)
            end
        else x=outputs[(ll-1)*2-1];
        end
        local drop_x=nn.Dropout(self.params.dropout)(x)
        local drop_h=nn.Dropout(self.params.dropout)(inputs[ll*2-1])
        local i2h=nn.Linear(self.params.dimension,4*self.params.dimension)(drop_x);
        local h2h=nn.Linear(self.params.dimension,4*self.params.dimension)(drop_h);
        local gates=nn.CAddTable()({i2h,h2h});
        local reshaped_gates =  nn.Reshape(4,self.params.dimension)(gates);
        local sliced_gates = nn.SplitTable(2)(reshaped_gates);
        local in_gate= nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
        local in_transform= nn.Tanh()(nn.SelectTable(2)(sliced_gates))
        local forget_gate= nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
        local out_gate= nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))
        local l1=nn.CMulTable()({forget_gate,inputs[ll*2]})
        local l2=nn.CMulTable()({in_gate, in_transform})
        local next_c=nn.CAddTable()({l1,l2});
        local next_h= nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
        table.insert(outputs,next_h);
        table.insert(outputs,next_c);
    end
    local module= nn.gModule(inputs,outputs);
    module:getParameters():uniform(-self.params.init_weight, self.params.init_weight)
    return module:cuda()
end

function model:test()
    local open_pos_train_file,open_neg_train_file
    if self.mode=="dev" then
        open_pos_train_file=io.open(self.params.pos_dev_file,"r")
        open_neg_train_file=io.open(self.params.neg_dev_file,"r")
    elseif self.mode=="test" then
        open_pos_train_file=io.open(self.params.pos_test_file,"r")
        open_neg_train_file=io.open(self.params.neg_test_file,"r")
    end
    local End=0;
    local batch_n=1;
    local right_instance=0;
    local total_instance=0;
    local ppl=0
    while End==0 do
        batch_n=batch_n+1;
        End,self.labels,self.Word_s,self.Mask_s,self.Left_s,self.Padding_s=self.Data:read_train(open_pos_train_file,open_neg_train_file)
        if End==1 then
            break;
        end
        for i=1,#self.Word_s do
            self.Word_s[i]=self.Word_s[i]:cuda()
        end
        self.labels=self.labels:cuda()
        self:model_forward()
        local prob,current_ppl=self:model_backward()
        ppl=ppl+current_ppl
        local _,indexes=torch.max(prob,2);
        for i=1,indexes:size(1)do
            total_instance=total_instance+1
            if indexes[i][1]==self.labels[i] then
                right_instance=right_instance+1;
            end
        end
    end
    open_pos_train_file:close()
    open_neg_train_file:close()
    print(right_instance/total_instance,ppl/batch_n)
    if self.params.output_file~="" then
        self.output:write("iter  "..(right_instance/total_instance).."\n")
    end
end

function model:update()
    local lr
    if self.lr~=nil then
        lr=self.lr
    else    
        lr=self.params.alpha
    end
    local grad_norm=0;
    for i=1,#self.Modules do
        local p,dp=self.Modules[i]:parameters()
        for j,m in pairs(dp) do
            grad_norm=grad_norm+m:norm()^2;
        end
    end
    grad_norm=grad_norm^0.5;
    if grad_norm>self.params.thres then
        lr=lr*self.params.thres/grad_norm;
    end
    for i=1,#self.Modules do
        self.Modules[i]:updateParameters(lr);
    end
end

function model:clear()
    for i=1,#self.Modules do
        self.Modules[i]:zeroGradParameters()
    end
end

function model:train()
    if self.params.saveModel then
        self:saveParams()
    end
    local timer=torch.Timer();
    self.iter=0;
    self.lr=self.params.alpha
    self.mode="test"
    self:test();
    while true do
        self.iter=self.iter+1;
        if self.params.output_file~="" then
            self.output:write("iter  "..self.iter.."\n")
        end
        print("iter  "..self.iter)
        start_halving=false
        if self.iter>self.params.start_halve then
            start_halving=true;
        end
        if start_halving then
            self.lr=self.lr*0.5;
        end
        local open_pos_train_file=io.open(self.params.pos_train_file,"r")
        local open_neg_train_file=io.open(self.params.neg_train_file,"r")
        local End=0;
        local batch_n=1;
        while End==0 do
            batch_n=batch_n+1;
            local time1=timer:time().real;
            self:clear()
            self.mode="train"
            End,self.labels,self.Word_s,self.Mask_s,self.Left_s,self.Padding_s=self.Data:read_train(open_pos_train_file,open_neg_train_file)
            if End==1 then
                break;
            end
            for i=1,#self.Word_s do
                self.Word_s[i]=self.Word_s[i]:cuda()
            end
            self.labels=self.labels:cuda()
            self:model_forward()
            self:model_backward()
            self:update()
            local time2=timer:time().real;
            --[[
            print("self.labels")
            print(self.labels)
            print("self.Word_s")
            print(self.Word_s[1])
            print(self.Word_s[2])
            print("self.Mask_s")
            print(self.Mask_s)
            print("self.Left_s")
            print(self.Left_s)
            print("self.Padding_s")
            print(self.Padding_s[1])
            print(self.Padding_s[2])
            --]]
            self.mode="test"
        end
        open_pos_train_file:close()
        open_neg_train_file:close()
        self.mode="test"
        self:test();
        if self.params.saveModel then
            self:save()
        end
    end
    if self.params.output_file~="" then
        self.output:close()
    end
end

return model
