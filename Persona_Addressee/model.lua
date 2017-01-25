local model={}
local base_model=torch.reload("../Atten/atten")
setmetatable(model,{ __index = base_model })
model.Data=torch.reload("./data")

function model:lstm_target_()
    local inputs = {}
    for ll=1,self.params.layers do
        local h_ll=nn.Identity()()
        table.insert(inputs,h_ll)
        local c_ll=nn.Identity()()
        table.insert(inputs,c_ll)
    end
    local context,source_mask
    context=nn.Identity()()
    table.insert(inputs,context)
    local x_=nn.Identity()()
    table.insert(inputs,x_)
    source_mask=nn.Identity()()
    table.insert(inputs,source_mask)
    local outputs = {}
    local LookupTable,input_word_embedding
    for ll=1,self.params.layers do
        local prev_h=inputs[ll*2-1];
        local prev_c=inputs[ll*2];
        if ll==1 then x=nn.LookupTable(self.params.vocab_target,self.params.dimension)(x_);
        else x=outputs[(ll-1)*2-1];
        end
        local drop_x=nn.Dropout(self.params.dropout)(x)
        local drop_h=nn.Dropout(self.params.dropout)(inputs[ll*2-1])
        local i2h=nn.Linear(self.params.dimension,4*self.params.dimension,false)(drop_x);
        local h2h=nn.Linear(self.params.dimension,4*self.params.dimension,false)(drop_h);
        local gates;
        if ll==1 then
            local atten_feed=self:attention();
            atten_feed.name='atten_feed';
            local context1=atten_feed({inputs[self.params.layers*2-1],context,source_mask})
            drop_f=nn.Dropout(self.params.dropout)(context1);
            f2h=nn.Linear(self.params.dimension,4*self.params.dimension,false)(drop_f);
            gates=nn.CAddTable()({nn.CAddTable()({i2h,h2h}),f2h});
            local speaker_index=nn.Identity()()
            table.insert(inputs,speaker_index)
            local speaker_v=nn.LookupTable(self.params.SpeakerNum,self.params.dimension)(speaker_index);
            local speaker_v=nn.Dropout(self.params.dropout)(speaker_v);
            if self.params.speakerSetting=="speaker_addressee" then
                local addressee_index=nn.Identity()()
                table.insert(inputs,addressee_index);
                local addressee_v=nn.LookupTable(self.params.AddresseeNum,self.params.dimension)(addressee_index);
                addressee_v=nn.Dropout(self.params.dropout)(addressee_v);
                local h1=nn.Linear(self.params.dimension,self.params.dimension)(speaker_v)
                local h2=nn.Linear(self.params.dimension,self.params.dimension)(addressee_v)
                v=nn.CAddTable()({h1,h2})
                v=nn.Tanh()(v)
                v=nn.Linear(self.params.dimension,4*self.params.dimension)(v)
            elseif self.params.speakerSetting=="speaker" then
                v=nn.Linear(self.params.dimension,4*self.params.dimension)(speaker_v)
            end
            gates=nn.CAddTable()({gates,v})
        else
            gates=nn.CAddTable()({i2h,h2h});
        end
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
    local soft_atten=self:attention();
    soft_atten.name='soft_atten';
    local soft_vector=soft_atten({outputs[self.params.layers*2-1],context,source_mask});
    table.insert(outputs,soft_vector);
    local module= nn.gModule(inputs,outputs);
    module:getParameters():uniform(-self.params.init_weight, self.params.init_weight)
    return module:cuda()
end

function model:train()
    if self.params.saveModel then
        self:saveParams()
    end
    local timer=torch.Timer();
    self.iter=0;
    local start_halving=false
    self.lr=self.params.alpha;
    print("iter  "..self.iter)
    self.mode="test"
    self:test()
    local batch_n=0
    while true do
        self.iter=self.iter+1;
        print("iter  "..self.iter)
        local time_string=os.date("%c")
        print(time_string)
        if self.output~=nil then
            self.output:write(time_string.."\n")
            self.output:write("iter  "..self.iter.."\n");
        end
        if self.params.start_halve~=-1 then
            if self.iter>self.params.start_halve then
                start_halving=true;
            end
        end
        if start_halving then
            self.lr=self.lr*0.5;
        end
        local open_train_file=io.open(self.params.train_file,"r")
        local End,Word_s,Word_t,Mask_s,Mask_t;
        local End=0;
        local batch_n=1;
        local time1=timer:time().real;
        while End==0 do
            batch_n=batch_n+1;
            self:clear()
            End,self.Word_s,self.Word_t,self.Mask_s,self.Mask_t,self.Left_s,self.Left_t,self.Padding_s,self.Padding_t,self.Source,self.Target,self.SpeakerID,self.AddresseeID=
                self.Data:read_train(open_train_file)
            if End==1 then
                break;
            end
            local train_this_batch=false;
            if (self.Word_s:size(2)<60 and self.Word_t:size(2)<60) then
                train_this_batch=true;
            end
            if train_this_batch then
                self.mode="train"
                local time1=timer:time().real;
                self.Word_s=self.Word_s:cuda();
                self.Word_t=self.Word_t:cuda();
                self.Padding_s=self.Padding_s:cuda();
                self:model_forward()
                self:model_backward()
                self:update()
                local time2=timer:time().real;
            end
        end
        open_train_file:close()
        self.mode="test"
        self:test();
        if self.params.saveModel then
            self:save()
        end
        local time2=timer:time().real;
        print(time2-time1)
        if self.iter==self.params.max_iter then
            break;
        end
    end
    if self.output~=nil then
        self.output:close()
    end
end

function model:test()
    local open_train_file
    if self.mode=="dev" then
        open_train_file=io.open(self.params.dev_file,"r")
    elseif self.mode=="test" then
        open_train_file=io.open(self.params.test_file,"r")
    end
    local sum_err_all=0 
    local total_num_all=0
    local End=0
    while End==0 do
        End,self.Word_s,self.Word_t,self.Mask_s,self.Mask_t,self.Left_s,self.Left_t,self.Padding_s,self.Padding_t,self.Source,self.Target,self.SpeakerID,self.AddresseeID=
            self.Data:read_train(open_train_file)
        if #self.Word_s==0 or End==1 then
            break;
        end
        if (self.Word_s:size(2)<self.params.source_max_length and self.Word_t:size(2)<self.params.target_max_length) then
            self.mode="test"
            self.Word_s=self.Word_s:cuda();
            self.Word_t=self.Word_t:cuda();
            self.Padding_s=self.Padding_s:cuda();
            self:model_forward()
            local sum_err,total_num=self:model_backward()
            sum_err_all=sum_err_all+sum_err
            total_num_all=total_num_all+total_num;
        end
    end
    open_train_file:close()
    print("perp ".. 1/torch.exp(-sum_err_all/total_num_all));
    if self.output~=nil then
        self.output:write("standard perp ".. 1/torch.exp(-sum_err_all/total_num_all).."\n")
    end
end

return model
