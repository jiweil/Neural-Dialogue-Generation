local decode_model={}
local stringx = require('pl.stringx')
local base_model=torch.reload("../Atten/atten")
setmetatable(decode_model,{ __index = base_model })

function decode_model:Initial(params)
    local params_file=torch.DiskFile(params.params_file,"r"):binary();
    local model_params=params_file:readObject()
    params_file:close();
    for i,v in pairs(model_params)do 
        if params[i]==nil then
            params[i]=v;
        end
    end
    self.params=params;
    self:DecoderInitial()
    if self.params.MMI then
        self:InitialReverse()
    end
    self.mode="test"
    --self:test()
    self.mode="decoding"
end

function decode_model:DecoderInitial()
    self.Data:Initial(self.params)
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
    self.store_t={}
    self:readModel()
    self:ReadDict()
end

function decode_model:InitialReverse()
    self.MMI_model=torch.reload("../Atten/atten");
    local file=torch.DiskFile(self.params.MMI_params_file,"r"):binary();
    local params=file:readObject();
    params.batch_size=self.params.batch_size
    params.onlyPred=true
    file:close();
    self.MMI_model.mode="test"
    self.MMI_model:Initial(params)
    local file = torch.DiskFile(self.params.MMI_model_file,"r"):binary()
    local model_params=file:readObject();
    file:close();
    for i=1,#self.MMI_model.Modules do
        local parameter,_=self.MMI_model.Modules[i]:parameters()
        for j=1,#parameter do
            parameter[j]:copy(model_params[i][j]);
        end
    end
    print("MMI model read done")
end

function decode_model:sample()
    self:model_forward()
    local batch_max_dec_length;
    if self.params.max_length==0 then batch_max_dec_length=torch.ceil(1.5*self.Word_s:size(2));
    else batch_max_dec_length=self.params.max_length;
    end
    local completed_history={};
    local beamHistory=torch.ones(self.Word_s:size(1),batch_max_dec_length):cuda()
    for t=1,batch_max_dec_length do
        local lstm_input=self:clone_(self.last)
        table.insert(lstm_input,self.context);
        if t==1 then 
            table.insert(lstm_input,torch.Tensor(self.Word_s:size(1)):fill(self.Data.EOS):cuda());
        else
            table.insert(lstm_input,beamHistory:select(2,t-1));
        end
        table.insert(lstm_input,self.Padding_s);
        self.lstms_t[1]:evaluate()
        local output=self.lstms_t[1]:forward(lstm_input);
        self.last={};
        self.store_t[t]={}
        for i=1,2*self.params.layers do
            self.store_t[t][i]=self:copy(output[i]);
            self.last[i]=self:copy(output[i]);
        end
        local prob=self.softmax:forward({output[#output],torch.Tensor(output[#output]:size(1)):fill(1):cuda()})
        prob=prob[2]
        prob=nn.Exp():cuda():forward(prob);
        if not self.params.allowUNK then
            prob:sub(1,-1,1,1):fill(0)
        end
        if self.params.setting=="StochasticGreedy" then
            select_P,select_words=torch.topk(prob,self.params.StochasticGreedyNum,2,true,true);
            prob=nn.Normalize(1):cuda():forward(select_P)
            next_words_index=torch.multinomial(prob, 1);
            next_words=torch.Tensor(self.Word_s:size(1),1):fill(0):cuda()
            for i=1,self.Word_s:size(1) do
                next_words[i][1]=select_words[i][next_words_index[i][1]]
            end
        else
            next_words=torch.multinomial(prob, 1);
        end
        local end_boolean_index=next_words:eq(self.Data.EOT)
        if end_boolean_index:sum()~=0 then
            local end_examples=torch.range(1,self.Word_s:size(1)):cuda()[end_boolean_index];
            for i=1,end_examples:size(1)do
                local example_index=end_examples[i];
                if completed_history[example_index]==nil then
                    if t~=1 then
                        completed_history[example_index]=beamHistory:sub(example_index,example_index,1,t-1);
                    else
                        completed_history[example_index]=torch.Tensor(1,1):fill(1):cuda()
                    end
                end
            end
        end
        beamHistory:select(2,t):copy(next_words)
    end
    for i=1,self.Word_s:size(1) do
        if completed_history[i]==nil then
            completed_history[i]=beamHistory:sub(i,i,1,-1);
        end
    end
    return completed_history;
end

function decode_model:decode_BS()
    self:model_forward()
    local span_index1=torch.expand(torch.reshape(torch.expand(torch.reshape(torch.range(1,self.params.beam_size),self.params.beam_size,1),self.params.beam_size,self.params.beam_size),1,self.params.beam_size*self.params.beam_size),self.Word_s:size(1),self.params.beam_size*self.params.beam_size):cuda();
    -- if batch_size=3  beam_size=2 span_index1 is
    --1 1 2 2
    --1 1 2 2 
    --1 1 2 2
    local span_index2=torch.expand(torch.reshape(self.params.beam_size*(torch.range(1,self.Word_s:size(1))-1),self.Word_s:size(1),1),self.Word_s:size(1),self.params.beam_size):cuda();
    -- if batch_size=3  beam_size=2 span_index2 is
    --0 0
    --2 2
    --4 4
    local span_index3=torch.expand(torch.reshape((torch.range(1,self.Word_s:size(1))-1)*self.params.beam_size*self.params.beam_size,self.Word_s:size(1),1),self.Word_s:size(1),self.params.beam_size):cuda();
    -- if batch_size=3  beam_size=2 span_index3 is
    -- 0 0
    -- 4 4
    -- 8 8
    local diverse_part 
    --penalizing intra-insibling
    if self.params.DiverseRate~=0 and self.params.DiverseRate~=nil then
        diverse_part=self.params.DiverseRate*torch.expand(torch.reshape(torch.range(0,self.params.beam_size-1),1,self.params.beam_size),self.params.beam_size*self.Word_s:size(1),self.params.beam_size):cuda();
    end
    local scores
    local completed_history={};
    local batch_max_dec_length;
    if self.params.max_length==0 then batch_max_dec_length=torch.ceil(1.5*self.Word_s:size(2));
    else batch_max_dec_length=self.params.max_length;
    end
    if self.params.target_length~=0 then 
        batch_max_dec_length=self.params.target_length+4
    end
    local beamHistory=torch.ones(self.Word_s:size(1)*self.params.beam_size,batch_max_dec_length):cuda()
    for t=1,batch_max_dec_length do
        local lstm_input=self:clone_(self.last)
        if t==2 then
            local context=torch.reshape(self.context,self.context:size(1),1,self.context:size(2),self.context:size(3))
            context=torch.expand(context,context:size(1),self.params.beam_size,context:size(3),context:size(4));
            self.context_beam=torch.reshape(context,context:size(1)*context:size(2),context:size(3),context:size(4));
            local padding=torch.reshape(self.Padding_s,self.Padding_s:size(1),1,self.Padding_s:size(2));
            padding=torch.expand(padding,padding:size(1),self.params.beam_size,padding:size(3));
            self.Padding_s_beam=torch.reshape(padding,padding:size(1)*padding:size(2),padding:size(3));
        end
        if t==1 then 
            table.insert(lstm_input,self.context);
            table.insert(lstm_input,torch.Tensor(self.Word_s:size(1)):fill(self.Data.EOS):cuda());
            table.insert(lstm_input,self.Padding_s);
        else
            table.insert(lstm_input,self.context_beam);
            table.insert(lstm_input,beamHistory:select(2,t-1));
            table.insert(lstm_input,self.Padding_s_beam);
        end
        self.lstms_t[1]:evaluate()
        local output=self.lstms_t[1]:forward(lstm_input);
        self.last={};
        for i=1,2*self.params.layers do
            self.last[i]=self:copy(output[i]);
        end
        local full_logP=self.softmax:forward({output[#output],torch.Tensor(output[#output]:size(1)):fill(1):cuda()})
        full_logP=full_logP[2]
        if not self.params.allowUNK then
            full_logP:sub(1,-1,1,1):fill(-100)
        end
        local select_logP,select_words=torch.topk(full_logP,self.params.beam_size,2,true,true);
        if self.params.DiverseRate~=0 and self.params.DiverseRate~=nil and t~=1 then
            select_logP=select_logP-diverse_part
        end
        if t==1 then
            beamHistory:select(2,1):copy(torch.reshape(select_words,select_words:nElement(),1))
            scores=select_logP;
            for i=1,2*self.params.layers do
                local vector=self:copy(self.last[i]);
                vector=torch.reshape(vector,self.Word_s:size(1),1,self.params.dimension);
                vector=torch.expand(vector,self.Word_s:size(1),self.params.beam_size,self.params.dimension);
                self.last[i]=torch.reshape(vector,self.Word_s:size(1)*self.params.beam_size,self.params.dimension);
            end
        else
            local reshape_score=torch.reshape(scores,scores:nElement(),1);
            local new_score=torch.expand(reshape_score,reshape_score:size(1),self.params.beam_size)+select_logP;
            -- (batch*beam)*beam
            new_score=torch.reshape(new_score,self.Word_s:size(1),self.params.beam_size*self.params.beam_size);
            --batch*(beam*beam)
            local reshape_words=torch.reshape(select_words,self.Word_s:size(1),self.params.beam_size*self.params.beam_size);
            local end_boolean_index=reshape_words:eq(self.Data.EOT)
            local extract_scores
            if end_boolean_index:sum()~=0 then
                extract_scores=new_score[end_boolean_index]
                new_score[end_boolean_index]=-10000;
            end
            if end_boolean_index:sum()~=0 and t>self.params.target_length then
                local end_index=torch.range(1,reshape_words:nElement()):cuda()[end_boolean_index]
                local end_examples=torch.floor((end_index-1)/(self.params.beam_size*self.params.beam_size))+1
                local end_fathers=span_index1[end_boolean_index]
                local end_index_in_history=(end_examples-1)*self.params.beam_size+end_fathers;
                end_index_in_history=end_index_in_history:long()
                local extract_history=beamHistory:index(1,end_index_in_history:cuda());
                for i=1,end_examples:size(1)do
                    local example_index=end_examples[i];
                    local min_length,max_length
                    if self.params.min_length==0 and self.params.max_length==0 then
                        local input_length=self.Padding_s:select(1,example_index):sum()
                        min_length=torch.floor(input_length*0.8);
                        max_length=torch.ceil(input_length*1.5);
                    else min_length=self.params.min_length
                        max_length=self.params.max_length
                    end
                    if t<=max_length and t>=min_length then
                        if self.params.target_length==0 or completed_history[example_index]==nil then
                            local his_sub=extract_history:sub(i,i,1,t-1);
                            if his_sub:ge(self.Data.EOS):sum()==0 then
                                if completed_history[example_index]==nil then
                                    completed_history[example_index]={};
                                end

                                local num_instance=#completed_history[example_index]+1
                                completed_history[example_index][num_instance]={}
                                completed_history[example_index][num_instance].string=his_sub
                                completed_history[example_index][num_instance].score=extract_scores[i]/(completed_history[example_index][num_instance].string:size(2)+1);
                            end
                        end
                    end
                end
            end
            local select_scores,select_index=torch.topk(new_score,self.params.beam_size,2,true,true);
            select_index=select_index:double():cuda()
            scores=torch.reshape(select_scores,select_scores:nElement(),1);
            local select_fathers=(torch.floor((select_index-1)/self.params.beam_size)+1);
            local select_index_in_history=(span_index2+select_fathers)
            select_index_in_history=torch.reshape(select_index_in_history,select_index_in_history:nElement()):long()
            beamHistory=beamHistory:index(1,select_index_in_history);
            local new_added_words_index=torch.reshape(span_index3+select_index,select_index:nElement()):long();
            local new_added_words=torch.reshape(reshape_words,reshape_words:nElement()):index(1,new_added_words_index);
            beamHistory:select(2,t):copy(new_added_words);
            for i=1,2*self.params.layers do
                self.last[i]=self:copy(self.last[i]:index(1,select_index_in_history));
            end
        end
    end
    for i=1,self.Word_s:size(1) do
        if completed_history[i]==nil then
            completed_history[i]={}
            completed_history[i][1]={};
            completed_history[i][1].string=beamHistory:sub((i-1)*self.params.beam_size+1,(i-1)*self.params.beam_size+1,1,-1);
            completed_history[i][1].score=scores[(i-1)*self.params.beam_size+1][1]
        end
    end
    if self.params.MMI then
        completed_history=self:MMI(completed_history)
    end
    return completed_history;
end


function decode_model:MMI(completed_history)
    local MMI_source={}
    local MMI_target={}
    local length={}
    local total_num=0;
    for i=1,self.Word_s:size(1) do
        total_num=total_num+#completed_history[i]
        for j=1,#completed_history[i] do
            MMI_source[#MMI_source+1]=completed_history[i][j].string;
            MMI_target[#MMI_target+1]=torch.cat(torch.Tensor({{self.Data.EOS}}),torch.cat(self.Source[i],torch.Tensor({self.Data.EOT})))
        end
    end

    local reverse_score
    local batch_size=10*self.params.batch_size
    for i=1,torch.ceil(#MMI_source/batch_size)do
        local Begin=batch_size*(i-1)+1;
        local End=batch_size*i;
        if End>#MMI_source then
            End=#MMI_source;
        end
        self.MMI_model.source={}
        self.MMI_model.target={}
        for j=Begin,End do
            self.MMI_model.source[#self.MMI_model.source+1]=MMI_source[j];
            self.MMI_model.target[#self.MMI_model.target+1]=MMI_target[j];
        end
        self.MMI_model.Word_s,self.MMI_model.Mask_s,self.MMI_model.Left_s,self.MMI_model.Padding_s=self.MMI_model.Data:get_batch(self.MMI_model.source,true)
        self.MMI_model.Word_t,self.MMI_model.Mask_t,self.MMI_model.Left_t,self.MMI_model.Padding_t=self.MMI_model.Data:get_batch(self.MMI_model.target,false)

        self.MMI_model.mode="test";
        self.MMI_model:model_forward()
        local current_score=self.MMI_model:SentencePpl()
        if reverse_score==nil then
            reverse_score=current_score;
        else
            reverse_score=torch.cat(reverse_score,current_score,1)
        end
    end
    num=0;
    --print(reverse_score)
    for i=1,self.Word_s:size(1) do
        for j=1,#completed_history[i] do
            num=num+1
            completed_history[i][j].score=0.5*completed_history[i][j].score+0.5*reverse_score[num]
        end
    end
    return completed_history;
end

function decode_model:decode()
    local open_train_file=io.open(self.params.InputFile,"r")
    local open_write_file=io.open(self.params.OutputFile,"w")
    local End=0
    local batch_n=0;
    local n_decode_instance=0
    local timer=torch.Timer();
    while End==0 do
        local time1=timer:time().real;
        End,self.Word_s,self.Word_t,self.Mask_s,self.Mask_t,self.Left_s,self.Left_t,self.Padding_s,self.Padding_t,self.Source,self.Target=  self.Data:read_train(open_train_file)
        if #self.Word_s==0 then
            break
        end
        n_decode_instance=n_decode_instance+self.Word_s:size(1)
        if self.params.max_decoded_num~=0 then
            if n_decode_instance>self.params.max_decoded_num then
                break
            end
        end
        batch_n=batch_n+1
        if batch_n%5000/(self.params.batch_size/128)==0  then
            print(batch_n)
            local time_string=os.date("%c")
            print(time_string)
        end
        self.mode="decoding"
        self.Word_s=self.Word_s:cuda();
        self.Padding_s=self.Padding_s:cuda();
        local completed_history
        if self.params.setting=="sampling" or self.params.setting=="StochasticGreedy" then
            completed_history=self:sample()
        else
            completed_history=self:decode_BS()
        end
        self:OutPut(completed_history,open_write_file,batch_n)
        assert(self.params.setting~=nil)
        local time2=timer:time().real;
        --print(time2-time1)
        if End==1 then
            break;
        end
    end
    open_write_file:close()
end


function decode_model:OutPut(completed_history,open_write_file,batch_n)
    if self.params.setting=="sampling" or self.params.setting=="StochasticGreedy" then
        for i=1,self.Word_s:size(1) do
            if self.params.output_source_target_side_by_side then
                local print_string=self:IndexToWord(self.Source[i][1])
                print_string=print_string.."|"
                print_string=print_string..self:IndexToWord(completed_history[i][1])
                open_write_file:write(print_string.."\n")
                if self.params.PrintOutIllustrationSample and i<=5 then
                    print(print_string)
                end
            else
                local index=self.params.batch_size*(batch_n-1)+i;
                local print_string=index.." "..completed_history[i]:size(2).." ".."0"
                print_string=print_string.." "..self:IndexToWord(completed_history[i][1]);
                open_write_file:write(print_string.."\n")
                if self.params.PrintOutIllustrationSample and i<=5 then
                    print(print_string)
                end
            end
        end
    else
        for i=1,self.Word_s:size(1) do
            if completed_history[i]~=nil then
                local best_score=-math.huge
                local best_output;
                for j=1,#completed_history[i] do
                    local current_score=completed_history[i][j].score
                    if self.params.NBest then
                        local index=self.params.batch_size*(batch_n-1)+i;
                        local print_string=index.." "..completed_history[i][j].string:size(2).." "..current_score
                        if self.params.output_source_target_side_by_side then
                            print_string=print_string.." "..self:IndexToWord(self.Source[i][1]).."|"
                        end
                        print_string=print_string..self:IndexToWord(completed_history[i][j].string)
                        open_write_file:write(print_string.."\n");
                        if self.params.PrintOutIllustrationSample and i<=5 then
                            print(print_string)
                        end
                    end
                    if current_score>best_score then
                        best_score=current_score;
                        best_output=completed_history[i][j].string;
                    end
                end
                if not self.params.NBest then
                    if self.params.output_source_target_side_by_side then
                        local print_string=self:IndexToWord(self.Source[i][1])
                        print_string=print_string.."|"
                        print_string=print_string..self:IndexToWord(best_output[1])
                        open_write_file:write(print_string.."\n");
                        if self.params.PrintOutIllustrationSample and i<=5 then
                            print(print_string)
                        end
                    else
                        local print_string=index.." "..best_output:size(2).." "..best_score
                        print_string=print_string.." "..self:IndexToWord(best_output[1])
                        open_write_file:write(print_string.."\n");
                        if self.params.PrintOutIllustrationSample and i<=5 then
                            print(print_string)
                        end
                    end
                end
            end
        end
    end
end

function decode_model:CollectSampleForIllustration(filename)
    local open_=io.open(filename)
    self.SampleSource={}
    self.SampleTarget={}
    local num=0
    while true do
        local line=open_:read("*line")
        if line==nil then break end
        num=num+1
        two_strings=stringx.split(line,"|")
        self.SampleSource[num]=self.Data:split(stringx.strip(two_strings[1]))
        self.SampleTarget[num]=torch.cat(torch.Tensor({{self.Data.EOS}}),torch.cat(self.Data:split(stringx.strip(two_strings[2])),torch.Tensor({self.Data.EOT})))
        if num==5 then break end
    end
end

function decode_model:DecodeIllustrationSample(open_write_file)
    self.Source=self.SampleSource
    self.Word_s,self.Mask_s,self.Left_s,self.Padding_s=self.Data:get_batch(self.SampleSource,true)
    self.Word_t,self.Mask_t,self.Left_t,self.Padding_t=self.Data:get_batch(self.SampleTarget,false)
    self.mode="decoding"
    self:model_forward()
    print(self.params.setting)
    if self.params.setting=="sampling" or self.params.setting=="StochasticGreedy" then
        completed_history=self:sample()
    else
        completed_history=self:decode_BS()
    end
    self:OutPut(completed_history,open_write_file)   
end

return decode_model
