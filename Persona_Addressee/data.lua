local Data={}
local base_data=torch.reload("../Atten/data")
setmetatable(Data,{ __index = base_data })
local stringx = require('pl.stringx')
function Data:read_train(open_train_file)
    local Y={}; 
    local Source={}; 
    local Target={};
    local i=0;
    local End=0;
    local SpeakerID
    local AddresseeID
    while 1==1 do
        i=i+1;
        local str=open_train_file:read("*line");
        if str==nil then
            End=1
            break;
        end
        two_strings=stringx.split(str,"|")
        assert(self.params.reverse~=nil)
        local addressee_line,addressee_id
        if self.params.speakerSetting=="speaker_addressee" then
            space=two_strings[1]:find(" ")
            addressee_id=tonumber(two_strings[1]:sub(1,space-1));
            addressee_line=two_strings[1]:sub(space+1,-1)
        else
            addressee_line=two_strings[1]
        end
        local speaker_line,speaker_id
        if self.params.speakerSetting=="speaker" or self.params.speakerSetting=="speaker_addressee"  then
            space=two_strings[2]:find(" ")
            speaker_id=tonumber(two_strings[2]:sub(1,space-1));
            speaker_line=two_strings[2]:sub(space+1,-1)
        else
            speaker_line=two_strings[2]
        end
        if addressee_id~=nil then
            if AddresseeID==nil then
                AddresseeID=torch.Tensor({addressee_id})
            else
                AddresseeID=torch.cat(AddresseeID,torch.Tensor({addressee_id}),1)
            end
        end
        if speaker_id~=nil then
            if SpeakerID==nil then
                SpeakerID=torch.Tensor({speaker_id});
            else
                SpeakerID=torch.cat(SpeakerID,torch.Tensor({speaker_id}),1)
            end
        end
        if self.params.reverse then
            Source[i]=reverse(self:split(stringx.strip(addressee_line)))
        else Source[i]=self:split(stringx.strip(addressee_line))
        end
        if self.params.reverse_target then
            local C=reverse(self:split(stringx.strip(speaker_line)))
            Target[i]=torch.cat(torch.Tensor({{self.EOS}}),torch.cat(C,torch.Tensor({self.EOT})))
        else
            Target[i]=torch.cat(torch.Tensor({{self.EOS}}),torch.cat(self:split(stringx.strip(speaker_line)),torch.Tensor({self.EOT})))
        end
        if i==self.params.batch_size then
            break;
        end
    end
    if End==1 then
        return End,{},{},{},{},{},{}
    end
    Words_s,Masks_s,Left_s,Padding_s=self:get_batch(Source,true)
    Words_t,Masks_t,Left_t,Padding_t=self:get_batch(Target,false)
    AddresseeID=AddresseeID:cuda()
    SpeakerID=SpeakerID:cuda()
    return End,Words_s,Words_t,Masks_s,Masks_t,Left_s,Left_t,Padding_s,Padding_t,Source,Target,SpeakerID,AddresseeID
end
return Data
