import sys
def NumToString(numString,Dict):
    G=numString.strip().split(" ")
    String=G[0];
    for i in range(0,len(G)):
        item=G[i]
        if item=="1":
            String=String+" UNK"
        else:
            String=String+" "+str(Dict[int(item)])
    return String
        


A=open(sys.argv[1],"r")
Dict={};
index=0
for line in A:
    index=index+1;
    Dict[index]=line.strip().lower()

input_file=sys.argv[2]
output_file=sys.argv[3]
A=open(input_file,"r")
B=open(output_file,"w")
for line in A:
    t=line.find("|")
    B.write(line[0:t]+" ")
    line=line[t+1:]
    B.write(NumToString(line,Dict)+"\n")
