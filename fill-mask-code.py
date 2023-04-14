from transformers import pipeline, pipelines
import numpy as np
import re
import time
import pandas as pd
nlp = pipeline("fill-mask", model='google/muril-base-cased', top_k=500)
def RegexMatch(original, prediction):
    regex = original.replace("<blank>", ".*?", 1)
    reg = re.compile(regex)
    # print(reg.search())
    test =reg.fullmatch(prediction)
    if(test !=None):
        return prediction
    return False

def Restore(text):
    n=0
    words = text.split()
    original = []
    for word in words:
        if "<blank>" in word:
            words[words.index(word)] = nlp.tokenizer.mask_token
            n+=1
            original.append(word)

    masked = ""
    for word in words:
        masked+=" "+word
    output = nlp(masked)
    if(n>1):
        result = []
        answer=[]
        i=0
        for o in range(len(output)):
            if(output[o]==["None"]):
                continue
            temp=[]
            for _ in output[o]:
                # print(_)
                temp.append((i,_["token_str"], _["score"]))
            result.append(temp)
            i+=1
        i=0
        for r in result:
            flag =0
            for _ in r:
                if(RegexMatch(original[i],_[1])):
                    answer.append(RegexMatch(original[i],_[1]))
                    flag=1
                    break
            if(flag==0):
                answer.append("Not Found")
            i+=1
        i=0
        words = text.split()
        for word in words:
            if "<blank>" in word:
                words[words.index(word)] = answer[i]
                i+=1
        restored_text = ""
        for word in words:
            restored_text+=" "+word
        # print(restored_text)
        return restored_text
    else:
        flag=0
        result = []
        answer=[]
        i=0
        for _ in output:
            result.append((i,_["token_str"], _["score"]))
            i+=1
        i=0
        for _ in result:
            # print(_)
            if(RegexMatch(original[i],_[1])):
                    answer.append(RegexMatch(original[0],_[1]))
                    flag=1
                    break
            if(flag==0):
                answer.append("Not Found")
            
        words = text.split()
        for word in words:
            if "<blank>" in word:
                words[words.index(word)] = answer[i]
                i+=1
        restored_text = ""
        for word in words:
            restored_text+=" "+word
        # print(restored_text)
        return restored_text

files = ["list of strings of damaged file names with illegible characters replaced with "<blank>""]

for file in files:
    start=time.time()
    with open(file) as f:
        text = f.read().split()
    n = 200
    text=[' '.join(text[i:i+n]) for i in range(0,len(text),n)]
    start=time.time()
    f= open("restored_"+file, "w")
    for line in text:
        # print("line"+line)
        f.write(Restore(line))
    end = time.time()
    print(end-start)
# print(err)
