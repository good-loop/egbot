import json, os, spacy, string, html
nlp = spacy.load('en')

def separateOddTokens(word, odd_tokens):
    for tkn in odd_tokens:
        word = recursivelySeparateOddToken(word, tkn, 0, len(word))
    return word

def recursivelySeparateOddToken(word, char, beg, end):
    #print(word, char, beg, end)
    loc = word.rfind(char, beg, end)
    #print(loc)
    if loc > -1 and len(word) > 1:
        #print(word, loc)
        if word[loc-1] != " " or (loc+1 != len(word) and word[loc+1] != " "):
            if char == "$" or char == "\\" or loc == len(word)-1:
                # print(word[loc-2:loc+3])
                # print("b",word)
                word = word[:loc] + " " + word[loc:]
                #print("a",word)
            elif loc == 0:
                word = word[loc] + " " + word[loc:]
            else:
                word = word[:loc] + " " + word[loc] + " " + word[loc+1:]
            #print(word, char, beg, loc, "---", word[loc-4:loc+5], "----")
            word = recursivelySeparateOddToken(word, char, beg, loc)
        else:
            loc = word.rfind(char, beg, loc)
            #print(word, char, beg, loc, "ooo", word[loc-4:loc+5], "oooo")
            return recursivelySeparateOddToken(word, char, beg, loc+1)
    # else:
    #     loc = word.rfind(char, beg, loc)
    #     if loc > 0:
    #         word = recursivelySeparateOddToken(word, char, beg, loc)
    return word

# def separateOddTokens(packed, odd_tokens):
#     unpacked = packed
#         unpacked = recursivelySeparateOddToken(unpacked, tkn, 0, len(unpacked))
#     return unpacked

# def recursivelySeparateOddToken(packed, char, beg, end):
#     unpacked = packed
#     for word in unpacked:
#         loc = word.rfind(char, 0, len(word))   
#         if loc > 0 and word[loc-1] != " " and len(word) > 1:
#             print("b",word)
#             unpacked = [x for x in [word[:loc],word[loc],word[loc+1:]] if x != '']
#             print(unpacked)
#             for x in unpacked:
#                 if len(x) > 1:
#                     unpacked = recursivelySeparateOddToken(x, char, 0, len(x))
#             print("a",unpacked)
#     return unpacked

def preprocessing(data):
    doc = nlp(data)
    temp = []
    odd_tokens = list(string.punctuation) 
    for tkn in doc:
        word = tkn.text
        word = separateOddTokens(word, odd_tokens)
        temp.append(word)
        #temp.append(separateOddTokens([word], odd_tokens))
    #print("final",temp)
    return ' '.join(temp)

for no in range(1,9):
    contents = ''
    #print()
    infile = "MathStackExchangeAPI_Part_" + str(no) + ".json"
    infile_path = os.path.abspath("../../data/build/"+infile)
    outfile = 'MathStackExchangeAPI_GloVe2.txt'
    outfile_path = os.path.abspath("../../data/glove/"+outfile)

    print("Opening " + infile)
    with open(infile_path) as f:        
        data = json.load(f)
        print("Finished loading")
        for i in range(0, len(data)):
            # adding question body
            contents += ' ' + preprocessing(html.unescape(data[i]["body_markdown"]))
            #print(contents)
            if("answers" in data[i].keys()):
                for ans in data[i]['answers']:
                    # adding answer body
                    contents += ' ' + preprocessing(html.unescape(ans["body_markdown"]))
        print("Saving\n")
        with open(outfile_path, "a") as f:
            f.write(contents)

print("Done :)\n")