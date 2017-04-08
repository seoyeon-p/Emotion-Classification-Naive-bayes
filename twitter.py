dict = {}

def readFile(filename):
    f = open(filename, "r", encoding = "utf-8", errors="ignore")
    foutput = open("twitteroutput.txt","w",encoding = "utf-8",errors="ignore")
    label_list = []
    while True:
        line = f.readline()
        if not line : break
        sentence = line.split("\t")[1]
        word_list = sentence.split()

        for word in sentence.split():
            if word[0] == "@":
                word_list.remove(word)
        final = " ".join(word_list)
        final = final.replace("#","")
        label = line.split("\t")[-1]
        label = label.replace(":","")

        foutput.write(final + "\t" + label)
        if label in dict:
            tmp = dict[label]
            tmp.append(final)
            dict[label] = tmp
        else:
            dict[label]=[final]

    for key in dict:
        file_name = key.strip() + ".txt"
        femotion = open(file_name,"w",encoding="utf-8")
        for i in range(0,len(dict[key])):
            femotion.write(dict[key][i]+"\n")

    foutput.close()
    femotion.close()
    f.close()

readFile("Jan9-2012-tweets-clean.txt")