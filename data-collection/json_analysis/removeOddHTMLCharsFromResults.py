import os, sys

print("Removing Forward Feed characters from results file \n")

if not sys.argv[1]:
    print("Error: I was expecting the file path as an argument ...")
    exit(1)

# weird html chars that have to be removed 
removable = ["&#xc;", "&#x0;", "&#x2;", "&#x12;", "&#x13;", "&#x14;", "&#x1f;", "&#x1d;"]

filepath = sys.argv[1]
print("Opening file")
with open(filepath) as f:
    data = f.read()
    print("Finished loading")

    for htmlChar in removable:
        if htmlChar in data:
            print ("found ", htmlChar)
            data = data.replace(htmlChar,'')

newfilepath = filepath+".xml"

with open(newfilepath, "w+") as f:
    f.write(data)
        
print("Done :)")


