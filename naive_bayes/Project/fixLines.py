
fix = "otherBooks/macbeth.txt"

unedited = open(fix, "r")
edited = open("book" + str(6) + ".txt", "w")
currBook = []
line = unedited.readline()

write = False
currPar = ""
while line:
    if "!-!-!" in line:
        write = True
    if write:
        currPar = currPar.replace("\n", " ")
        currPar += "\n"
        edited.write(currPar)
        currPar = ""
        write = False
    else:
        currPar += line
    line = unedited.readline()
