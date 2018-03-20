import csv

lists = [1, 2, 3]
printlist = [[str(i) + ',' + str(lists[i])] for i in range(3)]
print(printlist)
with open('result.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerow(["id,category"])
    spamwriter.writerows(printlist)