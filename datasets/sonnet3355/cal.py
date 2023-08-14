lengths = []
cnt = 0
image_size = 12
f = open("sonnet_train.txt", encoding="utf-8")
for line in f:
    lengths.append(len(line.split(" ")))
    if lengths[-1] > image_size ** 2:
        cnt += 1
print(sum(lengths) / len(lengths))
print(max(lengths))
print(cnt / len(lengths))