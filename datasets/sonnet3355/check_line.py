data = open("sonnet_valid.txt", "r", encoding='utf-8')
cnt = 0
max_n_words = 0
for line in data:
    sents = line.split("<eos>")[:-1]
    words = line.split()
    max_n_words = max(len(words), max_n_words)
    if len(sents) != 14:
        print(line)
        cnt += 1
print(cnt, max_n_words)
