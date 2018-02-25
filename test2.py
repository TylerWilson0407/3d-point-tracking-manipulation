a = [1, 2, 3]

b = [4, 5, 6]

for x, y in zip(a, b):
    x += 1
    y += 1

print(a, b)

for i in range(len(a)):
    a[i] += 1
    b[i] += 1

print(a, b)