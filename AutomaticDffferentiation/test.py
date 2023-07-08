def compute(a, b, c):
    print(a + b + c)


vec = [1, 2, 3]

compute(*[x for x in vec])
