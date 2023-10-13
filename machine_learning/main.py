def T(n):
    c, x = 1, 3
    print(f'T({n})')
    if n == 1:
        return c
    else:
        return x*T(n/x) + c*n