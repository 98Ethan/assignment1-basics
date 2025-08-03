s = 256
L = 4
d = 512
V = 10000


P = L * (16 * d**2 + 2*d) + V*d + d

# A = 14 L b s d  +  b s d  +  b s V
A_per_batch = 14 * L * s * d + s * d + s * V


# 40 gb
ram = 80 * 1024**3

# if float32
num_elements = ram / 4


batch = (num_elements - P)/ A_per_batch

print(f"Maximum batch size: {batch:.0f}")