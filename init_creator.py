import numpy


name = "explosion.txt"
n = 55
shape = (n, n, n)
e = numpy.ones(shape) * 400000
vx = numpy.ones(shape) * 0
vy = numpy.ones(shape) * 0
vz = numpy.ones(shape) * 0
ro = numpy.ones(shape) * 1.25

e[n // 2 + 1, n // 2 + 1, n // 2 + 1] = 500000

f = open(name, "w")
for x in range(n):
    for y in range(n):
        for z in range(n):
            f.write(f"{x} {y} {z} {ro[x, y, z]} {vx[x, y, z]} {vy[x, y, z]} {vz[x, y, z]} {e[x, y, z]}\n")
f.close()