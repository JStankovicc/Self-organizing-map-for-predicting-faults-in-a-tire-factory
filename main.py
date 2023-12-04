import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# Ucitavanje input fajla
input_data = pd.read_csv('hw2input.csv', sep=' ', dtype=float)

# Brzina obucavanja
eta = 0.001
# Broj neurona u redu
n = 4

def susedi(k):
    j = k % n
    i = k // n
    s = []
    if i > 0:
        s.append((i - 1) * n + j)
    if i < n - 1:
        s.append((i + 1) * n + j)
    if j > 0:
        s.append(i * n + j - 1)
    if j < n - 1:
        s.append(i * n + j + 1)
    return s

mapa_suseda = dict()
for i in range(n * n):
    mapa_suseda[i] = susedi(i)

m = 3
all_inputs = np.array(input_data.loc[:, 'x1':'x3'])

locs = [random.choice(list(range(len(all_inputs)))) for _ in range(16)]
locs.sort()
w = np.array(input_data.loc[locs, 'x1':'x3'])

def dist(w, x):
    return np.sqrt(np.sum((x - w) * (x - w)))

def odredi_najblizi_neuron(w, ulaz):
    mj, md = None, None
    for j in range(len(w)):
        d = dist(ulaz, w[j])
        if md is None or d < md:
            mj, md = j, d
    return mj

def norm(x):
    n = np.sqrt(np.sum(x.dot(x)))
    return x / n

promena_detektovana = True

klase = {i: [] for i in range(16)}
prethodne_klase = {}

iteracija = 0

while promena_detektovana:
    iteracija += 1
    promena_detektovana = False
    for i in range(len(all_inputs)):
        mj = odredi_najblizi_neuron(w, all_inputs[i])
        klase[mj].append(i)
        w[mj] = norm(w[mj] + eta * (all_inputs[i] - w[mj]))
        for j in mapa_suseda[mj]:
            w[j] = norm(w[j] + eta * 0.5 * (all_inputs[i] - w[j]))

    if prethodne_klase != klase:
        promena_detektovana = True

    prethodne_klase = klase
    klase = {i: [] for i in range(16)}
    if i % 10 == 0:
        print('Iteracija', i)

def odredi_klasu(neuron):
    if neuron < 4:
        return 'A'
    elif neuron < 8:
        return 'B'
    return 'C'

test_data = pd.read_csv('hw2test.csv', sep=' ', dtype=float)
td = np.array(test_data.loc[:, 'x1':'x3'])

rezultati = []
for i in range(len(td)):
    neuron = odredi_najblizi_neuron(w, td[i])
    klasa = odredi_klasu(neuron)
    rezultati.append((i+1, neuron, klasa))

# Ispisivanje rezultata testa u datoteku
with open('hw2output.csv', 'w') as f:
    for rezultat in rezultati:
        ulaz, neuron, klasa = rezultat
        f.write(f'Ulaz {ulaz} najblizi neuron {neuron} klasa {klasa}\n')

# Ispisivanje rezultata ulaza iz hw2input u datoteku hw2outputINPUT
#with open('hw2outputINPUT.csv', 'w') as f:
#    for i in range(len(all_inputs)):
#        neuron = odredi_najblizi_neuron(w, all_inputs[i])
#        klasa = odredi_klasu(neuron)
#        f.write(f'Ulaz {i+1} najblizi neuron {neuron} klasa {klasa}\n')

# Iscrtavanje grafika
colors=['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'lime', 'teal', 'navy']

# Plot za x1 and x2 ose za input
plt.figure(figsize=(8, 6))
for i in range(len(all_inputs)):
    neuron = odredi_najblizi_neuron(w, all_inputs[i])
    plt.scatter(all_inputs[i, 0], all_inputs[i, 1], color=colors[neuron], marker='o')

plt.gcf().canvas.manager.set_window_title('x1-x2')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Raspored ulaza (x1-x2)')
plt.savefig('x1_x2.jpg')
plt.show()

# Plot za x1 and x3 ose za input
plt.figure(figsize=(8, 6))
for i in range(len(all_inputs)):
    neuron = odredi_najblizi_neuron(w, all_inputs[i])
    plt.scatter(all_inputs[i, 0], all_inputs[i, 2], color=colors[neuron], marker='s')

plt.gcf().canvas.manager.set_window_title('x1-x3')

plt.xlabel('x1')
plt.ylabel('x3')
plt.title('Raspored ulaza (x1-x3)')
plt.savefig('x1_x3.jpg')
plt.show()

# Plot za x2 and x3 ose za input
plt.figure(figsize=(8, 6))
for i in range(len(all_inputs)):
    neuron = odredi_najblizi_neuron(w, all_inputs[i])
    plt.scatter(all_inputs[i, 1], all_inputs[i, 2], color=colors[neuron], marker='^')

plt.gcf().canvas.manager.set_window_title('x2-x3')


plt.xlabel('x2')
plt.ylabel('x3')
plt.title('Raspored ulaza (x2-x3)')
plt.savefig('x2_x3.jpg')
plt.show()

# 3D plot za input
fig_3d = plt.figure(figsize=(8, 6))
ax_3d_1 = fig_3d.add_subplot(111, projection='3d')

for i in range(len(all_inputs)):
    neuron = odredi_najblizi_neuron(w, all_inputs[i])
    ax_3d_1.scatter(all_inputs[i, 0], all_inputs[i, 1], all_inputs[i, 2], color=colors[neuron], marker='o')

plt.gcf().canvas.manager.set_window_title('3D')

ax_3d_1.set_xlabel('x1')
ax_3d_1.set_ylabel('x2')
ax_3d_1.set_zlabel('x3')
ax_3d_1.set_title('Raspored ulaza (3D)')
plt.savefig('3d.jpg')
plt.show()

# Iscrtavanje grafika za podatke iz hw2test.csv
test_inputs = np.array(test_data.loc[:, 'x1':'x3'])

# Plot za x1 i x2 ose za test
plt.figure(figsize=(8, 6))
for i in range(len(test_inputs)):
    neuron = odredi_najblizi_neuron(w, test_inputs[i])
    plt.scatter(test_inputs[i, 0], test_inputs[i, 1], color=colors[neuron], marker='o')

plt.gcf().canvas.manager.set_window_title('x1-x2 Test')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Raspored ulaza (x1-x2) - Test')
plt.savefig('x1_x2_test.jpg')
plt.show()

# Plot za x1 i x3 ose za test
plt.figure(figsize=(8, 6))
for i in range(len(test_inputs)):
    neuron = odredi_najblizi_neuron(w, test_inputs[i])
    plt.scatter(test_inputs[i, 0], test_inputs[i, 2], color=colors[neuron], marker='s')

plt.gcf().canvas.manager.set_window_title('x1-x3 Test')

plt.xlabel('x1')
plt.ylabel('x3')
plt.title('Raspored ulaza (x1-x3) - Test ')
plt.savefig('x1_x3_test.jpg')
plt.show()

# Plot za x2 i x3 ose za test
plt.figure(figsize=(8, 6))
for i in range(len(test_inputs)):
    neuron = odredi_najblizi_neuron(w, test_inputs[i])
    plt.scatter(test_inputs[i, 1], test_inputs[i, 2], color=colors[neuron], marker='^')

plt.gcf().canvas.manager.set_window_title('x2-x3 Test')

plt.xlabel('x2')
plt.ylabel('x3')
plt.title('Raspored ulaza (x2-x3) - Test')
plt.savefig('x2_x3_test.jpg')
plt.show()

# 3D plot za test podatke
fig_3d_test = plt.figure(figsize=(8, 6))
ax_3d_test = fig_3d_test.add_subplot(111, projection='3d')

for i in range(len(test_inputs)):
    neuron = odredi_najblizi_neuron(w, test_inputs[i])
    ax_3d_test.scatter(test_inputs[i, 0], test_inputs[i, 1], test_inputs[i, 2], color=colors[neuron], marker='o')

plt.gcf().canvas.manager.set_window_title('3D Test')

ax_3d_test.set_xlabel('x1')
ax_3d_test.set_ylabel('x2')
ax_3d_test.set_zlabel('x3')
ax_3d_test.set_title('Raspored ulaza (3D) - Test')
plt.savefig('3d_test.jpg')
plt.show()
