import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import random
from copy import deepcopy

def stworz_wektor_pojazdy(n, ust_pojazd):
    wynik = np.random.choice(np.arange(1, ust_pojazd + 1), size = n)
    wektor_pojazdy = []
    for i in range(0, ust_pojazd):
        pom = (np.array(np.where(wynik == i+1)) + 1).flatten()
        wektor_pojazdy.append(pom)
    return wektor_pojazdy

def potasuj(wektor_pojazdy, ust_pojazd):
    for i in range(0, ust_pojazd):
        np.random.shuffle(wektor_pojazdy[i])
    return wektor_pojazdy

def losuj_liczbe_bez_srodka(n, x):
    # n - ile ciezarowek
    # x - z ktorej ciezarowki wybrany do swapowania
    # return ciezarowke do ktorej trafia swapowany
    liczba = x
    while liczba == x:
        liczba = np.random.choice(np.arange(0, n), size = 1)[0]
    return liczba


def swap(wektor_pojazdy, klienci):
    kopia = deepcopy(wektor_pojazdy)

    if len(wektor_pojazdy) == 1:
        return kopia  # gdyby byla tylko jedna ciezarowka, to nie ma co swapowac

    podmiana = np.random.choice(np.arange(1, len(klienci) + 1), size=1)[0]
    for i in range(0, len(wektor_pojazdy)):
        if podmiana in wektor_pojazdy[i]:
            kopia[i] = wektor_pojazdy[i][wektor_pojazdy[i] != podmiana]
            pom = i
            break
    wylosowana = losuj_liczbe_bez_srodka(len(wektor_pojazdy), pom)
    kopia[wylosowana] = np.append(wektor_pojazdy[wylosowana], podmiana)
    return kopia


def reorder(wektor_pojazdy):
    kopia = deepcopy(wektor_pojazdy)
    ciezarowa_do_zmiany = np.random.choice(len(wektor_pojazdy), size=1)[0]

    temp_id = 0
    while len(wektor_pojazdy[ciezarowa_do_zmiany]) < 2:
        ciezarowa_do_zmiany = np.random.choice(len(wektor_pojazdy), size=1)[0]
        temp_id += 1
        if temp_id > 40:
            return kopia  # gdy nie da rady wybrac takich, ktore maja po 2 dlugosci

    idx_change = np.random.choice(len(wektor_pojazdy[ciezarowa_do_zmiany]), size=2, replace=False)

    temp = kopia[ciezarowa_do_zmiany][idx_change[0]]
    kopia[ciezarowa_do_zmiany][idx_change[0]] = wektor_pojazdy[ciezarowa_do_zmiany][idx_change[1]]
    kopia[ciezarowa_do_zmiany][idx_change[1]] = temp
    return kopia


def euclidean_dist(point1, point2):
    # point1, 2 są array dwuelementowymi (x,y)
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def odleglosci(coord_bazy, coord_klientow):
    liczba_klientow = len(coord_klientow)
    # coord_klientow to tablica o dim=(n,2), gdzie w kazdym wierszu wspolrzedne (x,y) klienta
    odleglosc = np.zeros((liczba_klientow + 1, liczba_klientow + 1), dtype=float)

    for i in range(liczba_klientow):
        odleglosc.itemset((0, i + 1), euclidean_dist(coord_bazy, coord_klientow[i]))
        odleglosc.itemset((i + 1, 0), odleglosc[0, i + 1])
        if i == 0:
            continue
        for j in range(i + 1, liczba_klientow + 1):
            odleglosc.itemset((i, j), euclidean_dist(coord_klientow[i - 1], coord_klientow[j - 1]))
            odleglosc.itemset((j, i), odleglosc[i, j])
    return odleglosc

def sprawdz_ladownosc(wektor_pojazdy, ladownosc, max_ladownosc):
    # max_ladownosc - liczba
    # ladownosc - wektor dla kazdego klienta jego kg pakunku
    for i in range(len(wektor_pojazdy)):
        suma = 0
        for j in range(len(wektor_pojazdy[i])):
            suma += ladownosc[wektor_pojazdy[i][j] - 1]
        if suma > max_ladownosc:
            return False
    return True

def sprawdz_odleglosci(wektor_pojazdy, max_odleglosc):
    distances = odleglosci(baza, klienci)
    for i in range(len(wektor_pojazdy)):
        suma = 0
        if len(wektor_pojazdy[i]) != 0:
            suma = suma + distances[0][wektor_pojazdy[i][0]]
            for j in range(len(wektor_pojazdy[i]) - 1):
                suma = suma + distances[wektor_pojazdy[i][j]][wektor_pojazdy[i][j+1]]
            suma = suma + distances[0][wektor_pojazdy[i][len(wektor_pojazdy[i]) - 1]]
            if suma > max_odleglosc:
                return False
    return True

def liczenie_odleglosci(wektor_pojazdy, baza, klienci):
    distances = odleglosci(baza, klienci)
    suma = 0
    for i in range(len(wektor_pojazdy)):
        if len(wektor_pojazdy[i]) != 0:
            suma = suma + distances[0,wektor_pojazdy[i][0]]
            for j in range(len(wektor_pojazdy[i]) - 1):
                suma = suma + distances[wektor_pojazdy[i][j]][wektor_pojazdy[i][j+1]]
            suma = suma + distances[0][wektor_pojazdy[i][len(wektor_pojazdy[i]) - 1]]
    return suma

alpha = 1e-2
beta = 1

def kara(wektor_pojazdy, baza, klienci, alpha, beta):
    ile_aut = 0
    for i in range(len(wektor_pojazdy)):
        if not len(wektor_pojazdy[i]):
            ile_aut += 1
    return alpha * ile_aut + beta * liczenie_odleglosci(wektor_pojazdy, baza, klienci)


def plot_paths(baza, klienci, wektor_pojazdy):
    # najpierw usuwamy nieuzyte ciezarowki w wektor_pojazdy
    k = i = 0
    while k < len(wektor_pojazdy):
        if len(wektor_pojazdy[i]) == 0:
            wektor_pojazdy.pop(i)
            i -= 1
        k += 1
        i += 1

    # tworzymy palete potrzebnych kolorow (co najwyzej uzyjemy 15, więcej nie przewiduje)
    kolory = plt.rcParams['axes.prop_cycle'].by_key()['color'] + \
             ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5']

    # na czarno rysujemy klientow, a roznymi kolorami trasy roznych ciezarowek
    plt.plot(klienci[:, 0], klienci[:, 1], c='black', marker='o', linewidth=0)
    plt.plot(baza[0], baza[1], 'r+')
    for i in range(len(wektor_pojazdy)):
        coords = np.concatenate(([baza], klienci[wektor_pojazdy[i] - 1], [baza]))
        plt.plot(coords[:, 0], coords[:, 1], kolory[i])
    plt.show()

# to mamy zadane
n = 8
baza = np.random.uniform(low = -1.0, high = 1.0, size = 2)
klienci = np.random.uniform(low = -1.0, high = 1.0, size = (n,2))

D = 1e3
d = np.ones(shape=len(klienci))
capacity = 2e10


def simulated_annealing_step(ust_pojazd, baza, klienci, T, num_iterations, d, capacity, D, alpha, beta):
    obecne = stworz_wektor_pojazdy(len(klienci), ust_pojazd)

    # kontrola odleglosci trasy i ładowności wylosowanego początkowego stanu
    # najpierw ladownosc, bo jak ladownosci nie bedzie, to zadne potasowanie miedzy soba nie da rozwiazania
    kontrolny_id = 0
    while not sprawdz_ladownosc(obecne, d, capacity):
        obecne = stworz_wektor_pojazdy(len(klienci), ust_pojazd)
        kontrolny_id += 1
        if kontrolny_id > 500:
            print('Niemożliwe znalezienie kombinacji spełniającej warunki.')
            return

    obecne = potasuj(obecne, ust_pojazd)
    kontrolny_id = 0
    while not sprawdz_odleglosci(obecne, D):
        obecne = potasuj(obecne, ust_pojazd)
        kontrolny_id += 1
        if kontrolny_id > 500:
            print('Niemożliwe znalezienie kombinacji spełniającej warunki.')
            return

    kontrolny_id = 0
    for j in range(num_iterations):

        if random.randint(0, 1):
            nowe_propozycja = reorder(obecne)
        else:
            nowe_propozycja = swap(obecne, klienci)
        if sprawdz_ladownosc(nowe_propozycja, d, capacity) and sprawdz_odleglosci(nowe_propozycja, D):
            nowe = nowe_propozycja
            obecny_koszt = kara(obecne, baza, klienci, alpha, beta)
            nowy_koszt = kara(nowe, baza, klienci, alpha, beta)
            if nowy_koszt < obecny_koszt or (nowy_koszt != obecny_koszt and \
                                             np.exp(-(nowy_koszt - obecny_koszt) / T) > np.random.uniform(0, 1, size=1)[
                                                 0]):
                obecne = nowe
        else:
            kontrolny_id += 1
            if kontrolny_id > num_iterations / 2:
                break
        T = T * 0.95

    #plot_paths(baza, klienci, obecne)

    return (obecne, obecny_koszt)

def sa(T_base, baza, klienci, num_iterations, d, capacity, D, alpha, beta):
    bests_koszty = np.zeros(shape=len(klienci), dtype=float)
    bests_solutions = []
    for i in range(0,len(klienci)):
        x = simulated_annealing_step(i+1, baza, klienci, T_base, num_iterations, d, capacity, D, alpha, beta)
        if x is not None:
            bests_koszty[i] = x[1]
            bests_solutions.append(x[0])
            print('init_pojazdy: ', i+1)
            print('Koszt: ', x[1])
        else:
            bests_koszty[i] = np.inf
            bests_solutions.append([])
    best = np.argmin(bests_koszty)
    return (bests_solutions[best], bests_koszty[best])

sa(8,baza, klienci, 1000, d, capacity, D, alpha, beta)

def simulated_annealing_step_prezentacja(ust_pojazd, baza, klienci, T, num_iterations, d, capacity, D):
   # wynik = generuj_losowy_wektor(len(klienci), ust_pojazd)
    wynik = len(klienci)
    obecne = stworz_wektor_pojazdy(wynik, ust_pojazd)

    fig, ax = plt.subplots()
    kolory = plt.rcParams['axes.prop_cycle'].by_key()['color'] + \
             ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5']

    # kontrola odleglosci trasy i ładowności wylosowanego początkowego stanu
    # najpierw ladownosc, bo jak ladownosci nie bedzie, to zadne potasowanie miedzy soba nie da rozwiazania
    kontrolny_id = 0
    while not sprawdz_ladownosc(obecne, d, capacity):
        #wynik = generuj_losowy_wektor(len(klienci), ust_pojazd)
        wynik=len(klienci)
        obecne = stworz_wektor_pojazdy(wynik, ust_pojazd)
        kontrolny_id += 1
        if kontrolny_id > 500:
            print('Niemożliwe znalezienie kombinacji spełniającej warunki.')
            return

    obecne = potasuj(obecne, ust_pojazd)
    kontrolny_id = 0
    while not sprawdz_odleglosci(obecne, D):
        obecne = potasuj(obecne, ust_pojazd)
        kontrolny_id += 1
        if kontrolny_id > 500:
            print('Niemożliwe znalezienie kombinacji spełniającej warunki.')
            return

            # na czarno rysujemy klientow, a roznymi kolorami trasy roznych ciezarowek
    ax.plot(klienci[:, 0], klienci[:, 1], c='black', marker='o', linewidth=0)
    ax.plot(baza[0], baza[1], 'r+')
    for i in range(len(obecne)):
        coords = np.concatenate(([baza], klienci[obecne[i] - 1], [baza]))
        ax.plot(coords[:, 0], coords[:, 1], kolory[i])
        ax.set_title(f'Iteracja: 0, koszt: {kara(obecne,baza,klienci,alpha,beta):.4f}, temp: {T:.2f}')
    plt.pause(0.1)
    # -------------------------------------------------------

    kontrolny_id = 0
    flag = False
    it = 1
    for j in range(num_iterations):
        ax.clear()
        ax.plot(klienci[:, 0], klienci[:, 1], c='black', marker='o', linewidth=0)
        ax.plot(baza[0], baza[1], 'r+')

        for i in range(2):
            if not i:
                nowe_propozycja = reorder(obecne)
            else:
                nowe_propozycja = swap(obecne, klienci)
            if sprawdz_ladownosc(nowe_propozycja, d, capacity) and sprawdz_odleglosci(nowe_propozycja, D):
                nowe = nowe_propozycja
                obecny_koszt = kara(obecne, baza, klienci,alpha,  beta)
                nowy_koszt = kara(nowe, baza, klienci,alpha,beta)
                if nowy_koszt < obecny_koszt or (nowy_koszt != obecny_koszt and \
                                                 np.exp(-(nowy_koszt - obecny_koszt) / T) >
                                                 np.random.uniform(0, 1, size=1)[0]):
                    obecne = nowe
                    for p in range(len(obecne)):
                        coords = np.concatenate(([baza], klienci[obecne[p] - 1], [baza]))
                        ax.plot(coords[:, 0], coords[:, 1], kolory[p])
                        ax.set_title(f'Iteracja: {it}, koszt: {nowy_koszt:.4f}, temp: {T:.2f}')
                    it += 1
                    plt.pause(0.15)
            else:
                kontrolny_id += 1
                if kontrolny_id > num_iterations/2:
                    flag = True
                    break


        if flag:
            break
        T = T * 0.97

    plt.show()

    return (obecne, obecny_koszt)

simulated_annealing_step_prezentacja(5, baza, klienci, 1000, 300, d, capacity, D)