from operator import attrgetter
import numpy as np
import matplotlib.animation as anim
import pylab
import math
import random


# Informacja dotycząca irysów
# pierwszy element - sepal length
# drugi element - sepal width
# trzeci element - petal length
# czwarty element - petal length
# piąty element - klasa


# wczytwywanie danych z pliku-------------------------------------------------------------------------------------------

def wczytanie():
    try:
        table = []
        with open("Iris.txt", "r") as f:
            lista_linii = [line.rstrip("\n") for line in f]
            for linia in lista_linii:
                for pole in linia.split(","):
                    table.append(pole)

        ir = [[]]
        licznik = 0
        licznik_2 = 0

        while licznik < len(table):
            if licznik % 5 != 4:
                ir[licznik_2].append(float(table[licznik]))
                licznik += 1
            else:
                if licznik + 1 != len(table):
                    ir[licznik_2].append(table[licznik])
                    ir.append([])
                    licznik += 1
                    licznik_2 += 1
                else:
                    ir[licznik_2].append(table[licznik])
                    licznik += 1
                    licznik_2 += 1

        return ir



    except FileNotFoundError:
        print("File not found")

def wczytanie_wina():
    try:
        table = []
        with open("wine.txt", "r") as f:
            lista_linii = [line.rstrip("\n") for line in f]
            for linia in lista_linii:
                for pole in linia.split(","):
                    table.append(pole)

        ir = [[]]
        licznik = 0
        licznik_2 = 0

        while licznik < len(table):
            if licznik % 5 != 4:
                ir[licznik_2].append(float(table[licznik]))
                licznik += 1
            else:
                if licznik + 1 != len(table):
                    ir[licznik_2].append(table[licznik])
                    ir.append([])
                    licznik += 1
                    licznik_2 += 1
                else:
                    ir[licznik_2].append(table[licznik])
                    licznik += 1
                    licznik_2 += 1

        return ir



    except FileNotFoundError:
        print("File not found")

# posegregowanie na trzy tablice----------------------------------------------------------------------------------------

def podziel_na(ir):
    table_setosa = []
    table_versicolor = []
    table_verginica = []

    licznik = 0
    for x in range(len(ir)):
        if ir[licznik][4] == "Iris-setosa":
            table_setosa.append(ir[licznik])
            licznik += 1
        elif ir[licznik][4] == "Iris-versicolor":
            table_versicolor.append(ir[licznik])
            licznik += 1
        else:
            table_verginica.append(ir[licznik])
            licznik += 1

    return table_setosa,table_versicolor,table_verginica

ir = wczytanie()
#setosa,versicolor,vergenica = podziel_na(ir)

#print(setosa, "\n", versicolor, "\n", vergenica)

# Dane statystczne------------------------------------------------------------------------------------------------------

def minimum(table_temp, a):
    min = 1000.0
    i = 0
    while i < len(table_temp):
        if table_temp[i][a] < min:
            min = table_temp[i][a]
        i += 1

    return min

def minimum(table_temp, a):
    min = 1000.0
    i = 0
    while i < len(table_temp):
        if table_temp[i][a] < min:
            min = table_temp[i][a]
        i += 1

    return min



def maximum(table_temp, a):
    max = 0
    i = 0
    while i < len(table_temp):
        if table_temp[i][a] > max:
            max = table_temp[i][a]
        i += 1

    return max


#-----------------------------------------------------------------------------------------------------------------------

def odleglosc_od_punktow(punkt1, punkt2):
    y = math.pow(punkt1[0], 2) + math.pow(punkt1[1], 2)
    z = math.pow(punkt2[0], 2) + math.pow(punkt2[1], 2)
    return round(math.sqrt(math.fabs(y - z)), 2)


def losuj_element(tab, index):
    element = []
    element.append(round(random.uniform(minimum(tab, index), maximum(tab, index)), 2))
    element.append(round(random.uniform(minimum(tab, index+1), maximum(tab, index + 1)), 2))
    return element

def losowanie_tablicy_elementow(tab, index, a): # 'a' oznacza ilość losowanych elementów
    table_temp = []
    i = 0
    while(i < a):
        table_temp.append(losuj_element(tab, index))
        i += 1

    return table_temp

#Szukanie najlepszego pasującego elementu (BMU - best matching unit)

def kolejna_dana(input_table, a, index):
    temp = input_table[index]
    dana = []
    dana.append(temp[a])
    dana.append(temp[a+1])
    return dana

def losowana_dana(input_table, a):
    temp = input_table[random.randrange(len(input_table)-1)]
    dana = []
    dana.append(temp[a])
    dana.append(temp[a+1])
    return dana


def najlepszy_pasujacy_element(unit_table, input_element):
    distance_table = []
    i = 0
    while(i < len(unit_table)):
        distance_table.append(round(odleglosc_od_punktow(input_element, unit_table[i]), 2))
        i += 1

    return distance_table.index(min(distance_table))


def sasiedztwo_bmu(table, iteracja, max_iteracji, a): #'a' oznacza typ danych
    promien_danych = maximum(table, a)/2
    stala_czasu = max_iteracji/math.log1p(promien_danych)
    return promien_danych * math.exp(-(iteracja/stala_czasu))




def adaptacja_gaussa(v, bmu, table, iteracja, max_iteracji, a):
    return math.exp(-(odleglosc_od_punktow(v, bmu))/2*pow(sasiedztwo_bmu(table, iteracja, max_iteracji, a), 2))

def adaptacja_gaussa_2(table, iteracja, max_iteracji, a):
    return math.exp(-(iteracja)/2*pow(sasiedztwo_bmu(table, iteracja, max_iteracji, a), 2))


def zmniejszenie_nauczania(iteracja, max_iteracji, poczatkowe_nauczanie):
    return poczatkowe_nauczanie * math.exp(-(iteracja/max_iteracji))


def przesun_neuron(v, bmu, dana, table, iteracja, max_iteracji, a):
    return np.array(v) + np.round(adaptacja_gaussa(v, bmu, table, iteracja, max_iteracji, a)* zmniejszenie_nauczania(iteracja, max_iteracji, 0.2)*(np.array(dana) - np.array(v)), 2)

def sortuj_po_odleglosci(unit_table, input_element, ages, prototypy):
    distance_table = []
    i = 0
    while (i < len(unit_table)):
        distance_table.append(round(odleglosc_od_punktow(input_element, unit_table[i]), 2))
        i += 1

    i = 0
    zipped = []
    while (i < len(unit_table)):
        tuple = (distance_table[i], unit_table[i], ages[i], prototypy[i])
        zipped.append(tuple)
        i += 1

    zipped.sort(key = lambda x: x[0])

    unit_table = []
    ages = []
    prototypy = []
    for x in zipped:
        unit_table.append(x[1])
        ages.append(x[2])
        prototypy.append((x[3]))


    return unit_table, ages, prototypy




# petal-length x petal-width --------------------------------------------------------------------------------------------

fig = pylab.figure()

def wyswietl(table, BMU, dana, a, prototypy):
    tableX = []
    tableX3 = []
    tableX1 = []

    for x in range(len(ir)):
        tableX.append(ir[x][a])
    for x in range(len(table)):
        tableX3.append(table[x][0])
    for x in range(len(prototypy)):
        tableX1.append(prototypy[x][0])

    tableY = []

    tableY3 = []
    tableY1 = []

    for x in range(len(ir)):
        tableY.append(ir[x][a+1])

    for x in range(len(table)):
        tableY3.append(table[x][1])
    for x in range(len(prototypy)):
        tableY1.append(prototypy[x][1])

    fig.clear()
    pylab.plot(tableX, tableY, 'ro', color='blue')
    pylab.plot([x for x in tableX3], [x for x in tableY3], 'ro', color="black")
    #pylab.plot([x for x in tableX1], [x for x in tableY1], 'ro', color="green")
    pylab.plot(BMU[0], BMU[1], 'ro', color='red')
    pylab.plot(dana[0], dana[1], 'ro', color='yellow')

    pylab.grid(True)
    pylab.pause(0.2)
def wyswielt_blad(a):
    fig.clear()
    x = 0
    blad = []
    while(x+a<len(blad_kwantyzacj)):
        blad.append(blad_kwantyzacj[x])
        x += a
    x = range(len(blad))
    pylab.plot(x, blad, color='blue')
    pylab.show()
def sprawdzanie_epok(units, iteracja_2, BMUindex, ages, table, a):
    if not np.array_equal(units[iteracja_2], units[BMUindex]) and ages[iteracja_2][1] == 0:
        ages[iteracja_2][0] += 1

    if ages[iteracja_2][0] >= 10:
        units[iteracja_2] = losuj_element(table, a)
        ages[iteracja_2][0] = 0
        return 1

    else:
        return 0


def FunkcjaG(units,iteracja_2,BMU,table,iteracja,limit_iteracji,a):
    if odleglosc_od_punktow(units[iteracja_2], BMU) <= sasiedztwo_bmu(table, iteracja, limit_iteracji, a):
        if odleglosc_od_punktow(units[iteracja_2], BMU) >= 0.5 or np.array_equal(units[iteracja_2], BMU):
            return 1

        else:
            return 0
    else:
        return 0

def blad_kwantyzacji(BMU, prototypy):
    distance = []
    i = 0
    while i < len(prototypy):
        distance.append(odleglosc_od_punktow(BMU, prototypy[i]))
        i += 1
    return sum(distance) / len(distance)

def konwersja_na_array(units):
    protypy = []
    i = 0
    while i < len(units):
        protypy.append(np.array(units[i]))
        i += 1

    return protypy

blad_kwantyzacj = []

def kohonen(table, limit_iteracji, a, ilosc_neuronow):
    iteracja = 0
    units = losowanie_tablicy_elementow(table, a, ilosc_neuronow)
    prototypy = konwersja_na_array(units)

    ages = []
    x = 0
    while x < len(units):
        ages.append([0, 0])
        x += 1

    while iteracja < limit_iteracji:
        #dana = kolejna_dana(table, a, iteracja)
        dana = losowana_dana(table, a)
        BMU = units[najlepszy_pasujacy_element(units, dana)]
        BMUindex = najlepszy_pasujacy_element(units, dana)
        ages[BMUindex][1] = 1
        iteracja_2 = 0
        while iteracja_2 < len(units):
            if FunkcjaG(units, iteracja_2, BMU, table, iteracja, limit_iteracji, a):
                units[iteracja_2] = przesun_neuron(units[iteracja_2], BMU, dana, table, iteracja, limit_iteracji, a)

            sprawdzanie_epok(units, iteracja_2, BMUindex, ages, table, a)

            iteracja_2 += 1

        if (iteracja % 20) == 0:
            wyswietl(units, BMU, dana, a, prototypy)

        iteracja += 1
        blad_kwantyzacj.append(blad_kwantyzacji(BMU, prototypy))


    pylab.pause(3.0)
    iteracja_2 = 0
    while iteracja_2 < len(units):
        if ages[iteracja_2][1] == 0:
            del ages[iteracja_2]
            del units[iteracja_2]

        else:
            iteracja_2 += 1

    wyswietl(units, BMU, dana, a, prototypy)
    print(len(units))
    pylab.pause(10.0)
    return units

def gaz_neuronowy(table, limit_iteracji, a, ilosc_neuronow):
    iteracja = 0
    units = losowanie_tablicy_elementow(table, a, ilosc_neuronow)
    prototypy = konwersja_na_array(units)
    print(prototypy)
    ages = []
    x = 0
    while x < len(units):
        ages.append([0, 0])
        x += 1

    dana = losowana_dana(table, a)

    while iteracja < limit_iteracji:
        #dana = kolejna_dana(table, a, iteracja)
        dana = losowana_dana(table, a)
        units, ages, prototypy = sortuj_po_odleglosci(units, dana, ages, prototypy)
        BMU = units[0]
        ages[0][1] = 1

        iteracja_2 = 0
        while iteracja_2 < len(units):
            if FunkcjaG(units, iteracja_2, BMU, table, iteracja, limit_iteracji, a):
                units[iteracja_2] = przesun_neuron(units[iteracja_2], BMU, dana, table, iteracja, limit_iteracji, a)

            sprawdzanie_epok(units, iteracja_2, 0, ages, table, a)

            iteracja_2 += 1

        if (iteracja % 100) == 0:
            wyswietl(units, BMU, dana, a, prototypy)

        blad_kwantyzacj.append(blad_kwantyzacji(BMU, prototypy))

        iteracja += 1

    iteracja_2 = 0

    while iteracja_2 < len(units):
        if ages[iteracja_2][1] == 0:
            del ages[iteracja_2]
            del units[iteracja_2]

        else:
            iteracja_2 += 1

    wyswietl(units, BMU, dana, a, prototypy)
    print(len(units))
    pylab.pause(10.0)

    return units
#kohonen(ir, 1000, 2, 30)
gaz_neuronowy(ir, 2500, 2, 60)