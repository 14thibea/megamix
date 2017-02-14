import csv
import matplotlib

#Lecture du fichier

fichier = open("D:/Mines/Cours/Stages/Stage_ENS/Problem/EMGaussienne.data",'r')
reader = csv.reader(fichier,delimiter=" ")
x=[]
y=[]
for line in reader:
    a,b = line;
    x.append(float(a));
    y.append(float(b));
fichier.close()

#Affichage du graphique

matplotlib.pyplot.scatter(x,y);