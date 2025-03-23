# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 20:02:08 2024

@author: joris
"""

import numpy as np
import matplotlib.pyplot as plt


#%% Tracer des courbes Q(h/DN) et U(h/DN)


print("")
DN = float(input("DN en mm : "))/1000
Ks = float(input("Ks : "))
I = float(input("Pente en % : "))
print("")
h_eau = float(input("Hauteur d'eau en mm : "))/1000
h_DN = (h_eau/DN)*100
print("On a h/DN =",round(h_DN,2),"%")

DN_table = DN
Ks_table = Ks
I_table = I

# Les valeurs de Ks, DN et I de base
Ks_base = 1
DN_base = 1
I_base = 1

h_valeurs = np.linspace(0.00001, DN_base, 1000)

def Q_base(h):
    return (Ks_base*(1/8)*(2*np.arccos(((DN_base/2)-h)/(DN_base/2))-np.sin(2*np.arccos(((DN_base/2)-h)/(DN_base/2))))*DN_base**2*(DN_base*0.25*(1-((np.sin(2*np.arccos(((DN_base/2)-h)/(DN_base/2))))/(2*np.arccos(((DN_base/2)-h)/(DN_base/2))))))**(2/3)*np.sqrt(I_base/100))/Q_max

def U_base(h):
    return (Ks_base*(DN_base*0.25*(1-((np.sin(2*np.arccos(((DN_base/2)-h)/(DN_base/2))))/(2*np.arccos(((DN_base/2)-h)/(DN_base/2))))))**(2/3)*np.sqrt(I_base/100))/U_max

Q_max = Ks_base*((np.pi*DN_base**2)/4)*(DN_base/4)**(2/3)*np.sqrt(I_base/100)
U_max = Ks_base*(DN_base/4)**(2/3)*np.sqrt(I_base/100)

Q_valeurs = Q_base(h_valeurs)
U_valeurs = U_base(h_valeurs)

# Trouver l'indice de la première valeur où Q dépasse 1 en y
indice_Q_depasse_1 = np.argmax(Q_valeurs > 1)
# Trouver l'indice de la première valeur où U dépasse 1 en y
indice_U_depasse_1 = np.argmax(U_valeurs > 1)

# Valeur de h/DN où Q dépasse 1 en y
h_DN_Q_depasse_1 = h_valeurs[indice_Q_depasse_1]
# Valeur de h/DN où U dépasse 1 en y
h_DN_U_depasse_1 = h_valeurs[indice_U_depasse_1]

# Trouver l'indice du maximum pour Q
indice_max_Q = np.argmax(Q_valeurs)
# Trouver l'indice du maximum pour U
indice_max_U = np.argmax(U_valeurs)

# Valeurs de h/DN au maximum pour Q et U
h_DN_max_Q = h_valeurs[indice_max_Q]
h_DN_max_U = h_valeurs[indice_max_U]

plt.plot(h_valeurs, Q_valeurs, label='Q_h / Qps')
plt.plot(h_valeurs, U_valeurs, label='U_h / Ups')
plt.title('Rapport des débits ou vitesses sur leur pleine section en fonction de h/DN')
plt.xlabel('h / DN')  
plt.ylabel('U_h / Ups  ;  Q_h / Qps')
plt.axvline(x=(h_eau/DN), color='black', linestyle='--')
plt.grid(True)
plt.legend()

print("")
print(f"Q > Qps pour h/DN = {h_DN_Q_depasse_1*100:.2f}% soit h = {int(h_DN_Q_depasse_1*DN*1000)}mm")
print(f"U > Ups pour h/DN = {h_DN_U_depasse_1*100:.2f}% soit h = {int(h_DN_U_depasse_1*DN*1000)}mm")
print("")
print(f"Q = Qmax pour h/DN = {h_DN_max_Q*100:.2f}% soit h = {int(h_DN_max_Q*DN*1000)}mm")
print(f"U = Umax pour h/DN = {h_DN_max_U*100:.2f}% soit h = {int(h_DN_max_U*DN*1000)}mm")

plt.show()

#%% Calcul Q et U pour tous les h ainsi que les maxs

Q_eau = Ks*(1/8)*(2*np.arccos(((DN/2)-h_eau)/(DN/2))-np.sin(2*np.arccos(((DN/2)-h_eau)/(DN/2))))*DN**2*(DN*0.25*(1-((np.sin(2*np.arccos(((DN/2)-h_eau)/(DN/2))))/(2*np.arccos(((DN/2)-h_eau)/(DN/2))))))**(2/3)*np.sqrt(I/100)
U_eau = Ks*(DN*0.25*(1-((np.sin(2*np.arccos(((DN/2)-h_eau)/(DN/2))))/(2*np.arccos(((DN/2)-h_eau)/(DN/2))))))**(2/3)*np.sqrt(I/100)

S_h_eau = (1/8)*(2*np.arccos(((DN/2)-h_eau)/(DN/2))-np.sin(2*np.arccos(((DN/2)-h_eau)/(DN/2))))*DN**2
S_tot = (np.pi*DN**2)/4
rapport_S_Stot = S_h_eau/S_tot

print("")
print("Le débit est de :",round((Q_eau)*1000,2),"L/s")
print("La vitesse est de :",round((U_eau),2),"m/s")
print("La section est rempli à :",round(rapport_S_Stot*100,2),"%")
L_miroir = DN*np.sin(np.arccos(((DN/2)-h_eau)/(DN/2)))
print("Largueur au miroir :",round(L_miroir,2),"m")
L_miroir_DN = L_miroir/DN
print("On a L_miroir/DN =",round(L_miroir_DN*100,2),"%")

def Q(h):
        return Ks*(1/8)*(2*np.arccos(((DN/2)-h)/(DN/2))-np.sin(2*np.arccos(((DN/2)-h)/(DN/2))))*DN**2*(DN*0.25*(1-((np.sin(2*np.arccos(((DN/2)-h)/(DN/2))))/(2*np.arccos(((DN/2)-h)/(DN/2))))))**(2/3)*np.sqrt(I/100)

def U(h):
        return Ks*(DN*0.25*(1-((np.sin(2*np.arccos(((DN/2)-h)/(DN/2))))/(2*np.arccos(((DN/2)-h)/(DN/2))))))**(2/3)*np.sqrt(I/100)

valeurs_x = np.linspace(0.02, DN, 50)
valeurs_Q = Q(valeurs_x)
valeurs_U = U(valeurs_x)

donnees_tableau0 = [['X', 'X/DN', 'Fonction 4', 'Fonction 5'],]

for i in range(50):
    x_tronque = format(valeurs_x[i], '.2f')
    x_DN_tronque = format(100*valeurs_x[i]/DN, '.0f')
    table_valeurs_Q = format(valeurs_Q[i]*1000, '.2f')
    table_valeurs_U = format(valeurs_U[i], '.2f')

    donnees_tableau0.append([x_tronque, x_DN_tronque, table_valeurs_Q, table_valeurs_U])

titres_colonnes_partie01 = ['Hauteur d\'eau ( m )', 'h / DN ( % )', 'Q ( L / s )', 'U ( m / s )']
titres_colonnes_partie02 = ['Hauteur d\'eau ( m )', 'h / DN ( % )', 'Q ( L / s )', 'U ( m / s )']

donnees_tableau_partie01 = [titres_colonnes_partie01] + donnees_tableau0[1:26]
donnees_tableau_partie02 = [titres_colonnes_partie02] + donnees_tableau0[26:]

#%%

h_values = np.linspace(0.001, DN, 1000)

Q_valeurs=Q(h_values)
U_valeurs=U(h_values)

Q_max = np.max(Q(h_values))
U_Q_max = U(h_DN_max_Q*DN)

U_max = np.max(U(h_values))
Q_U_max = Q(h_DN_max_U*DN)

rapport_Q_Q_max = Q_eau/Q_max
rapport_U_U_max = U_eau/U_max

Qps=Q(DN)
Ups=U(DN)
rapport_Q_Qps = Q_eau/Qps
rapport_U_Ups = U_eau/Ups


def Q_Q_max(h):
    return (Ks*(1/8)*(2*np.arccos(((DN/2)-h)/(DN/2))-np.sin(2*np.arccos(((DN/2)-h)/(DN/2))))*DN**2*(DN*0.25*(1-((np.sin(2*np.arccos(((DN/2)-h)/(DN/2))))/(2*np.arccos(((DN/2)-h)/(DN/2))))))**(2/3)*np.sqrt(I/100))/(Q_max)
    
def U_U_max(h):
    return (Ks*(DN*0.25*(1-((np.sin(2*np.arccos(((DN/2)-h)/(DN/2))))/(2*np.arccos(((DN/2)-h)/(DN/2))))))**(2/3)*np.sqrt(I/100))/(U_max)

valeurs_Q_Q_max = Q_Q_max(valeurs_x)
valeurs_U_U_max = U_U_max(valeurs_x)

donnees_tableau_max = [['X', 'X/DN', 'Fonction 1', 'Fonction 2'],]

for i in range(50):
    x_tronque = format(valeurs_x[i], '.2f')
    x_DN_tronque = format(100*valeurs_x[i]/DN, '.0f')
    table_valeurs_Q_Q_max = format(valeurs_Q_Q_max[i], '.3f')
    table_valeurs_U_U_max = format(valeurs_U_U_max[i], '.3f')

    donnees_tableau_max.append([x_tronque, x_DN_tronque ,table_valeurs_Q_Q_max, table_valeurs_U_U_max])


titres_colonnes_partie_max1 = ['Hauteur d\'eau ( m )', 'h / DN ( % )', 'Q / Q_max', 'U / U_max']
titres_colonnes_partie_max2 = ['Hauteur d\'eau ( m )', 'h / DN ( % )', 'Q / Q_max', 'U / U_max']

donnees_tableau_partie_max1 = [titres_colonnes_partie_max1] + donnees_tableau_max[1:26]
donnees_tableau_partie_max2 = [titres_colonnes_partie_max2] + donnees_tableau_max[26:]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.axis('off')
table1 = ax1.table(cellText=donnees_tableau_partie_max1, loc='center', cellLoc='center', colLabels=None)
table1.auto_set_font_size(False)
table1.set_fontsize(8)
table1.scale(1.2, 1.2)
ax2.axis('off')
table2 = ax2.table(cellText=donnees_tableau_partie_max2, loc='center', cellLoc='center', colLabels=None)
table2.auto_set_font_size(False)
table2.set_fontsize(8)
table2.scale(1.2, 1.2)
plt.show()

def Q_Q_PS(h):
    return (Ks*(1/8)*(2*np.arccos(((DN/2)-h)/(DN/2))-np.sin(2*np.arccos(((DN/2)-h)/(DN/2))))*DN**2*(DN*0.25*(1-((np.sin(2*np.arccos(((DN/2)-h)/(DN/2))))/(2*np.arccos(((DN/2)-h)/(DN/2))))))**(2/3)*np.sqrt(I/100))/(Qps)
    
def U_U_PS(h):
    return (Ks*(DN*0.25*(1-((np.sin(2*np.arccos(((DN/2)-h)/(DN/2))))/(2*np.arccos(((DN/2)-h)/(DN/2))))))**(2/3)*np.sqrt(I/100))/(Ups)

valeurs_Q_Q_PS = Q_Q_PS(valeurs_x)
valeurs_U_U_PS = U_U_PS(valeurs_x)

donnees_tableau_PS = [['X', 'X/DN', 'Fonction 1', 'Fonction 2'],]

for i in range(50):
    x_tronque = format(valeurs_x[i], '.2f')
    x_DN_tronque = format(100*valeurs_x[i]/DN, '.0f')
    table_valeurs_Q_Q_PS = format(valeurs_Q_Q_PS[i], '.3f')
    table_valeurs_U_U_PS = format(valeurs_U_U_PS[i], '.3f')

    donnees_tableau_PS.append([x_tronque, x_DN_tronque, table_valeurs_Q_Q_PS, table_valeurs_U_U_PS])


titres_colonnes_partie_PS1 = ['Hauteur d\'eau ( m )', 'h / DN ( % )', 'Q / Q_PS', 'U / U_PS']
titres_colonnes_partie_PS2 = ['Hauteur d\'eau ( m )', 'h / DN ( % )', 'Q / Q_PS', 'U / U_PS']

donnees_tableau_partie_PS1 = [titres_colonnes_partie_PS1] + donnees_tableau_PS[1:26]
donnees_tableau_partie_PS2 = [titres_colonnes_partie_PS2] + donnees_tableau_PS[26:]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.axis('off')
table1 = ax1.table(cellText=donnees_tableau_partie_PS1, loc='center', cellLoc='center', colLabels=None)
table1.auto_set_font_size(False)
table1.set_fontsize(8)
table1.scale(1.2, 1.2)
ax2.axis('off')
table2 = ax2.table(cellText=donnees_tableau_partie_PS2, loc='center', cellLoc='center', colLabels=None)
table2.auto_set_font_size(False)
table2.set_fontsize(8)
table2.scale(1.2, 1.2)
plt.show()


def H_spé(h):
    V_H_spé = Ks*(DN*0.25*(1-((np.sin(2*np.arccos(((DN/2)-h)/(DN/2))))/(2*np.arccos(((DN/2)-h)/(DN/2))))))**(2/3)*np.sqrt(I/100)
    return h + (V_H_spé**2 / (2 * 9.81))

def H_vitesse(h):
    V_H_spé = Ks*(DN*0.25*(1-((np.sin(2*np.arccos(((DN/2)-h)/(DN/2))))/(2*np.arccos(((DN/2)-h)/(DN/2))))))**(2/3)*np.sqrt(I/100)
    return (V_H_spé**2 / (2 * 9.81))
    
def y(h):
    return h

H_spé_valeurs = H_spé(h_values)
y_valeurs = y(h_values)
H_vitesse_valeurs = H_vitesse(h_values)

H_spé_h_eau = H_spé(h_eau)
H_spé_vitesse = ((U_eau**2)/(2*9.81))/H_spé(h_eau)
H_spé_hauteur = h_eau/H_spé(h_eau)
H_spé_PS = DN + (Ups**2 / (2 * 9.81))

# Trouver la valeur maximale de H_spé et le h correspondant
max_H_spé = np.max(H_spé_valeurs)
index_max_H_spé = np.argmax(H_spé_valeurs)
h_eau_H_spé_max = h_values[index_max_H_spé]

print("")
print("La charge spécifique totale est de :", round(H_spé_h_eau,2),"m")
print("La charge en hauteur est de :",h_eau,"m soit",round(H_spé_hauteur*100,2),"%")
print("La charge en vitesse est de :",round(((U_eau**2)/(2*9.81)),2),"m soit",round(H_spé_vitesse*100,2),"%")
print("")
print("H* pleinne section est de :",round(H_spé_PS,2),"m")
print("H* max est de :", round(max_H_spé,2),"m pour h =",round(h_eau_H_spé_max*1000),"mm soit h/DN =",round(100*h_eau_H_spé_max/DN,2),"%")

plt.plot(h_values*1000, H_spé_valeurs, label='H*(h)')
plt.plot(h_values*1000, y_valeurs, label='H* = h')
plt.plot(h_values*1000, H_vitesse_valeurs, label='H* = V(h)^2/2g')
plt.axhline(y=(H_spé_h_eau), color='black', linestyle='--')
plt.axvline(x=(h_eau*1000), color='black', linestyle='--')
plt.legend()
plt.xlim(0, DN*1000)
plt.ylim(0, max_H_spé*1.2)  
plt.xlabel('Hauteur d\'eau (mm)')
plt.ylabel('H* (m)')
plt.suptitle('Charge spécifique en fonction de la hauteur d\'eau')
plt.title(f'DN = {round(DN*1000)} mm    Ks = {Ks}    I = {I} %')
plt.grid(True)
plt.show()

print("")
print("Le débit max est de :", round(Q_max * 1000, 2), "L/s")
print("La vitesse pour le débit max est de :", round(U_Q_max,2), "m/s")
print("")
print("La vitesse max est de :", round(U_max, 2), "m/s")
print("Le débit pour la vitesse max est de :", round(Q_U_max*1000, 2), "L/s")
print("")
print("Le débit PS est de :", round(Qps * 1000, 2), "L/s")
print("La vitesse PS est de :", round(Ups, 2), "m/s")
print("")
print("Le rapport Q/Qmax est de :", round(rapport_Q_Q_max*100,2),"%")
print("Le rapport U/Umax est de :", round(rapport_U_U_max*100,2),"%")
print("")
print("Le rapport Q/Qps est de :", round(rapport_Q_Qps*100,2),"%")
print("Le rapport U/Ups est de :", round(rapport_U_Ups*100,2),"%")


Fr = Q_eau/np.sqrt(9.81*h_eau**4*DN)
hc = np.sqrt(Q_eau/np.sqrt(9.81*DN))

print("")
print("La hauteur critique est de :",round(hc*1000),"mm")
print("La hauteur normale est de :",round(h_eau*1000),"mm")

print("")
if Fr < 0.90:
    print("Le nombre de Froude est de :",round(Fr,2))
    print("Le régime est FLUVIAL")
elif 0.9 <= Fr <= 1.1:
    print("Le nombre de Froude est de :",round(Fr,2))
    print("Le régime est CRITIQUE")
else:
    print("Le nombre de Froude est de :",round(Fr,2))
    print("Le régime est TORRENTIEL")

plt.plot(h_values*1000,Q_valeurs)
plt.title(f'DN = {round(DN*1000)} mm    Ks = {Ks}    I = {I} %')
plt.ylabel('Débits en m^3/s')  
plt.xlabel('Hauteur d\'eau (mm)')
plt.axhline(y=(Q_eau), color='red', linestyle='--')
plt.axvline(x=(h_eau*1000), color='red', linestyle='--')
plt.grid(True)
plt.show()

plt.plot(h_values*1000,U_valeurs)
plt.title(f'DN = {round(DN*1000)} mm    Ks = {Ks}    I = {I} %')
plt.ylabel('Vitesse en m/s')  
plt.xlabel('Hauteur d\'eau (mm)')
plt.axhline(y=(U_eau), color='red', linestyle='--')
plt.axvline(x=(h_eau*1000), color='red', linestyle='--')
plt.grid(True)
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.axis('off')
table1 = ax1.table(cellText=donnees_tableau_partie01, loc='center', cellLoc='center', colLabels=None)
table1.auto_set_font_size(False)
table1.set_fontsize(8)
table1.scale(1.2, 1.2)
ax2.axis('off')
table2 = ax2.table(cellText=donnees_tableau_partie02, loc='center', cellLoc='center', colLabels=None)
table2.auto_set_font_size(False)
table2.set_fontsize(8)
table2.scale(1.2, 1.2)
plt.show()


def generer_cercle_centre(rayon, centre_y):
    theta = np.linspace(0, 2*np.pi, 500)
    x = rayon * np.cos(theta)
    y = rayon * np.sin(theta) + centre_y
    return x, y

# Définir le rayon du cercle et le centre en y
rayon_cercle = DN/2
centre_y = DN/2

# Générer les coordonnées du cercle centré en (0, R)
x_cercle, y_cercle = generer_cercle_centre(rayon_cercle, centre_y)

# Tracer le cercle
plt.plot(x_cercle*1000, y_cercle*1000, color='black')
plt.title(f'DN = {round(DN*1000)} mm    Ks = {Ks}    I = {I} %    Q = {round(Q_eau*1000,2)} L/s')
plt.xlabel('Longueur en mm')
plt.ylabel('Hauteur d\'eau (mm)')
plt.axhline(y=(hc*1000), color='red', linestyle='--', label='Hauteur critique')
plt.axhline(y=(h_eau*1000), color='b', linestyle='--', label='Hauteur normale/d\'eau')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

def fonction_hc(h):
    Q_eau = Ks*(1/8)*(2*np.arccos(((DN/2)-h)/(DN/2))-np.sin(2*np.arccos(((DN/2)-h)/(DN/2))))*DN**2*(DN*0.25*(1-((np.sin(2*np.arccos(((DN/2)-h)/(DN/2))))/(2*np.arccos(((DN/2)-h)/(DN/2))))))**(2/3)*np.sqrt(I/100)
    return np.sqrt(Q_eau/np.sqrt(9.81*DN))

hc_valeurs = fonction_hc(h_values)

plt.plot(h_values*1000, hc_valeurs*1000, label=' Courbe f(h) = hc')
plt.plot(h_values*1000, y_valeurs*1000, color='black', label='Droite h = hn')
plt.suptitle("Hauteur critique en fonction de la hauteur d'eau")
plt.title(f'DN = {round(DN*1000)} mm    Ks = {Ks}    I = {I} %')
plt.xlabel('Hauteur d\'eau (mm)')
plt.ylabel('Hauteur critique (mm)')
plt.axhline(y=(hc*1000), color='red', linestyle='--')
plt.axvline(x=(h_eau*1000), color='red', linestyle='--')
plt.legend()
plt.grid(True)
plt.show()

def Fr_fonction(h):
    return Q(h)/np.sqrt(9.81*h**4*DN)

# Supposons que vous avez un tableau de 1000 points
nombre_de_points = 1000
points = h_values

# Initialisation des variables pour suivre les passages
passage_de_09999_a_10001 = False
passage_de_10001_a_09999 = False

for i in range(1, nombre_de_points):
    valeur_precedente = Fr_fonction(points[i-1])
    valeur_actuelle = Fr_fonction(points[i])

    # Vérifier le passage de 0.9999 à 1.0001
    if valeur_precedente < 1.0001 <= valeur_actuelle:
        passage_de_09999_a_10001 = True
        valeur_passage_1 = points[i]

    # Vérifier le passage de 1.0001 à 0.9999
    elif valeur_precedente > 0.9999 >= valeur_actuelle:
        passage_de_10001_a_09999 = True
        valeur_passage_2 = points[i]

print("")
if passage_de_09999_a_10001 and passage_de_10001_a_09999:
    print("Passage fluvial/torrentiel à h =", round(valeur_passage_1 * 1000), "mm")
    print("Passage torrentiel/fluvial à h =", round(valeur_passage_2 * 1000), "mm")
elif passage_de_09999_a_10001:
    print("Passage fluvial/torrentiel à h =", round(valeur_passage_1 * 1000), "mm")
elif passage_de_10001_a_09999:
    print("Passage torrentiel/fluvial à h =", round(valeur_passage_2 * 1000), "mm")
else:
    print("Aucune transition fluviale/torrentielle")

print("")
# Obtenir Fr_max et la valeur de h associée
Fr_max, h_max = np.max(Fr_fonction(h_values)), h_values[np.argmax(Fr_fonction(h_values))]
    
print(f"Le nombre de Froude max est de {Fr_max:.2f} pour h = {round(h_max*1000)} mm")
if 0.9 < Fr_max < 1.1:
    print ("Risque de transition en régime critique")

Fr_values = Fr_fonction(h_values)
plt.plot(h_values*1000, Fr_values, label='f(h) = Fr')
plt.suptitle("Nombre de Froude en fonction de la hauteur d'eau")
plt.title(f'DN = {round(DN*1000)} mm    Ks = {Ks}    I = {I} %')
plt.xlabel('Hauteur d\'eau (mm)')
plt.ylabel("Nombre de Froude")
plt.axhline(y=(1), color='black', label='Passage critique')
plt.axhline(y=(Fr), color='red', linestyle='--')
plt.axvline(x=(h_eau*1000), color='red', linestyle='--')
plt.legend()
plt.grid(True)
plt.show()

#%% Calcul débit et vitesse selon différentes pentes

pentes = [round(I-0.2,1), round(I-0.1,1), I, round(I+0.1,1), round(I+0.2,1)]  # Liste des pentes en %


def Q(h, I):
    if h == 0:
        return 0
    else:
        return Ks * (1/8) * (2 * np.arccos(((DN/2)-h)/(DN/2)) - np.sin(2 * np.arccos(((DN/2)-h)/(DN/2)))) * DN**2 * (DN * 0.25 * (1 - ((np.sin(2 * np.arccos(((DN/2)-h)/(DN/2)))) / (2 * np.arccos(((DN/2)-h)/(DN/2))))))**(2/3) * np.sqrt(I/100)

plt.figure()

for pente in pentes:
    I = pente
    Q_values = [Q(h, I) for h in h_values]
    plt.plot(h_values*1000, Q_values, label=f'Pente = {pente}%')

plt.title(f'DN = {round(DN*1000)} mm    Ks = {Ks}')
plt.ylabel('Débits en m^3/s')
plt.xlabel('Hauteur d\'eau (mm)')
plt.legend()
plt.grid(True)
plt.show()

def U(h, I):
    if h == 0:
        return 0
    else:
        return Ks * (DN * 0.25 * (1 - ((np.sin(2 * np.arccos(((DN/2)-h)/(DN/2)))) / (2 * np.arccos(((DN/2)-h)/(DN/2))))))**(2/3) * np.sqrt(I/100)

plt.figure()

for pente in pentes:
    I = pente
    U_values = [U(h, I) for h in h_values]
    plt.plot(h_values*1000, U_values, label=f'Pente = {pente}%')

plt.title(f'DN = {round(DN*1000)} mm    Ks = {Ks}')
plt.ylabel('Vitesse en m/s')
plt.xlabel('Hauteur d\'eau (mm)')
plt.legend()
plt.grid(True)
plt.show()

#%% Calcul débit et vitesse selon différents Ks

Ks_values = [round(Ks-20, 1), round(Ks-10, 1), round(Ks, 1), round(Ks+10, 1), round(Ks+20, 1)]
I = pentes[2]

def Q(h, Ks):
    if h == 0:
        return 0
    else:
        return Ks * (1/8) * (2 * np.arccos(((DN/2)-h)/(DN/2)) - np.sin(2 * np.arccos(((DN/2)-h)/(DN/2)))) * DN**2 * (DN * 0.25 * (1 - ((np.sin(2 * np.arccos(((DN/2)-h)/(DN/2)))) / (2 * np.arccos(((DN/2)-h)/(DN/2))))))**(2/3) * np.sqrt(I/100)

plt.figure()

for Ks in Ks_values:
    Q_values = [Q(h, Ks) for h in h_values]
    plt.plot(h_values*1000, Q_values, label=f'Ks = {Ks}')

plt.title(f'DN = {round(DN*1000)} mm    Pente = {I}%')
plt.ylabel('Débits en m^3/s')
plt.xlabel('Hauteur d\'eau (mm)')
plt.legend()
plt.grid(True)
plt.show()

def U(h, Ks):
    if h == 0:
        return 0
    else:
        return Ks * (DN * 0.25 * (1 - ((np.sin(2 * np.arccos(((DN/2)-h)/(DN/2)))) / (2 * np.arccos(((DN/2)-h)/(DN/2))))))**(2/3) * np.sqrt(I/100)

plt.figure()

for Ks in Ks_values:
    U_values = [U(h, Ks) for h in h_values]
    plt.plot(h_values*1000, U_values, label=f'Ks = {Ks}')

plt.title(f'DN = {round(DN*1000)} mm    Pente = {I}%')
plt.ylabel('Vitesse en m/s')
plt.xlabel('Hauteur d\'eau (mm)')
plt.legend()
plt.grid(True)
plt.show()

#%% Calcul débit et vitesse selon différents DN

Ks = Ks_values[2]
I = pentes[2]
DN_values = [round(DN-0.1, 3), round(DN-0.05, 3), round(DN, 3), round(DN+0.05, 3), round(DN+0.1, 3)]

def Q(h, DN):
    if h == 0:
        return 0
    else:
        return Ks * (1/8) * (2 * np.arccos(((DN/2)-h)/(DN/2)) - np.sin(2 * np.arccos(((DN/2)-h)/(DN/2)))) * DN**2 * (DN * 0.25 * (1 - ((np.sin(2 * np.arccos(((DN/2)-h)/(DN/2)))) / (2 * np.arccos(((DN/2)-h)/(DN/2))))))**(2/3) * np.sqrt(I/100)

plt.figure()

for DN in DN_values:
    # Génération de la liste h_values en fonction de DN
    h_values_DN = np.linspace(0.00001, DN, 1000)

    Q_values = [Q(h, DN) for h in h_values_DN]
    plt.plot(h_values_DN*1000, Q_values, label=f'DN = {round(DN*1000)} mm')

plt.title(f'Ks = {Ks}    Pente = {I}%')
plt.ylabel('Débits en m^3/s')
plt.xlabel('Hauteur d\'eau (mm)')
plt.legend()
plt.grid(True)
plt.show()

def U(h, DN):
    if h == 0:
        return 0
    else:
        return Ks * (DN * 0.25 * (1 - ((np.sin(2 * np.arccos(((DN/2)-h)/(DN/2)))) / (2 * np.arccos(((DN/2)-h)/(DN/2))))))**(2/3) * np.sqrt(I/100)

plt.figure()

for DN in DN_values:
    # Génération de la liste h_values en fonction de DN
    h_values_DN = np.linspace(0.00001, DN, 1000)

    U_values = [U(h, DN) for h in h_values_DN]
    plt.plot(h_values_DN*1000, U_values, label=f'DN = {round(DN*1000)} mm')

plt.title(f'Ks = {Ks}    Pente = {I}%')
plt.ylabel('Vitesse en m/s')
plt.xlabel('Hauteur d\'eau (mm)')
plt.legend()
plt.grid(True)
plt.show()

#%%

def fonction_1(x):
    return 2*np.arccos(((DN_base/2)-x)/(DN_base/2))-np.sin(2*np.arccos(((DN_base/2)-x)/(DN_base/2)))

def fonction_2(x):
    return ((1/8)*(2*np.arccos(((DN_base/2)-x)/(DN_base/2))-np.sin(2*np.arccos(((DN_base/2)-x)/(DN_base/2))))*DN_base**2)/(np.pi*0.25)

def fonction_3(x):
    return (DN_base*0.25*(1-((np.sin(2*np.arccos(((DN_base/2)-x)/(DN_base/2))))/(2*np.arccos(((DN_base/2)-x)/(DN_base/2))))))/(0.25)

def fonction_B(x):
    return 2*np.sqrt(x*(DN_base-x))

valeurs_x = np.linspace(0.02, 1, 50)
valeurs_y1 = fonction_1(valeurs_x)
valeurs_y2 = fonction_2(valeurs_x)
valeurs_y3 = fonction_3(valeurs_x)
valeurs_B = fonction_B(valeurs_x)

donnees_tableau = [['X', 'Fonction 1', 'Fonction 2', 'Fonction 3'],]

for i in range(50):
    x_tronque = format(valeurs_x[i], '.2f')
    y1_tronque = format(valeurs_y1[i], '.3f')
    y2_tronque = format(valeurs_y2[i], '.3f')
    y3_tronque = format(valeurs_y3[i], '.3f')
    B_tronque = format(valeurs_B[i], '.3f')
    
    donnees_tableau.append([x_tronque, y1_tronque, y2_tronque, y3_tronque, B_tronque])

titres_colonnes_partie1 = ['h / DN', 'Angle θ', 'S / S_ps', 'Rh / Rh_ps', 'B / B_DN/2']
titres_colonnes_partie2 = ['h / DN', 'Angle θ', 'S / S_ps', 'Rh / Rh_ps', 'B / B_DN/2']

donnees_tableau_partie1 = [titres_colonnes_partie1] + donnees_tableau[1:26]
donnees_tableau_partie2 = [titres_colonnes_partie2] + donnees_tableau[26:]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.axis('off')
table1 = ax1.table(cellText=donnees_tableau_partie1, loc='center', cellLoc='center', colLabels=None)
table1.auto_set_font_size(False)
table1.set_fontsize(8)
table1.scale(1.2, 1.2)
ax2.axis('off')
table2 = ax2.table(cellText=donnees_tableau_partie2, loc='center', cellLoc='center', colLabels=None)
table2.auto_set_font_size(False)
table2.set_fontsize(8)
table2.scale(1.2, 1.2)
plt.show()

#%%

DN = DN_table

def Fr_fonction_table(x):
    return (Ks*(1/8)*(2*np.arccos(((DN/2)-x)/(DN/2))-np.sin(2*np.arccos(((DN/2)-x)/(DN/2))))*DN**2*(DN*0.25*(1-((np.sin(2*np.arccos(((DN/2)-x)/(DN/2))))/(2*np.arccos(((DN/2)-x)/(DN/2))))))**(2/3)*np.sqrt(I/100))/np.sqrt(9.81*x**4*DN)

valeurs_x = np.linspace(0.02, DN, 50)
valeurs_y6 = fonction_hc(valeurs_x)
valeurs_y7 = Fr_fonction_table(valeurs_x)

donnees_tableau3 = [['X', 'Fonction 6', 'Fonction 7'],]

for i in range(50):
    x_tronque = format(valeurs_x[i], '.2f')
    y6_tronque = format(valeurs_y6[i], '.2f')
    y7_tronque = format(valeurs_y7[i], '.2f')

    donnees_tableau3.append([x_tronque, y6_tronque, y7_tronque])


titres_colonnes_partie31 = ['Hauteur d\'eau (m)', 'Hc (m)', 'Froude']
titres_colonnes_partie32 = ['Hauteur d\'eau (m)', 'Hc (m)', 'Froude']

donnees_tableau_partie31 = [titres_colonnes_partie31] + donnees_tableau3[1:26]
donnees_tableau_partie32 = [titres_colonnes_partie32] + donnees_tableau3[26:]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.axis('off')
table1 = ax1.table(cellText=donnees_tableau_partie31, loc='center', cellLoc='center', colLabels=None)
table1.auto_set_font_size(False)
table1.set_fontsize(8)
table1.scale(1.2, 1.2)
ax2.axis('off')
table2 = ax2.table(cellText=donnees_tableau_partie32, loc='center', cellLoc='center', colLabels=None)
table2.auto_set_font_size(False)
table2.set_fontsize(8)
table2.scale(1.2, 1.2)
plt.show()

#%%

DN = DN_table
Ks = Ks_table
I = I_table

def Hspé_table(x):
    V_H_spé_table = Ks*(DN*0.25*(1-((np.sin(2*np.arccos(((DN/2)-x)/(DN/2))))/(2*np.arccos(((DN/2)-x)/(DN/2))))))**(2/3)*np.sqrt(I/100)
    return x + (V_H_spé_table**2 / (2 * 9.81))

def Hspé_hauteur_table(x):
    return (x / Hspé_table(x))*100

def Hspé_vitesse_table(x):
    V_H_spé_table = Ks*(DN*0.25*(1-((np.sin(2*np.arccos(((DN/2)-x)/(DN/2))))/(2*np.arccos(((DN/2)-x)/(DN/2))))))**(2/3)*np.sqrt(I/100)
    return (((V_H_spé_table**2)/(2*9.81))/ Hspé_table(x))*100
    
Hspé_table_valeurs = Hspé_table(valeurs_x)
Hspé_hauteur_table_valeurs = Hspé_hauteur_table(valeurs_x)
Hspé_vitesse_table_valeurs = Hspé_vitesse_table(valeurs_x)

donnees_tableau4 = [['X', 'Hspé_hauteur_table', 'Hspé_vitesse_table', 'Hspé_table'],]

for i in range(50):
    x_tronque = format(valeurs_x[i], '.2f')
    y8_tronque = format(Hspé_hauteur_table_valeurs[i], '.2f')
    y9_tronque = format(Hspé_vitesse_table_valeurs[i], '.2f')
    y10_tronque = format(Hspé_table_valeurs[i], '.2f')

    donnees_tableau4.append([x_tronque, y8_tronque, y9_tronque, y10_tronque])

titres_colonnes_partie41 = ['Hauteur d\'eau (m)', 'h / H* (%)', 'V^2/2g / H* (%)', 'H* (m)']
titres_colonnes_partie42 = ['Hauteur d\'eau (m)', 'h / H* (%)', 'V^2/2g / H* (%)', 'H* (m)']

donnees_tableau_partie41 = [titres_colonnes_partie41] + donnees_tableau4[1:26]
donnees_tableau_partie42 = [titres_colonnes_partie42] + donnees_tableau4[26:]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.axis('off')
table1 = ax1.table(cellText=donnees_tableau_partie41, loc='center', cellLoc='center', colLabels=None)
table1.auto_set_font_size(False)
table1.set_fontsize(8)
table1.scale(1.2, 1.2)
ax2.axis('off')
table2 = ax2.table(cellText=donnees_tableau_partie42, loc='center', cellLoc='center', colLabels=None)
table2.auto_set_font_size(False)
table2.set_fontsize(8)
table2.scale(1.2, 1.2)
plt.show()

