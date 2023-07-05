import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split

# Lecture du fichier des données des élections présidentielles de 2022
elec2022 = pd.read_csv("data.csv", sep=';', encoding='ISO-8859-1')
secur2022 = pd.read_csv("data_securite_2022.csv", sep=';', encoding='ISO-8859-1')
data = elec2022.merge(secur2022,on='code_departement')

# Liste des noms des candidats
noms_candidats = ['nom_candidat_1',
                  'nom_candidat_2',
                  'nom_candidat_3',
                  'nom_candidat_4',
                  'nom_candidat_5',
                  'nom_candidat_6',
                  'nom_candidat_7',
                  'nom_candidat_8',
                  'nom_candidat_9',
                  'nom_candidat_10',
                  'nom_candidat_11',
                  'nom_candidat_12']

# Liste des colonnes de voix pour chaque candidat
colonnes_voix = ['voix_1', 'voix_2', 'voix_3', 'voix_4', 'voix_5', 'voix_6', 'voix_7', 'voix_8', 'voix_9', 'voix_10', 'voix_11', 'voix_12']


# Création d'une nouvelle colonne "gagnant"
data['gagnant'] = data[colonnes_voix].idxmax(axis=1)
data['gagnant'] = data['gagnant'].apply(lambda x: noms_candidats[int(x[-1]) - 1])

# Affichage des données mises à jour
# Sélection des indicateurs (X) et de la variable cible (Y)
indicateurs = ['nb_inscrits', 'nb_abstentions', 'nb_votants', 'nb_blancs', 'nb_nuls', 'nb_exprimes', 'taux_pour_1000']
X = data[indicateurs]
colonnes_voix_gagnant = ['voix_1', 'voix_2', 'voix_3', 'voix_4', 'voix_5', 'voix_6', 'voix_7', 'voix_8', 'voix_9', 'voix_10', 'voix_11', 'voix_12']
Y = data[colonnes_voix_gagnant].max(axis=1)
print("Y >>> ", data[colonnes_voix_gagnant])

# Division des données en ensembles d'entraînement et de test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Création et entraînement du modèle de régression linéaire
model = LinearRegression()
model.fit(X_train, Y_train)

# Prédiction sur l'ensemble de test
Y_pred = model.predict(X_test)

print("Prédiction : ", round(Y_pred[0]), )
print('X TEST >>', X_test)

# Exemple de prédiction pour de nouvelles données
nouvelles_donnees = pd.DataFrame([[50000, 5000, 30000, 800, 1000, 15000, 4]], columns=indicateurs)
nouvelle_prediction = model.predict(nouvelles_donnees)
print("Prédiction pour de nouvelles données : {}".format(nouvelle_prediction))

