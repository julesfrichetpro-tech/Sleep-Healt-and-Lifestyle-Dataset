# Importation des bibliothèques nécessaires
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from scipy.stats import kruskal

# Message de bienvenue
print("\n")
print("="*80)
print("="*80)
print("""

       ███████╗████████╗ █████╗ ██████╗ ████████╗██╗███╗   ██╗ ██████╗ 
       ██╔════╝╚══██╔══╝██╔══██╗██╔══██╗╚══██╔══╝██║████╗  ██║██╔════╝ 
       ███████╗   ██║   ███████║██████╔╝   ██║   ██║██╔██╗ ██║██║  ███╗
       ╚════██║   ██║   ██╔══██║██╔══██╗   ██║   ██║██║╚██╗██║██║   ██║
       ███████║   ██║   ██║  ██║██║  ██║   ██║   ██║██║ ╚████║╚██████╔╝
       ╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝╚═╝  ╚═══╝ ╚═════╝ 

                         É T U D E   D E   L A   S A N T É  
                               D U   S O M M E I L

""")
print("="*80)
print("="*80)
print("\n")
print("Ce script réalise une analyse complète des données de santé du sommeil.")
print("Il génère des visualisations, effectue des tests statistiques et construit")
print("un modèle d'arbre de décision pour prédire les troubles du sommeil.")
print("\nÉtapes de l'analyse :")
print("  1. Exploration des données (statistiques descriptives)")
print("  2. Visualisations (histogrammes, boxplots, heatmap)")
print("  3. Tests statistiques (Kruskal-Wallis)")
print("  4. Modélisation (arbre de décision)")
print("  5. Optimisation et évaluation du modèle")
print("\nTous les graphiques seront sauvegardés automatiquement.")
print("\n")
print("="*80)
print("="*80 + "\n")

# Chargement des données
Sleep_healt = pd.read_csv("Sleep_healt.csv")

# Affichage des informations générales sur le DataFrame
print("=========== Informations sur le DataFrame ===========")
Sleep_healt.info()
print("\n")

# Affichage des statistiques descriptives du DataFrame
print("================= Statistiques descriptives =================")
print(Sleep_healt.describe())
print("\n")

# Affichage du nombre de valeurs uniques pour chaque variable
print("== Nombre de valeurs uniques ==")
print(Sleep_healt.nunique())
print("\n")

# Affichage du nombre de valeurs manquantes pour chaque variable
print("==== Valeurs manquantes ====")
print(Sleep_healt.isnull().sum())
print("\n")

# Fonction pour sauvegarder les graphiques
plot_counter = 0
def save_plot_with_title():
    global plot_counter
    plot_counter += 1
    title = plt.gca().get_title()
    fichier_sauver = "Graphiques"
    compteur_title = f"{plot_counter}.{title}"
    filename = compteur_title.replace(" ", "_").replace("'", "").replace('"', "") + ".png"
    os.makedirs(fichier_sauver, exist_ok=True)
    full_path = os.path.join(fichier_sauver, filename)
    plt.savefig(full_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved as {full_path}")

# Histogramme de la variable de Sexe
Sleep_healt["Gender"].value_counts().plot(kind="bar")
plt.title("Distribution par sexe")
plt.xlabel("Sexe")
plt.ylabel("Nombres de patients")
plt.xticks(rotation=0)
save_plot_with_title()
plt.show()


# Histogramme de la variable 'Age'
Sleep_healt["Age"].hist()
plt.title("Distribution de l\'âge")
plt.xlabel('Âge')
plt.ylabel("Nombres de patients")
save_plot_with_title()
plt.show()

bins = [25, 30, 35, 40, 45, 50, 55, 60]  # bornes des intervalles
labels = ["25-30", "30-35", "35-40", "40-45", "45-50", "50-55", "55-60"]  # étiquettes correspondantes

# Création des classes d'âge dans une nouvelle colonne
Sleep_healt["Tranche_Age"] = pd.cut(Sleep_healt["Age"], bins=bins, labels=labels, right=False)

# Comptage du nombre de patients par tranche d'âge
counts = Sleep_healt["Tranche_Age"].value_counts().sort_index()

# Tracé du graphique en barres
plt.bar(counts.index, counts.values)

plt.xlabel("Tranche d'âge")
plt.ylabel("Nombres de patients")
plt.title("Distribution de l'âge par tranches")
plt.xticks(rotation=0)
save_plot_with_title()
plt.show()

# Histogramme de la variable 'Occupation'
Sleep_healt["Occupation"].value_counts().plot(kind="bar")
plt.title("Distribution des occupations")
plt.xlabel("Occupations")
plt.ylabel("Nombres de patients")
plt.tight_layout()
plt.xticks(rotation=45)
save_plot_with_title()
plt.show()

# Histogramme précis de la variable 'Sleep Duration'
Sleep_healt["Sleep Duration"].value_counts().sort_index().plot(kind="bar")
plt.title("Distribution du temps de sommeil")
plt.xlabel("Temps de sommeil")
plt.ylabel("Nombres de patients")
plt.xticks(rotation=0)
plt.yticks(rotation=0)
save_plot_with_title()
plt.show()

# Définition des intervalles de durée de sommeil
bins = [5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9]
labels = ["5h30-6h", "6h-6h30", "6h30-7h", "7h-7h30", "7h30-8h","8h-8h30", "8h30-9h"]

# Création des classes de durée de sommeil
Sleep_healt["Tranche_Sommeil"] = pd.cut(Sleep_healt["Sleep Duration"], bins=bins, labels=labels, right=False)

# Comptage du nombre de patients par tranche de sommeil
counts = Sleep_healt["Tranche_Sommeil"].value_counts().sort_index()

# Tracé du graphique en barres
plt.figure(figsize=(8, 5))
plt.bar(counts.index, counts.values)

# Personnalisation
plt.title("Distribution du temps de sommeil par tranches horaires")
plt.xlabel("Durée du sommeil (heures)")
plt.ylabel("Nombres de patients")
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()
save_plot_with_title()
plt.show()

# Histogramme de la variable 'Sleep Duration'
Sleep_healt["Sleep Duration"].hist()
plt.title("Distribution du temps de sommeil")
plt.xlabel("Temps de sommeil")
plt.ylabel("Nombres de patients")
plt.xticks(rotation=0)
plt.yticks(rotation=0)
save_plot_with_title()
plt.show()

# Histogramme de la variable 'Quality of Sleep'
Sleep_healt["Quality of Sleep"].value_counts().sort_index().plot(kind="bar")
plt.title("Distribution de la qualitée de sommeil")
plt.xlabel("Echelle de la qualitée du sommeil")
plt.ylabel("Nombres de patients")
plt.xticks(rotation=0)
plt.yticks(rotation=0)
save_plot_with_title()
plt.show()

# Histogramme de la variable 'Physical Activity Level' ## A revoir Par tranche ##
Sleep_healt["Physical Activity Level"].hist()
plt.title("Distribution d\'activité physique")
plt.xlabel("Niveau d\'activité physique")
plt.ylabel("Nombres de patients")
save_plot_with_title()
plt.show()

# Histogramme de la variable 'Stress Level'
# Comptage des catégories
counts = Sleep_healt["Stress Level"].value_counts()

# Tracé du graphique en barres
plt.bar(counts.index, counts.values)
plt.title("Distribution du niveau de stress")
plt.xlabel("Niveau du stress")
plt.ylabel("Nombres de patients")
plt.xticks(rotation=0)
save_plot_with_title()
plt.show()


# Histogramme de la variable 'BMI Category'
Sleep_healt["BMI Category"].value_counts().plot(kind="bar")
plt.title("Distribution en fonction des catégories de l\'IMC")
plt.xlabel("Categorie d\'IMC")
plt.ylabel("Nombres de patients")
plt.xticks(rotation=0)
save_plot_with_title()
plt.show()

# Fonction pour catégoriser la pression sanguine (hypotension + hypertension)
# Selon le site : https://icm46.fr/comment-interpreter-resultats-pression-arterielle/
def categorize_blood_pressure(bp_string):
    try:
        # Séparer systolique et diastolique
        systolic, diastolic = map(int, bp_string.split("/"))

        # Catégorisation Hypertension
        if systolic > 160 or diastolic > 100:
            return "Hypertension sévère"
        elif (systolic >= 140 and systolic <= 159) or (diastolic >= 90 and diastolic <= 99):
            return "Hypertension modérée"
        elif (systolic >= 130 and systolic <= 139) or (diastolic >= 80 and diastolic <= 89):
            return "Hypertension légère"

        # Catégorisation Hypotension
        elif systolic < 80 or diastolic < 50:
            return "Hypotension sévère"
        elif (systolic >= 80 and systolic <= 89) or (diastolic >= 50 and diastolic <= 59):
            return "Hypotension modérée"
        elif systolic < 90 or diastolic < 60:
            return "Hypotension légère"

        # Normale (entre hypotension et hypertension)
        else:
            return "Normale"
    except:
        return "Non spécifié"

# Appliquer la catégorisation
Sleep_healt["BP_Category"] = Sleep_healt["Blood Pressure"].apply(categorize_blood_pressure)

# Histogramme avec l'ordre logique
category_order = ["Hypotension sévère", "Hypotension modérée", "Hypotension légère", "Normale", "Hypertension légère", "Hypertension modérée", "Hypertension sévère"]

bp_counts = Sleep_healt["BP_Category"].value_counts()
bp_counts = bp_counts.reindex(category_order, fill_value=0)

# Créer le graphique
plt.figure(figsize=(10, 6))
bp_counts.plot(kind="bar")
plt.title("Distribution des catégories de pression sanguine")
plt.xlabel("Catégorie de pression sanguine")
plt.ylabel("Nombre de patients")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
save_plot_with_title()
plt.show()

# Histogramme de la variable 'Heart Rate'
Sleep_healt["Heart Rate"].hist()
plt.title("Distribution du rythme cardiaque")
plt.xlabel("Rythme cardiaque")
plt.ylabel("Nombre de patients")
save_plot_with_title()
plt.show()

# Histogramme de la variable 'Daily Steps'
Sleep_healt["Daily Steps"].hist()
plt.title("Daily Steps")
plt.xlabel("Daily Steps")
plt.ylabel("Nombres de patients")

# Histogramme de la variable 'Sleep Disorder'
Sleep_healt["Sleep Disorder"].value_counts().plot(kind="bar")
plt.title("Trouble du sommeil")
plt.xlabel("Daily Steps")
plt.ylabel("Nombres de patients")
plt.xticks(rotation=0)
save_plot_with_title()
plt.show()

# Génération des boxplots bivariés pour chaque variable numérique sur des graphiques séparés
num_vars = ["Age", "Sleep Duration", "Quality of Sleep", "Physical Activity Level", "Stress Level", "Heart Rate", "Daily Steps"]

# Encodage de la variable cible pour analyse bivariée
Sleep_healt["Outcome"] = Sleep_healt["Sleep Disorder"].map({"No Sleep Disorder": 0, "Insomnia": 1, "Sleep Apnea": 1})

Sleep_healt["BMI_numeric"] = Sleep_healt["BMI Category"].map({"Normal": 0, "Normal Weight": 0, "Overweight": 1, "Obese": 2})
num_vars.append("BMI_numeric")

for col in num_vars:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=Sleep_healt, x="Outcome", y=col)
    if col == "BMI_numeric":
        plt.title(f'Boxplot de la variable "BMI Category" par rapport à la variable cible "Outcome"')
        plt.ylabel("BMI Category (0=Normal, 1=Overweight, 2=Obese)")
    else:
        plt.title(f'Boxplot de la variable "{col}" par rapport à la variable cible "Outcome"')
        plt.ylabel(col)

    plt.xlabel("Trouble du sommeil (1) ou pas de trouble (0)")
    medians = Sleep_healt.groupby("Outcome")[col].median()
    for i, median in enumerate(medians):
        plt.text(i, median, f'{median:.1f}',
                 ha="center", va='center', fontweight='bold',
                 color="white", bbox=dict(boxstyle='round', facecolor="black", alpha=0.8))
    plt.grid(axis='y', alpha=0.3)
    save_plot_with_title()
    plt.show()
print("\n")

num_vars.remove("BMI_numeric")

# Calcul des statistiques de test de Kruskal-Wallis et des p-valeurs pour chaque variable numérique
p_values = []
for col in num_vars:
    h, pval = kruskal(Sleep_healt[col][Sleep_healt["Outcome"]==0], Sleep_healt[col][Sleep_healt["Outcome"]==1])
    p_values.append(pval)

# Création d'une liste pour stocker les valeurs de significativité
is_significant = []

# Vérification de la significativité pour chaque p-valeur et stockage dans la liste is_significant
for pval in p_values:
    if pval < 0.05:
        is_significant.append("Oui")
    else:
        is_significant.append("Non")

# Affichage des résultats dans un tableau
result_table = pd.DataFrame({"Variable": num_vars, "P-value": p_values, "Significative": is_significant})
result_table.set_index("Variable", inplace=True)
print("\n")
print(result_table)
print("\n")
print("Test de Kruskal-Wallis - Signicativité des variables")
print(result_table)
print("\n")

# Sélection des variables numériques
df_num = Sleep_healt[num_vars].copy()

# Créer df_heatmap
df_heatmap = df_num.copy()

# Ajout des variables catégorielles encodées
df_heatmap["Gender"] = Sleep_healt["Gender"].map({"Male": 0, "Female": 1})
df_heatmap["Occupation"] = Sleep_healt["Occupation"].astype("category").cat.codes
df_heatmap["BMI Category"] = Sleep_healt["BMI_numeric"]
df_heatmap["BP_Category"] = Sleep_healt["BP_Category"].astype("category").cat.codes
df_heatmap["Sleep Disorder"] = Sleep_healt["Sleep Disorder"].map({"No Sleep Disorder": 0, "Insomnia": 1, "Sleep Apnea": 2})

# Génération de la heatmap de corrélation
plt.figure(figsize=(14, 12))
sns.heatmap(df_heatmap.corr(), cmap="coolwarm", annot=True, fmt=".2f",cbar_kws={"label": "Corrélation"}, square=True)
plt.title("Heatmap de corrélation des variables")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
save_plot_with_title()
plt.show()
# Montre les relations linéaires entre toutes les variables
# C'est une analyse bivariée (relation entre 2 variables à la fois)
# Mesure : Corrélation de Pearson (-1 à +1)

# Modélisation de l'arbre decisionnel

# Encodage des variables catégorielles pour la modélisation
Sleep_healt_model = Sleep_healt.copy()
Sleep_healt_model["Gender"] = Sleep_healt_model["Gender"].map({"Male": 0, "Female": 1})
Sleep_healt_model["Occupation"] = Sleep_healt_model["Occupation"].astype("category").cat.codes
Sleep_healt_model["BP_Category"] = Sleep_healt_model["BP_Category"].astype("category").cat.codes

# Séparation des données en variables prédictives et variable cible
X = Sleep_healt_model.drop(["Person ID", "Blood Pressure", "Outcome", "Sleep Disorder","Tranche_Age","Tranche_Sommeil", "BMI Category"], axis=1)
X = X.rename(columns={"BMI_numeric": "BMI Category"})
y = Sleep_healt_model["Outcome"]
# Montre quelles variables sont les plus utiles pour prédire les troubles du sommeil
# C'est une analyse multivariée (l'arbre utilise plusieurs variables ensemble)
# Mesure : Contribution à la qualité de prédiction

# Division des données en ensemble d'entraînement et ensemble de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Construction de l'arbre de décision
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

plt.figure(figsize=(40,30))
plot_tree(tree, feature_names=X.columns)
plt.title("Classificateur par arbre décision")
save_plot_with_title()
plt.show()

# Affichage de l'importance des variables
importances = tree.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,5))
plt.bar(range(X_train.shape[1]), importances[indices])
plt.xticks(range(X_train.shape[1]), X.columns[indices], rotation=45)
plt.title("Importance des variables")
save_plot_with_title()
plt.show()
print("\n")

# Prédictions sur l'ensemble d'entraînement
y_train_pred = tree.predict(X_train)

# Évaluation des performances sur l'ensemble d'entraînement
train_accuracy = accuracy_score(y_train, y_train_pred)
train_confusion = confusion_matrix(y_train, y_train_pred)
train_classification = classification_report(y_train, y_train_pred)

print("==========Performance sur la base apprentissage==========") # ou print("PERFORMANCES - ARBRE NON ÉLAGUÉ - BASE D'APPRENTISSAGE")
print("Train Accuracy : ", train_accuracy)
print("Train Confusion matrix : \n", train_confusion)
print("Train Classification report : \n", train_classification)
print("\n")

# Prédictions sur l'ensemble de test
y_test_pred = tree.predict(X_test)

# Évaluation des performances sur l'ensemble de test
print("==========Performance sur la base test==========") # ou print("PERFORMANCES - ARBRE NON ÉLAGUÉ - BASE DE TEST")
test_accuracy = accuracy_score(y_test, y_test_pred)
test_confusion = confusion_matrix(y_test, y_test_pred)
test_classification = classification_report(y_test, y_test_pred)

print("Test Accuracy : ", test_accuracy)
print("Test Confusion matrix : \n", test_confusion)
print("Test Classification report : \n", test_classification)
print("\n")

# Prédictions sur l'ensemble d'entraînement
y_train_pred = tree.predict(X_train)
y_train_prob = tree.predict_proba(X_train)[:, 1]

# Prédictions sur l'ensemble de test
y_test_pred = tree.predict(X_test)
y_test_prob = tree.predict_proba(X_test)[:, 1]

# Courbe ROC pour la base train
fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_train_prob)
roc_auc_train = roc_auc_score(y_train, y_train_prob)

# Courbe ROC pour la base test
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_test_prob)
roc_auc_test = roc_auc_score(y_test, y_test_prob)

plt.figure()
plt.plot(fpr_train, tpr_train, label="Train ROC curve (area = %0.2f)" % roc_auc_train, linewidth=2)
plt.plot(fpr_test, tpr_test, label="Test ROC curve (area = %0.2f)" % roc_auc_test, linewidth=2)
plt.plot([0, 1], [0, 1], "k--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic")
plt.legend(loc="lower right")
save_plot_with_title()
plt.show()

# Liste des profondeurs maximales à tester
max_depths = range(1, 15)

# Liste pour stocker les scores de validation croisée
cv_scores = []

# Pour chaque profondeur maximale, entraîner un arbre de décision et calculer le score de validation croisée
for depth in max_depths:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    scores = cross_val_score(tree, X_train, y_train, cv=10)
    cv_scores.append(np.mean(scores))

# Tracer le graphique de validation croisée pour choisir la complexité de l'arbre optimal
plt.plot(max_depths, cv_scores)
plt.xlabel("Profondeur maximale de l\'arbre")
plt.ylabel("Score de validation croisée")
plt.title("Validation croisée pour l\'arbre de décision")
save_plot_with_title()
plt.show()
print("\n")

# Trouver la profondeur maximale qui donne le meilleur score de validation croisée
best_depth = max_depths[np.argmax(cv_scores)]
print("="*80)
print("La meilleure profondeur maximale est :", best_depth)
print(f"{"="*80}\n")
print("\n")

# Construire l'arbre de décision avec la profondeur maximale choisie
tree = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
tree.fit(X_train, y_train)

plt.figure(figsize=(20,10))
plot_tree(tree, feature_names=X.columns)
plt.title("Arbre de décision avec la profondeur choisie")
save_plot_with_title()
plt.show()

# Affichage de l'importance des variables
importances = tree.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,5))
plt.title("Importance des variables")
plt.bar(range(X_train.shape[1]), importances[indices])
plt.xticks(range(X_train.shape[1]), X.columns[indices], rotation=45)
save_plot_with_title()
plt.show()

# Prédictions sur l'ensemble d'entraînement
y_train_pred = tree.predict(X_train)

# Évaluation des performances sur l'ensemble d'entraînement
train_accuracy = accuracy_score(y_train, y_train_pred)
train_confusion = confusion_matrix(y_train, y_train_pred)
train_classification = classification_report(y_train, y_train_pred)

print("==========Performance sur la base apprentissage==========") # ou print(f"PERFORMANCES - ARBRE ÉLAGUÉ (PROFONDEUR={best_depth}) - BASE D'APPRENTISSAGE")
print("Train Accuracy : ", train_accuracy)
print("Train Confusion matrix : \n", train_confusion)
print("Train Classification report : \n", train_classification)
print("\n")

# Prédictions sur l'ensemble de test
y_test_pred = tree.predict(X_test)

# Évaluation des performances sur l'ensemble de test
print("==========Performance sur la base test==========") # ou print(f"PERFORMANCES - ARBRE ÉLAGUÉ (PROFONDEUR={best_depth}) - BASE DE TEST")
test_accuracy = accuracy_score(y_test, y_test_pred)
test_confusion = confusion_matrix(y_test, y_test_pred)
test_classification = classification_report(y_test, y_test_pred)

print("Test Accuracy : ", test_accuracy)
print("Test Confusion matrix : \n", test_confusion)
print("Test Classification report : \n", test_classification)
print("\n")

y_train_pred = tree.predict(X_train)
# Prédictions sur l'ensemble d'entraînement
y_train_pred = tree.predict(X_train)
y_train_prob = tree.predict_proba(X_train)[:, 1]

# Prédictions sur l'ensemble de test
y_test_pred = tree.predict(X_test)
y_test_prob = tree.predict_proba(X_test)[:, 1]

# Courbe ROC pour la base train
fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_train_prob)
roc_auc_train = roc_auc_score(y_train, y_train_prob)

# Courbe ROC pour la base test
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_test_prob)
roc_auc_test = roc_auc_score(y_test, y_test_prob)

plt.figure()
plt.plot(fpr_train, tpr_train, label="Train ROC curve (area = %0.2f)" % roc_auc_train, linewidth=2)
plt.plot(fpr_test, tpr_test, label="Test ROC curve (area = %0.2f)" % roc_auc_test, linewidth=2)
plt.plot([0, 1], [0, 1], "k--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic")
plt.legend(loc="lower right")
save_plot_with_title()
plt.show()

print(f"""
================================================================================
                         Analyse terminée
================================================================================
Nombre total de graphiques générés : {plot_counter}
Précision du modèle final (test) : {test_accuracy*100:.2f}%
Profondeur optimale de l'arbre : {best_depth}

Résumé des performances du modèle élagué :
  - Accuracy (train) : {train_accuracy*100:.2f}%
  - Accuracy (test)  : {test_accuracy*100:.2f}%
  - AUC ROC (train)  : {roc_auc_train:.2f}
  - AUC ROC (test)   : {roc_auc_test:.2f}

Tous les graphiques ont été sauvegardés dans le répertoire "Graphiques".
""")

print("\n")
print("="*80)
print("="*80)
print("""

            ████████╗██╗  ██╗███████╗     ███████╗███╗   ██╗██████╗ 
            ╚══██╔══╝██║  ██║██╔════╝     ██╔════╝████╗  ██║██╔══██╗
               ██║   ███████║█████╗       █████╗  ██╔██╗ ██║██║  ██║
               ██║   ██╔══██║██╔══╝       ██╔══╝  ██║╚██╗██║██║  ██║
               ██║   ██║  ██║███████╗     ███████╗██║ ╚████║██████╔╝
               ╚═╝   ╚═╝  ╚═╝╚══════╝     ╚══════╝╚═╝  ╚═══╝╚═════╝ 

                         É T U D E   D E   L A   S A N T É  
                               D U   S O M M E I L

""")
print("="*80)
print("="*80)
print("\n")
