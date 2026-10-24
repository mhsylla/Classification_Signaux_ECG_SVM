# Classification des Signaux ECG MIT-BIH avec SVM

**Module :** Data Science
**Auteur :** Ameth Sylla
**Professeur :** Monsieur Delpech
**Date :** 05/02/2026

---

## Description du projet

Ce projet vise a developper un systeme de classification automatique de battements cardiaques a partir de signaux ECG (electrocardiogramme). En s'appuyant sur la base de donnees MIT-BIH Arrhythmia Database et un classifieur SVM (Support Vector Machine), le systeme est capable de distinguer les battements normaux des battements anormaux et d'identifier des types specifiques d'arythmies.

L'analyse automatique des signaux ECG est un enjeu majeur de sante publique : les maladies cardiovasculaires representent l'une des principales causes de mortalite dans le monde. Un outil capable de detecter rapidement et avec precision des anomalies cardiaques peut contribuer au diagnostic precoce et a l'amelioration de la prise en charge des patients.

---

## Objectifs

- Charger et traiter les enregistrements ECG depuis la base MIT-BIH Arrhythmia Database.
- Extraire des caracteristiques (features) pertinentes a partir des signaux bruts.
- Appliquer differentes strategies de regroupement des types de battements.
- Entrainer et evaluer un modele SVM pour la classification.
- Analyser l'impact des features et du regroupement sur les performances du modele.

---

## Donnees utilisees

- **Source :** MIT-BIH Arrhythmia Database
- **URL :** https://physionet.org/content/mitdb/1.0.0/
- **Publication originale :** http://ecg.mit.edu/george/publications/mitdb-embs-2001.pdf

### Caracteristiques du dataset

| Caracteristique             | Detail                                       |
|-----------------------------|----------------------------------------------|
| Nombre d'enregistrements    | 48 enregistrements ECG de 30 minutes chacun  |
| Leads                       | 2 leads simultanement                        |
| Nombre de battements        | ~112 606 battements                          |
| Frequence d'echantillonnage | 360 Hz                                       |

### Classes de battements

| Symbole        | Description                                      |
|----------------|--------------------------------------------------|
| N              | Battement normal                                 |
| V              | Contraction ventriculaire prematuree (PVC)       |
| A              | Battement atrial premature                       |
| F              | Fusion (battement normal + ventriculaire)        |
| j, f, x        | Autres types d'arythmies ou battements aberrants |
| Q, +, /, ~     | Battements non classifies, artefacts ou signaux speciaux |

---

## Anatomie d'un battement cardiaque

Un battement cardiaque complet est compose de trois phases successives decrites par trois types d'ondes :

- **Onde P :** Decrit la depolarisation des oreillettes.
- **Complexe QRS :** Reflete la depolarisation rapide des ventricules. C'est le point le plus marque du signal.
- **Onde T :** Represente la repolarisation des ventricules.

Chaque segment de battement est extrait autour du pic R (50 echantillons avant, 50 apres), soit une fenetre de 100 points.

---

## Modele : SVM (Support Vector Machine)

Le SVM est un modele de classification qui cherche a separer les donnees en tracant une frontiere optimale appelee **hyperplan**, de facon a **maximiser la marge** entre cette frontiere et les points les plus proches de chaque classe. Selon la structure des donnees :

- **Kernel lineaire :** La marge est directe si les donnees sont lineairement separables.
- **Kernel polynomial ou RBF :** Une transformation implicite projette les donnees dans un espace de dimension superieure ou elles deviennent separables, sans calculer explicitement les coordonnees. C'est le **kernel trick**.

---

## Structure du projet

```
Classification_Signaux_ECG/
|-- ECG_Classification.ipynb    # Notebook principal (Jupyter)
|-- main.py                     # Point d'entree Python
|-- pyproject.toml              # Configuration du projet
|-- README.md                   # Documentation
```

---

## Plan du notebook

### 1. Chargement des enregistrements ECG avec WFDB

Chargement des enregistrements ECG et de leurs annotations depuis la MIT-BIH Arrhythmia Database via la bibliotheque `wfdb`. Les enregistrements disponibles vont de 100 a 234 (certains sont invalides et sont ignores automatiquement via un bloc `try/except`).

Pour chaque battement, un segment de 100 echantillons est extrait autour du pic R (50 avant, 50 apres).

### 2. Extraction des features

#### 2.1 Features simples (statistiques)

| Feature     | Description                   |
|-------------|-------------------------------|
| Moyenne     | Valeur moyenne du segment     |
| Ecart-type  | Dispersion du signal          |
| Minimum     | Valeur minimale               |
| Maximum     | Valeur maximale               |
| Energie     | Somme des carres des valeurs  |

#### 2.2 Features avancees

| Feature             | Description                                       |
|---------------------|---------------------------------------------------|
| Amplitude du pic R  | Hauteur du pic R principal                        |
| Largeur QRS         | Duree du complexe QRS (en ms)                     |
| Frequence dominante | Frequence dominante via FFT                       |
| Energie FFT         | Energie spectrale du signal                       |
| Entropie            | Entropie normalisee de la distribution du signal  |

### 3. Regroupement des types de battements

#### 3.1 Regroupement simple

| Groupe | Battements inclus                  |
|--------|------------------------------------|
| N      | Battements normaux                 |
| A      | Battements atriaux prematures      |
| T      | Tous les autres types (anomalies)  |

#### 3.2 Regroupement avance / medical

Regroupement plus fin en categories medicalement significatives (types specifiques d'arythmies).

### 4. Visualisation des donnees

- **Visualisation 2D :** Representation des features dans un plan 2D (via PCA ou selection de features).
- **Visualisation 3D :** Representation interactive en 3D avec Plotly.
- **Visualisation des formes d'ondes :** Representation temporelle des principaux types de battements.

### 5. Classification avec SVM

Trois experimentations sont realisees :

| Experience | Regroupement | Features  |
|------------|--------------|-----------|
| SVM 1      | Simple       | Simples   |
| SVM 2      | Simple       | Avancees  |
| SVM 3      | Avance       | Avancees  |

Le pipeline inclut :
- Normalisation des donnees (`StandardScaler`)
- Encodage des labels (`LabelEncoder`)
- Separation train/test (`train_test_split`)
- Classifieur SVM (`SVC`)

### 6. Analyse des resultats et comparaisons

- Visualisation des predictions vs vrais labels
- Matrice de confusion (`confusion_matrix`)
- Metriques d'evaluation : accuracy, precision, recall, F1-score (`classification_report`)
- Discussion sur l'impact des features avancees et du regroupement sur la precision

---

## Installation et utilisation

### Prerequis

- Python >= 3.13
- Environnement virtuel recommande

### Installation des dependances

```bash
# Creer et activer l'environnement virtuel
python -m venv .venv
source .venv/bin/activate     # Linux/macOS
.venv\Scripts\activate        # Windows

# Installer les dependances
pip install wfdb numpy pandas scipy scikit-learn matplotlib seaborn plotly
```

### Lancer le notebook

```bash
jupyter lab ECG_Classification.ipynb
```

---

## Dependances principales

| Bibliotheque   | Usage                                                    |
|----------------|----------------------------------------------------------|
| `wfdb`         | Lecture des enregistrements ECG et annotations MIT-BIH   |
| `numpy`        | Calculs numeriques et manipulation de tableaux           |
| `pandas`       | Structuration des donnees en DataFrame                   |
| `scipy`        | Traitement du signal (FFT, detection de pics, entropie)  |
| `scikit-learn` | Preprocessing, PCA, SVM, metriques d'evaluation          |
| `matplotlib`   | Visualisations 2D                                        |
| `seaborn`      | Visualisations statistiques avancees                     |
| `plotly`       | Visualisations interactives 3D                           |

---

