# Classification de battements cardiaques (ECG MIT-BIH) avec SVM

**Auteur :** Ameth Sylla
**Date :** février 2026
**Contexte :** projet réalisé dans le cadre du module Data Science (Master 1 Data Science / IA)
**Stack :** Python, WFDB, NumPy, Pandas, SciPy, Scikit-learn, Matplotlib, Seaborn, Plotly

---

## 1. Fiche express (à relire en 5 minutes)

**Pitch en 30 secondes.** J'ai construit un classifieur de battements cardiaques à partir de la base MIT-BIH Arrhythmia Database (112 606 battements, 48 enregistrements). Chaque battement est découpé en une fenêtre de 100 points autour du pic R. J'en extrais des features statistiques, morphologiques et fréquentielles, puis j'entraîne un SVM à noyau RBF avec pondération des classes pour compenser le fort déséquilibre (67 % de battements normaux). Le passage de features simples à des features avancées fait monter l'accuracy de 68 % à 80 % sur 6 classes cliniques.

**Chiffres clés**

| Élément | Valeur |
|---|---|
| Enregistrements | 48 (30 min chacun, 360 Hz) |
| Battements exploités | 112 606 |
| Dérivation utilisée | 1 seule (première voie du signal) |
| Fenêtre par battement | 100 échantillons (50 avant, 50 après le pic R), soit ≈ 278 ms |
| Features | 5 simples, 10 avancées |
| Modèle | `StandardScaler` + `SVC(kernel="rbf", C=1, gamma="scale", class_weight="balanced")` |
| Split | 80 % train / 20 % test, stratifié, `random_state=42` (22 522 battements de test) |
| Meilleur résultat (6 classes) | accuracy 0,80, F1 macro 0,67, rappel macro 0,82 |
| Résultat (3 classes, features simples) | accuracy 0,84, F1 macro 0,79, rappel macro 0,86 |

**Les 4 messages à retenir**

1. Les features avancées apportent un gain net : +12 points d'accuracy et +0,10 de F1 macro à nombre de classes égal.
2. `class_weight="balanced"` donne un bon rappel sur les classes rares (≈ 0,75), mais au prix d'une faible précision (≈ 0,18 à 0,20 pour « Fusion » et « Supraventriculaire »).
3. Le split est aléatoire au niveau du battement : des battements d'un même patient sont dans le train et dans le test. Les scores sont donc probablement **optimistes** (voir section 12).
4. Les classes « Fusion » et « Supraventriculaire » sont difficiles sans information de contexte temporel (intervalle RR). C'est la première amélioration à faire (section 13).

---

## 2. Contexte et objectifs

L'électrocardiogramme (ECG) enregistre l'activité électrique du cœur. Chaque battement produit une forme d'onde caractéristique, et certaines anomalies (arythmies, extrasystoles, blocs de conduction) modifient cette forme. Classer automatiquement les battements est un enjeu de diagnostic précoce, de télésurveillance (Holter, montres connectées) et de recherche clinique.

**Objectifs du projet**

- Charger les enregistrements et annotations de la MIT-BIH Arrhythmia Database.
- Segmenter chaque battement et en extraire des features pertinentes.
- Tester deux niveaux de regroupement des types de battements (simple à 3 classes, avancé à 6 classes).
- Entraîner et évaluer un SVM, et mesurer l'effet des features et du regroupement.
- Analyser honnêtement les forces et les limites du modèle.

---

## 3. Données

- **Source :** MIT-BIH Arrhythmia Database, <https://physionet.org/content/mitdb/1.0.0/>
- **Publication originale :** <http://ecg.mit.edu/george/publications/mitdb-embs-2001.pdf>
- **Chargement :** à la volée avec `wfdb` (`pn_dir='mitdb'`), donc **une connexion internet est nécessaire**.
- **Annotations :** fichiers `atr`, posés par des cardiologues. La position (`ann.sample`) et le symbole (`ann.symbol`) de chaque battement en sont issus.

### 3.1 Caractéristiques

| Caractéristique | Détail |
|---|---|
| Enregistrements utilisés | 48 (numéros 100 à 234, liste explicite dans le notebook) |
| Durée | 30 minutes par enregistrement |
| Dérivations | 2 disponibles, **seule la première est utilisée** |
| Fréquence d'échantillonnage | 360 Hz |
| Battements retenus | 112 606 |

### 3.2 Distribution des annotations brutes

| Symbole | Nombre | Signification (annotation MIT-BIH) |
|---|---:|---|
| N | 75 045 | Battement normal |
| L | 8 075 | Bloc de branche gauche |
| R | 7 259 | Bloc de branche droit |
| V | 7 129 | Extrasystole ventriculaire (PVC) |
| / | 7 028 | Battement stimulé (pacemaker) |
| A | 2 546 | Extrasystole atriale |
| + | 1 260 | Changement de rythme (annotation) |
| f | 982 | Fusion battement stimulé + normal |
| F | 802 | Fusion battement ventriculaire + normal |
| ~ | 615 | Changement de qualité du signal |
| ! | 472 | Onde de flutter ventriculaire |
| " | 437 | Commentaire (annotation) |
| j | 229 | Échappement jonctionnel |
| x | 193 | Onde P non conduite |
| a | 150 | Extrasystole atriale aberrante |
| \| | 132 | Artefact isolé de type QRS |
| E | 106 | Échappement ventriculaire |
| J | 83 | Extrasystole jonctionnelle |
| Q | 33 | Battement non classifiable |
| e | 16 | Échappement atrial |
| [ et ] | 6 et 6 | Début et fin de flutter/fibrillation ventriculaire |
| S | 2 | Extrasystole supraventriculaire |

**Point d'attention :** la classe « N » représente 66,6 % des battements. Une accuracy élevée peut donc masquer de mauvais résultats sur les anomalies. C'est pourquoi on suit aussi le F1 macro et le rappel par classe.

---

## 4. Rappel : anatomie d'un battement

Un battement complet se compose de trois phases :

- **Onde P** : dépolarisation des oreillettes.
- **Complexe QRS** : dépolarisation rapide des ventricules, c'est le point le plus marqué du signal (le sommet est le **pic R**).
- **Onde T** : repolarisation des ventricules.

La fenêtre de 100 points (≈ 278 ms) est centrée sur le pic R. Elle capte le QRS mais coupe en partie les ondes P et T.

---

## 5. Pipeline

```
Enregistrements MIT-BIH (wfdb)
        │
        ▼
Segmentation : 100 points autour de chaque annotation (pic R)
        │
        ▼
Extraction de features  ──►  5 simples  /  10 avancées
        │
        ▼
Regroupement des labels ──►  3 classes (simple)  /  6 classes (avancé)
        │
        ▼
Split 80/20 stratifié  ──►  StandardScaler  ──►  SVC (RBF, class_weight)
        │
        ▼
Évaluation : accuracy, précision, rappel, F1, matrice de confusion
```

**Note :** la position des battements vient des annotations. Le projet **ne détecte pas** lui-même les pics R (voir limites).

---

## 6. Segmentation

Pour chaque annotation (`idx`, `symbole`) :

- fenêtre `[idx - 50, idx + 50[` sur la première voie (`record.p_signal[:, 0]`) ;
- le battement est conservé uniquement si la fenêtre contient exactement 100 points (les battements trop proches du début ou de la fin sont écartés) ;
- aucun filtrage ni correction de ligne de base n'est appliqué.

---

## 7. Features

### 7.1 Features simples (5)

| Feature | Calcul | Ce que ça capture |
|---|---|---|
| Moyenne | `np.mean` | Niveau moyen (inclut la dérive de ligne de base) |
| Écart-type | `np.std` | Dispersion, amplitude générale |
| Minimum | `np.min` | Creux le plus profond |
| Maximum | `np.max` | Sommet (souvent le pic R) |
| Énergie | `sum(x²)` | Puissance globale du battement |

### 7.2 Features avancées (10 = 5 simples + 5 nouvelles)

| Feature | Calcul exact dans le notebook | Ce que ça capture | Remarque |
|---|---|---|---|
| `R_amp` | Maximum des pics trouvés par `find_peaks` (seuil : moyenne + 0,5 écart-type) | Intensité de l'activité ventriculaire | Vaut 0 si aucun pic n'est trouvé (QRS négatif, par exemple) |
| `QRS_width` | Distance (en ms) entre les points où le signal repasse sous 10 % de `R_amp`, de part et d'autre du pic | Durée de conduction, longue en cas de bloc ou d'extrasystole ventriculaire | Heuristique sans correction de ligne de base |
| `Dom_Freq` | Indice du maximum du module de la FFT (première moitié du spectre) | Contenu fréquentiel | C'est un **indice de bin**, pas des Hz (Hz = indice × 360 / 100). Très souvent 0 (composante continue) |
| `FFT_energy` | `sum(|FFT|²)` | Énergie spectrale | Égale à 100 × Énergie (identité de Parseval) : **redondante** |
| `Entropy` | Entropie de l'histogramme des amplitudes (10 classes) | Désordre de la distribution des valeurs | Mesure la distribution des amplitudes, pas la complexité temporelle |

---

## 8. Regroupement des types de battements

### 8.1 Regroupement simple (3 classes)

| Groupe | Contenu | Effectif |
|---|---|---:|
| **N** | Battement normal | 75 045 |
| **A** (Anomalie) | V, A, F, j, f, L, R, E, J, a, e, S | 27 379 |
| **T** (Technique / bruit) | Q, +, /, \|, ~, x, !, ", [, ] | 10 182 |

> Dans le code, la troisième classe s'appelle `B` ; dans les rapports de classification affichés, elle apparaît sous le nom `T`. C'est la même classe. À harmoniser lors d'une prochaine exécution.

### 8.2 Regroupement avancé / clinique (6 classes)

| Groupe | Labels | Effectif | Signification |
|---|---|---:|---|
| **Normal** | N | 75 045 | Rythme sinusal normal |
| **Bloc** | L, R | 15 334 | Blocs de branche (retard de conduction) |
| **Ventriculaire** | V, E | 7 235 | Extrasystoles et échappements ventriculaires |
| **Supraventriculaire** | A, a, J, j, S, e | 3 026 | Battements d'origine auriculaire ou jonctionnelle |
| **Fusion** | F, f | 1 784 | Fusion entre un battement normal et un battement ectopique ou stimulé |
| **Bruit** | Q, +, /, \|, ~, x, !, ", [, ] | 10 182 | Artefacts, annotations techniques, battements stimulés |

**Attention :** ce regroupement est proche de la norme AAMI (EC57 : N, S, V, F, Q), mais pas identique. Il isole notamment les blocs L et R, que la norme AAMI inclut dans « N ». La classe « Bruit » est hétérogène : elle mélange des artefacts, des annotations qui ne sont pas des battements (`+`, `~`, `"`) et des battements stimulés (`/`, 7 028 exemples, soit les deux tiers de la classe).

---

## 9. Modèle : SVM (Support Vector Machine)

**Principe.** Le SVM cherche l'hyperplan qui sépare les classes en **maximisant la marge** (distance aux points les plus proches de chaque classe). Si les données ne sont pas linéairement séparables, le **kernel trick** les projette implicitement dans un espace de plus grande dimension, sans calculer les coordonnées. Le noyau **RBF** est un bon choix par défaut pour des frontières non linéaires.

**Configuration utilisée**

| Paramètre | Valeur | Rôle |
|---|---|---|
| `kernel` | `"rbf"` | Frontières non linéaires |
| `C` | `1` | Pénalisation des erreurs (grand C = frontière plus stricte, risque de surapprentissage) |
| `gamma` | `"scale"` | Portée d'influence d'un point, calculée à partir de la variance des données |
| `class_weight` | `"balanced"` | Poids inversement proportionnel à la fréquence de la classe : les erreurs sur les classes rares coûtent plus cher |
| `StandardScaler` | oui | Le SVM RBF est sensible aux échelles : on centre et on réduit chaque feature |

Les hyperparamètres n'ont **pas été optimisés** (aucune recherche sur `C` et `gamma`).

---

## 10. Protocole expérimental

- **Split :** `train_test_split(test_size=0.2, random_state=42, stratify=...)`, soit 90 084 battements d'entraînement et 22 522 de test.
- **Pipeline :** `StandardScaler` puis `SVC`, appris uniquement sur le train (pas de fuite du scaler).
- **Métriques :** accuracy, précision, rappel, F1 par classe, moyennes macro et pondérée, matrice de confusion.
- **Pourquoi le F1 macro ?** Il donne le même poids à chaque classe, donc il n'est pas dominé par la classe « Normal ».

**Les trois expériences**

| Expérience | Features | Classes |
|---|---|---|
| SVM 1 | Simples (5) | Simple (3) |
| SVM 2 | Simples (5) | Avancé (6) |
| SVM 3 | Avancées (10) | Avancé (6) |

---

## 11. Résultats

### 11.1 Synthèse

| Expérience | Accuracy | F1 macro | Rappel macro | Précision macro |
|---|---:|---:|---:|---:|
| SVM 1 : features simples, 3 classes | 0,84 | 0,79 | 0,86 | 0,75 |
| SVM 2 : features simples, 6 classes | 0,68 | 0,57 | 0,75 | 0,54 |
| **SVM 3 : features avancées, 6 classes** | **0,80** | **0,67** | **0,82** | **0,64** |

### 11.2 Détail par classe

**SVM 1 (3 classes, features simples)**

| Classe | Précision | Rappel | F1 | Support |
|---|---:|---:|---:|---:|
| A (anomalie) | 0,74 | 0,81 | 0,77 | 5 476 |
| N | 0,96 | 0,84 | 0,89 | 15 010 |
| T (technique) | 0,56 | 0,93 | 0,70 | 2 036 |

**SVM 2 (6 classes, features simples)**

| Classe | Précision | Rappel | F1 | Support |
|---|---:|---:|---:|---:|
| Bloc | 0,68 | 0,79 | 0,73 | 3 034 |
| Bruit | 0,59 | 0,88 | 0,70 | 2 036 |
| Fusion | 0,10 | 0,62 | 0,17 | 352 |
| Normal | 0,98 | 0,61 | 0,75 | 15 010 |
| Supraventriculaire | 0,18 | 0,73 | 0,28 | 612 |
| Ventriculaire | 0,69 | 0,87 | 0,77 | 1 478 |

**SVM 3 (6 classes, features avancées)**

| Classe | Précision | Rappel | F1 | Support |
|---|---:|---:|---:|---:|
| Bloc | 0,78 | 0,85 | 0,81 | 3 067 |
| Bruit | 0,91 | 0,92 | 0,92 | 2 036 |
| Fusion | 0,18 | 0,75 | 0,29 | 357 |
| Normal | 0,98 | 0,77 | 0,86 | 15 010 |
| Supraventriculaire | 0,20 | 0,74 | 0,32 | 605 |
| Ventriculaire | 0,80 | 0,88 | 0,84 | 1 447 |

### 11.3 Lecture des résultats

- **Features simples, 3 puis 6 classes (SVM 1 vers SVM 2).** L'accuracy chute de 0,84 à 0,68. Cinq statistiques globales ne suffisent pas à distinguer des types d'arythmie fins.
- **Apport des features avancées (SVM 2 vers SVM 3).** L'accuracy passe de 0,68 à 0,80 et le F1 macro de 0,57 à 0,67. Les plus gros gains : « Bruit » (F1 0,70 vers 0,92), « Normal » (0,75 vers 0,86), « Bloc » (0,73 vers 0,81), « Ventriculaire » (0,77 vers 0,84). La largeur du QRS et l'amplitude du pic R aident à séparer les battements larges (blocs, ventriculaires) des battements normaux.
- **Compromis précision / rappel.** Avec `class_weight="balanced"`, les classes rares sont bien détectées (rappel ≈ 0,75) mais avec beaucoup de fausses alertes : une précision de 0,18 à 0,20 correspond à environ 4 à 5 faux positifs pour 1 vraie détection. À l'inverse, 23 % des battements normaux sont étiquetés comme anormaux (rappel 0,77).
- **Pourquoi « Fusion » et « Supraventriculaire » restent difficiles.** Un battement de fusion a une forme intermédiaire entre normal et ventriculaire. Un battement supraventriculaire a un QRS souvent quasi normal : ce qui le distingue est son **arrivée prématurée**, une information que la fenêtre d'un seul battement ne contient pas.

---

## 12. Limites connues

À connaître pour pouvoir en parler honnêtement.

1. **Fuite de données entre patients (la plus importante).** Le split est aléatoire par battement : des battements du même patient se retrouvent dans le train et dans le test. Le modèle peut apprendre des particularités du patient (forme du signal, niveau de ligne de base) et non seulement celles de la pathologie. Les scores sont probablement optimistes. L'évaluation de référence sur MIT-BIH est **inter-patients**.
2. **Pas de contexte temporel.** Aucune feature d'intervalle RR (avant et après le battement), pourtant essentielle pour les battements prématurés.
3. **Une seule dérivation** sur les deux disponibles.
4. **Pas de prétraitement du signal** : ni filtrage passe-bande, ni correction de la dérive de ligne de base, ni normalisation par battement.
5. **Features imparfaites** : `FFT_energy` est redondante avec l'énergie (facteur 100), `Dom_Freq` est un indice de bin souvent égal à 0, `R_amp` et `QRS_width` valent 0 pour certains QRS négatifs, l'entropie porte sur la distribution des amplitudes.
6. **Hyperparamètres non optimisés** et **un seul split** : pas de validation croisée, donc pas d'intervalle de confiance sur les scores.
7. **Comparaison SVM 2 / SVM 3 pas strictement identique.** Le split de SVM 2 est stratifié sur le regroupement simple, celui de SVM 3 sur le regroupement avancé. Les jeux de test diffèrent légèrement (par exemple « Bloc » : 3 034 contre 3 067 exemples).
8. **Classe « Bruit » hétérogène** (artefacts, annotations non-battements, battements stimulés).
9. **Les positions des battements viennent des annotations.** En conditions réelles, il faudrait un détecteur de pics R (par exemple Pan-Tompkins).
10. **Cohérence du notebook.** La cellule 22 (SVM 1) ne contient pas `class_weight`, alors que la sortie affichée de l'entraînement l'indique. Il faut relancer le notebook de bout en bout pour que le code et les sorties correspondent.

---

## 13. Pistes d'amélioration (par ordre de priorité)

| Priorité | Amélioration | Pourquoi |
|---|---|---|
| 1 | **Évaluation inter-patients** : stocker l'identifiant d'enregistrement pour chaque battement, utiliser `GroupKFold` ou le protocole DS1/DS2 de la littérature (de Chazal) | Supprime la fuite entre patients, donne des scores réalistes |
| 2 | **Features RR** : intervalle avec le battement précédent et le suivant, ratio RR courant / moyen (les positions sont déjà dans `ann.sample`) | Clé pour les battements supraventriculaires et prématurés |
| 3 | **Meilleures métriques** : matrice de confusion normalisée, courbes précision-rappel, rappel par classe en tête | Évite d'être trompé par l'accuracy |
| 4 | **Optimiser le modèle** : `GridSearchCV` sur `C` et `gamma` avec validation croisée par groupes ; comparer à Random Forest, XGBoost, un CNN 1D | Vérifier que le SVM est bien le bon choix |
| 5 | **Prétraitement** : filtre passe-bande (≈ 0,5-40 Hz), suppression de la ligne de base, normalisation par battement | Réduit l'effet du patient et du bruit |
| 6 | **Corriger les features** : `Dom_Freq` en Hz après retrait de la moyenne, supprimer `FFT_energy`, ajouter des features d'ondelettes | Plus d'information utile, moins de redondance |
| 7 | **Utiliser les deux dérivations** | Information morphologique complémentaire |
| 8 | **Autres stratégies de déséquilibre** : sur-échantillonnage (SMOTE) sur le train uniquement, ajustement des seuils de décision | Meilleur compromis précision / rappel |
| 9 | **Détection automatique des pics R** | Rendre le pipeline utilisable sur un signal brut |
| 10 | **Interprétabilité** : importance des features par permutation | Comprendre ce qui pilote les décisions |

---

## 14. Structure du dépôt

```
Classification_Signaux_ECG_SVM/
├── ECG_Classification.ipynb   # Notebook principal (chargement, features, SVM, résultats)
├── main.py                    # Point d'entrée Python
├── pyproject.toml             # Configuration et dépendances du projet
├── .python-version            # Version de Python
├── ecg.pptx                   # Support de présentation
└── README.md                  # Ce fichier
```

**Plan du notebook :** 1) chargement WFDB, 2) visualisation des principaux types de battements, 3) features simples, 4) regroupement simple, 5) SVM 1, 6) regroupement avancé et SVM 2, 7) features avancées, 8) visualisations 2D/3D, 9) SVM 3, 10) conclusion et perspectives.

---

## 15. Installation et exécution

**Prérequis :** Python ≥ 3.13 (voir `pyproject.toml`) et une connexion internet (téléchargement des données PhysioNet via WFDB).

```bash
# Créer et activer un environnement virtuel
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# Installer les dépendances
pip install wfdb numpy pandas scipy scikit-learn matplotlib seaborn plotly jupyterlab

# Lancer le notebook
jupyter lab ECG_Classification.ipynb
```

Le notebook a été conçu pour **Google Colab** (la cellule de montage de Drive est commentée). Le premier chargement des 48 enregistrements peut prendre plusieurs minutes. La reproductibilité est assurée par `random_state=42`.

**Dépendances principales**

| Bibliothèque | Usage |
|---|---|
| `wfdb` | Lecture des enregistrements et annotations MIT-BIH |
| `numpy`, `pandas` | Calcul numérique et DataFrames |
| `scipy` | Détection de pics, FFT, entropie |
| `scikit-learn` | Standardisation, SVM, split, métriques |
| `matplotlib`, `seaborn`, `plotly` | Visualisations 2D, statistiques et 3D interactives |

---

## 16. Glossaire

| Terme | Définition |
|---|---|
| **ECG** | Électrocardiogramme : enregistrement de l'activité électrique du cœur |
| **Dérivation (lead)** | Une « vue » du signal selon un axe d'électrodes |
| **Pic R** | Sommet du complexe QRS, point de repère de chaque battement |
| **QRS** | Dépolarisation des ventricules ; sa largeur renseigne sur la conduction |
| **Extrasystole** | Battement prématuré (supraventriculaire ou ventriculaire) |
| **Bloc de branche** | Retard de conduction dans une branche du faisceau de His (L : gauche, R : droite) |
| **Battement de fusion** | Fusion d'un battement normal et d'un battement ectopique ou stimulé |
| **Intervalle RR** | Temps entre deux pics R consécutifs |
| **Noyau RBF** | Noyau gaussien du SVM, pour les frontières non linéaires |
| **`C` / `gamma`** | Pénalisation des erreurs / portée d'influence d'un point |
| **`class_weight`** | Pondération des classes pour compenser le déséquilibre |
| **Stratification** | Conserver les proportions de classes dans train et test |
| **Précision / Rappel / F1** | Fiabilité des prédictions / capacité à tout retrouver / compromis des deux |
| **F1 macro** | Moyenne des F1 par classe, sans pondération par l'effectif |
| **Fuite de données (leakage)** | Information du test qui se retrouve indirectement dans l'entraînement |
| **AAMI (EC57)** | Norme de regroupement des battements : N, S, V, F, Q |


---

## 18. Références

- Moody, G.B., Mark, R.G. *The impact of the MIT-BIH Arrhythmia Database.* IEEE Engineering in Medicine and Biology Magazine, 2001.
- Goldberger et al. *PhysioBank, PhysioToolkit, and PhysioNet.* Circulation, 2000.
- Base de données : <https://physionet.org/content/mitdb/1.0.0/>
- Documentation WFDB-Python : <https://wfdb.readthedocs.io/>
- Documentation Scikit-learn (SVM) : <https://scikit-learn.org/stable/modules/svm.html>
