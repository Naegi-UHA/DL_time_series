# Classification ECG200

Projet de classification de signaux ECG avec des modèles de deep learning.

Le projet permet d’entraîner plusieurs modèles sur le dataset ECG200, de comparer leurs résultats, puis d’exporter le modèle choisi pour l’utiliser dans l’application de déploiement.

## Sommaire

- [Prérequis](#prérequis)
- [Installation](#installation)
- [Dataset](#dataset)
- [Structure du projet](#structure-du-projet)
- [Vérifier le dataset](#vérifier-le-dataset)
- [Entraîner les modèles](#entraîner-les-modèles)
- [Comparer les modèles](#comparer-les-modèles)
- [Exporter le modèle choisi](#exporter-le-modèle-choisi)
- [Déroulement conseillé](#déroulement-conseillé)
- [Nettoyer les fichiers générés](#nettoyer-les-fichiers-générés)
- [Notes utiles](#notes-utiles)
- [Déploiement](#déploiement)

## Prérequis

Il faut avoir Python installé.

Les commandes d’entraînement doivent être lancées depuis la racine du projet, et non depuis le dossier `training`.

Exemple :

```bash
python -m training.src.train --config training/configs/cnn.yaml
```

## Installation

Depuis la racine du projet, créer puis activer un environnement virtuel :

```bash
python -m venv .venv
source .venv/bin/activate
```

Sur Windows :

```bash
python -m venv .venv
.venv\Scripts\activate
```

Installer ensuite les dépendances Python :

```bash
pip install -r training/requirements.txt
```

Le fichier `requirements.txt` se trouve dans le dossier `training`, car il concerne uniquement la partie entraînement des modèles.

## Dataset

Les fichiers du dataset ECG200 doivent être placés dans le dossier `data`, à la racine du projet.

Structure attendue :

```txt
data/
├── ECG200_TRAIN.tsv
└── ECG200_TEST.tsv
```

Les labels d’origine présents dans les fichiers TSV sont :

```txt
-1
1
```

Le code garde ces labels lisibles dans les résultats.

Pendant l’entraînement uniquement, les labels sont convertis en interne :

```txt
-1 -> 0
 1 -> 1
```

Cette conversion est nécessaire car Keras attend des identifiants de classes comme `0` et `1` avec `sparse_categorical_crossentropy`.

## Structure du projet

```txt
.
├── data/                  # dataset ECG200
├── training/              # code d'entraînement des modèles
│   ├── configs/           # paramètres d'entraînement
│   ├── requirements.txt   # dépendances Python
│   └── src/               # scripts Python
├── deployment/            # application et fichiers de déploiement
└── README.md
```

## Vérifier le dataset

Avant d’entraîner un modèle, il est possible de générer quelques graphiques pour vérifier rapidement les données :

```bash
python -m training.src.visualize
```

Cette commande crée :

```txt
training/outputs/figures/ecg_examples.png
training/outputs/figures/class_distribution_train.png
```

## Entraîner les modèles

Trois modèles sont disponibles :

```txt
mlp
cnn
rnn
```

Commandes :

```bash
python -m training.src.train --config training/configs/mlp.yaml
python -m training.src.train --config training/configs/cnn.yaml
python -m training.src.train --config training/configs/rnn.yaml
```

Chaque entraînement crée un dossier dans :

```txt
training/outputs/models/
```

Exemple :

```txt
training/outputs/models/cnn/
├── best_model.keras
├── summary.json
├── preprocessing.json
├── history.json
├── loss.png
├── accuracy.png
└── model_summary.txt
```

## Comparer les modèles

Après avoir entraîné les modèles, lancer :

```bash
python -m training.src.evaluate
```

Cette commande crée :

```txt
training/outputs/metrics/comparison.csv
training/outputs/metrics/comparison.json
```

Ces fichiers permettent de comparer les modèles avec des métriques comme :

```txt
accuracy
precision
recall
f1-score
temps d'entraînement
temps d'inférence
```

## Exporter le modèle choisi

Une fois le meilleur modèle choisi, il faut l’exporter pour l’application de déploiement.

Exemple avec le modèle CNN :

```bash
python -m training.src.export_model --model <nom_du_model>
```

L’export crée ou copie les fichiers suivants :

```txt
best_model.keras
summary.json
preprocessing.json
label_map.json
metadata.json
```

Le fichier `label_map.json` permet de convertir la sortie du modèle vers les vrais labels ECG.

Exemple :

```json
{
  "0": -1,
  "1": 1
}
```

Ainsi, même si le modèle prédit `0` ou `1`, l’application peut afficher les labels d’origine du dataset : `-1` ou `1`.

## Déroulement conseillé

Depuis un projet propre, l’ordre habituel est :

```bash
pip install -r training/requirements.txt
python -m training.src.visualize
python -m training.src.train --config training/configs/mlp.yaml
python -m training.src.train --config training/configs/cnn.yaml
python -m training.src.train --config training/configs/rnn.yaml
python -m training.src.evaluate
python -m training.src.export_model --model <nom_du_model>
```

Ensuite, lancer l’application de déploiement.

## Nettoyer les fichiers générés

Le dossier `training/outputs` contient uniquement des fichiers générés.

Il peut être supprimé sans casser le code :

```bash
rm -rf training/outputs
```

Le dossier sera recréé automatiquement lorsque les scripts seront relancés.

## Notes utiles

Les fichiers dans `training/configs/` permettent de modifier les paramètres d’entraînement.

Les fichiers dans `training/src/models/` contiennent les architectures des modèles.

Les fichiers dans `training/outputs/` sont des résultats générés, pas du code source.

Pour utiliser un modèle entraîné dans l’application de déploiement, il faut toujours lancer la commande d’export :

```bash
python -m training.src.export_model --model <nom_du_model>
```

## Déploiement

Pour la partie déploiement, voir le `README.md` dans le dossier `deployment`.

