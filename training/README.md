# Training

Partie autonome pour :
- visualiser le dataset ECG200
- entraîner un mini MLP, un mini CNN 1D et un mini RNN
- comparer leurs résultats
- exporter le meilleur modèle vers la partie déploiement

## Commandes (depuis la racine du projet)

### 1) Visualisation
```bash
python3 -m training.src.visualize
```

### 2) Entraînement
```bash
python3 -m training.src.train --config training/configs/mlp.yaml
python3 -m training.src.train --config training/configs/cnn.yaml
python3 -m training.src.train --config training/configs/rnn.yaml
```

### 3) Comparaison
```bash
python3 -m training.src.evaluate
```

### 4) Export du modèle retenu
```bash
python3 -m training.src.export_model --model cnn
```

## Structure
- `configs/` : YAML séparés pour chaque modèle
- `src/models/` : un fichier Python par modèle
- `outputs/models/<nom_modele>/` : résultats de chaque entraînement
- `outputs/metrics/` : tableau comparatif final
