# Déploiement du modèle ECG

Ce dossier contient la partie déploiement de l'application.

Il lance deux services Docker :

- `flask_app` : API Python qui charge le modèle Keras et fait les prédictions ;
- `spring_app` : application Spring Boot qui sert l'interface web et transmet les requêtes à l'API Flask.

L'utilisateur passe par Spring Boot. L'API Flask n'est pas exposée directement sur la machine hôte.

## Prérequis

Avoir Docker et Docker Compose installés.

Avant de lancer le projet, vérifier que les fichiers du modèle sont présents dans :

```text
flask_app/models/
```

Fichiers attendus :

```text
best_model.keras
preprocessing.json
summary.json
label_map.json
metadata.json
```

## Lancement

Depuis ce dossier :

```bash
docker compose up --build
```
ou simplement (si le fichier à les droits d'exécution) :

```bash
./go 
```

Puis ouvrir l'interface :

```text
http://localhost:8080
```

Pour vérifier que Spring Boot communique bien avec Flask :

```text
http://localhost:8080/health-proxy
```

Réponse attendue :

```json
{"status":"ok"}
```

## Utilisation

L'interface permet de tester le modèle de deux façons :

- en collant directement un signal ECG de 96 valeurs ;
- en envoyant un fichier `.txt`, `.csv` ou `.tsv`.

Le backend accepte aussi une ligne issue du dataset avec un label en première colonne, puis les 96 valeurs du signal.

## Arrêt

```bash
docker compose down
```

## Structure utile

```text
.
├── docker-compose.yml
├── flask_app/
│   ├── main.py
│   ├── classify.py
│   └── models/
└── spring_app/
    ├── src/main/resources/static/index.html
    └── src/main/java/.../SpringAppController.java
```

## Notes

- Le modèle est monté en lecture seule dans le conteneur Flask.
- Spring Boot utilise le nom du service Docker `flask_app` pour joindre l'API Flask.
- Le port exposé côté utilisateur est uniquement `8080`.
