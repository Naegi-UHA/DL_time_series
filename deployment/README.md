# Deployment ECG

## Lancement
Depuis le dossier `deployment/` (ou la racine de ce zip) :

```bash
docker compose up --build
```

Puis ouvrir :
- Front Spring Boot : http://localhost:8080
- API Flask : http://localhost:8080/health-proxy (proxy) ou http://localhost:80 dans le réseau Docker uniquement

## Pré-requis
Avant le build, vérifier que `flask_app/models/` contient bien les artefacts exportés depuis la partie training :
- best_model.keras
- preprocessing.json
- summary.json
- label_map.json
- metadata.json
