# Fine-tuning de REVE

## Installation

```bash
pip install -r requirements.txt
```

## Structure

```
train.py          # Entraînement
data.py           # Chargement des données
engine.py         # Boucle d'entraînement / évaluation
reve-repro-main/  # Code de REVE
```

## Utilisation

```bash
python train.py --epochs 1 --mode "linear"
```
