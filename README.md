# Strategy Game Attack Predictor

This project builds a machine learning system that predicts the probability of winning an attack in a strategy game and recommends strong attacking strategies against a given base.

## What the project does

- Generates a synthetic dataset of battles using rule-based game logic.
- Trains a `RandomForestRegressor` to predict win probability.
- Accepts attack inputs with troops, spells, heroes, clan castle donation, and siege machines.
- Evaluates many army combinations and recommends the best strategy for a target base.

## Project structure

- `battle_simulator.py`: synthetic data generation and battle scoring rules
- `train_model.py`: dataset creation, model training, and model persistence
- `recommend_strategy.py`: brute-force search for the best attack setup
- `app.py`: Flask web application for prediction and recommendations
- `requirements.txt`: Python dependencies

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_model.py
python recommend_strategy.py
python app.py
```

## Inputs modeled

The attacker configuration includes:

- Troops
- Spells
- Heroes
- Clan castle donation
- Siege machines

The defender configuration includes:

- Base level
- Anti-air defense
- Splash defense
- Wall strength
- Inferno strength
- Trap pressure

## Output

The trained model predicts a win probability from `0` to `1`. The recommender searches candidate armies and returns the setup with the highest predicted success probability.

## Website

The website provides:

- A form for entering troops, spells, heroes, clan castle donation, siege machine, and base defenses
- One-click base presets for common defending styles
- A predicted win percentage for the selected army
- A recommended best strategy for the same defending base
- A ranked shortlist of top candidate attack plans
- A visual comparison chart between your army and the top candidate strategies

## API

The project also includes JSON endpoints:

- `GET /health`
- `POST /api/predict`

Example JSON body:

```json
{
  "troop_barbarian": 12,
  "troop_archer": 16,
  "troop_giant": 8,
  "troop_wizard": 10,
  "troop_dragon": 4,
  "troop_balloon": 6,
  "troop_healer": 2,
  "troop_pekka": 3,
  "spell_rage": 2,
  "spell_heal": 2,
  "spell_freeze": 2,
  "spell_lightning": 1,
  "hero_king": 75,
  "hero_queen": 80,
  "hero_warden": 55,
  "hero_champion": 30,
  "clan_castle": "cc_yeti",
  "siege_machine": "log_launcher",
  "base_level": 14,
  "anti_air_defense": 7,
  "splash_defense": 6,
  "wall_strength": 7,
  "inferno_strength": 7,
  "trap_pressure": 5
}
```

## Deployment

The repo is ready for simple deployment with:

- `gunicorn`
- `Procfile`
- `Dockerfile`
- `.gitignore`
