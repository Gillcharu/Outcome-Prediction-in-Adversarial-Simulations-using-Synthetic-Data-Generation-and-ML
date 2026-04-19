# Outcome Prediction in Adversarial Simulations using Synthetic Data Generation and ML

This project predicts the probability of success in a strategy-game attack using synthetic battle simulations and machine learning. It does not depend on real battle logs. Instead, it generates thousands of battles with rule-based interactions, trains a regression model on the synthetic data, and recommends stronger attack strategies for a selected defending base.

## Why this project stands out

- It predicts a probability of success instead of only returning `win` or `lose`.
- It creates its own dataset through synthetic battle simulation.
- It combines prediction with brute-force recommendation, turning the model into a decision-support tool.

## Core idea

The system models a battle as a relationship between:

- the attacking setup
- the defending base
- the interaction between troop style and defensive pressure

Examples of rule-based simulation logic:

- air-heavy armies are penalized by strong anti-air defenses
- ground-heavy armies are affected by walls, inferno pressure, and splash-heavy layouts
- siege machines and clan castle support change matchup quality
- pets, heroes, guardians, and spells shift the strength of different attack styles

These simulated battles are used to build a dataset, which is then used to train a `RandomForestRegressor` to estimate `win_probability`.

## Features

- Synthetic battle generation using rule-based combat logic
- Machine learning prediction with `RandomForestRegressor`
- Full attack input support:
  - `TROOPS`
  - `SPELLS`
  - `HEROES`
  - `PETS`
  - `GUARDIANS`
  - `CLAN CASTLE`
  - `SIEGE MACHINE`
- Full defense input support:
  - `CANNON`
  - `ARCHER TOWER`
  - `WALL`
  - `MORTAR`
  - `AIR DEFENCE`
  - `WIZARD TOWER`
  - `AIR SWEEPER`
  - `HIDDEN TESLA`
  - `BOMB TOWER`
  - `X-BOW`
  - `INFERNO TOWER`
  - `EAGLE ARTILLERY`
  - `SCATTERSHOT`
  - `SPELL TOWER`
  - `MONOLITH`
  - `TRAPS`
- Web interface for interactive prediction and strategy comparison
- API endpoint for programmatic access
- Brute-force strategy recommendation for a selected base

## Project structure

- `battle_simulator.py`
  Synthetic data generation, attack/base schema, and battle scoring rules.
- `train_model.py`
  Dataset generation, model training, evaluation, and artifact export.
- `recommend_strategy.py`
  Candidate strategy generation and ranking.
- `app.py`
  Flask application for the website and API.
- `templates/index.html`
  Main user interface.
- `static/style.css`
  Frontend styling.
- `static/app.js`
  Frontend interactions and chart rendering.
- `tests/test_app.py`
  Basic app-level tests.

## ML pipeline

1. Generate synthetic battles using logical game rules.
2. Convert attack and defense configurations into tabular features.
3. Train a `RandomForestRegressor` on `win_probability`.
4. Use the trained model to score new armies against a target base.
5. Search through many candidate attack setups and return the strongest options.

## Inputs modeled

### Attack-side features

#### Troops

- Barbarian
- Archer
- Wizard
- Goblin
- Giant
- Wall Breaker
- Balloon
- Healer
- Dragon
- P.E.K.K.A
- Baby Dragon
- Miner
- Electro Dragon
- Yeti
- Dragon Rider
- Electro Titan
- Root Rider
- Thrower
- Meteor Golem
- Minion
- Hog Rider
- Valkyrie
- Golem
- Witch
- Lava Hound
- Bowler
- Ice Golem
- Apprentice Warden
- Headhunter
- Druid
- Furnace

#### Spells

- Lightning
- Heal
- Rage
- Jump
- Freeze
- Clone
- Invisibility
- Recall
- Revive
- Totem
- Poison
- Earthquake
- Haste
- Skeleton
- Bat
- Overgrowth
- Ice Block

#### Heroes

- Barbarian King
- Archer Queen
- Minion Prince
- Grand Warden
- Royal Champion
- Dragon Duke

#### Pets

- L.A.S.S.I
- Electro Owl
- Mighty Yak
- Unicorn

#### Guardians

- Ground Guardian
- Air Guardian
- Healing Guardian

#### Additional attack fields

- Clan Castle
- Siege Machine

### Defense-side features

- Base Level
- Cannon
- Archer Tower
- Wall
- Mortar
- Air Defence
- Wizard Tower
- Air Sweeper
- Hidden Tesla
- Bomb Tower
- X-Bow
- Inferno Tower
- Eagle Artillery
- Scattershot
- Spell Tower
- Monolith
- Traps

## Dataset

The generated dataset is saved at:

- [synthetic_battles.csv](/Users/charugill/Documents/New%20project%202/artifacts/synthetic_battles.csv)

It contains:

- feature columns for attack composition
- feature columns for detailed base defenses
- aggregate defense pressure features used by the simulator
- `win_probability` as the prediction target

## Model

The project uses:

- `RandomForestRegressor`

Why regression instead of classification:

- probability is more informative than a binary label
- small strategy differences are easier to compare
- the recommendation system can optimize for highest predicted chance of success

## Website

The website lets you:

- enter attack and defense values manually
- use base presets for quick testing
- predict the win probability of your selected army
- compare your army against top recommended strategies
- inspect summarized attack and defense selections

Run the app locally and open:

- [http://127.0.0.1:5000](http://127.0.0.1:5000)

## API

Available endpoints:

- `GET /health`
- `POST /api/predict`

Example:

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"troop_barbarian":10,"troop_archer":12,"spell_rage":2,"hero_barbarian_king":80,"defense_cannon":5,"defense_air_defence":7,"base_level":15,"clan_castle":"cc_yeti","siege_machine":"log_launcher"}'
```

## Setup

```bash
cd "/Users/charugill/Documents/New project 2"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run locally

Train the model and generate the dataset:

```bash
python3 train_model.py
```

Start the web app:

```bash
python3 app.py
```

Run tests:

```bash
python3 -m unittest discover -s tests
```

## Output artifacts

- trained model: [model.joblib](/Users/charugill/Documents/New%20project%202/artifacts/model.joblib)
- generated dataset: [synthetic_battles.csv](/Users/charugill/Documents/New%20project%202/artifacts/synthetic_battles.csv)

## Deployment

The repo includes:

- `Dockerfile`
- `Procfile`
- `gunicorn`

Example Docker flow:

```bash
docker build -t attack-predictor .
docker run -p 5000:5000 attack-predictor
```

## Future improvements

- stronger simulation rules based on deeper game mechanics
- larger candidate search space for recommendations
- feature importance visualization
- better frontend filtering and grouping for large input sets
- deployment with a public demo link
