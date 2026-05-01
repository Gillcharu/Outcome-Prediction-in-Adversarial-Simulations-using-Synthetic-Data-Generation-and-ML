# Outcome Prediction in Adversarial Simulations using Synthetic Data Generation and ML

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![CI](https://img.shields.io/github/actions/workflow/status/Gillcharu/Outcome-Prediction-in-Adversarial-Simulations-using-Synthetic-Data-Generation-and-ML/ci.yml?branch=main&style=flat-square&label=CI)](https://github.com/Gillcharu/Outcome-Prediction-in-Adversarial-Simulations-using-Synthetic-Data-Generation-and-ML/actions)

This project predicts the probability of success in a strategy-game attack using synthetic battle simulations and machine learning. Instead of relying on real battle logs, it generates a large synthetic dataset with rule-based combat logic, trains a regression model on those simulated outcomes, and recommends stronger attacking strategies for a selected defending base.

## 30-second summary

- `Problem`: predict attack success without access to real battle logs
- `Approach`: simulate battles, train a Random Forest regressor, and rank stronger strategies
- `Stack`: Python, Flask, scikit-learn, pandas, NumPy
- `Output`: win probability plus recommendation reasoning
- `Interface`: multi-page web flow with attack setup, defence setup, and results

## Overview

The project combines three connected ideas:

- synthetic data generation for adversarial simulations
- machine learning for outcome probability prediction
- candidate-search recommendation for better strategy selection

This makes the project more than a basic classifier. It is a full decision-support pipeline that can estimate success and recommend what to do next.

## Why this project is strong

- It predicts a probability instead of only returning `win` or `lose`.
- It creates its own dataset through rule-based synthetic simulation.
- It supports a large feature space across attack and defense composition.
- It uses a trained ML model and a recommendation layer on top.
- It includes a web app and API, making it easy to demonstrate.

## Problem statement

Real battle data is not always available for strategy-game research projects. To solve that, this project simulates battles using domain-inspired rules such as:

- air-heavy armies struggle against strong anti-air defenses
- ground pushes are affected by walls, infernos, and splash structures
- siege machines and clan castle choices change matchup quality
- spells, heroes, pets, and guardians influence style-specific effectiveness

These simulations are used to build a supervised learning dataset with `win_probability` as the target.

## System workflow

```text
Attack Input + Base Input
        ->
Synthetic Battle Simulation
        ->
Generated Dataset
        ->
Random Forest Regression Model
        ->
Win Probability Prediction
        ->
Candidate Strategy Recommendation
```

## Features

- Synthetic battle generation using rule-based combat logic
- Probability prediction with `RandomForestRegressor`
- Detailed attack-side feature support
- Detailed defense-side feature support
- Multi-page web interface for attack setup, defence setup, and results
- JSON API with input validation for programmatic access
- Candidate search for top attacking strategies
- Recommendation explanations that describe why a strategy fits the selected base
- Exported evaluation artifacts including metrics and feature importance
- Feature-importance visualization in the results page

## Inputs modeled

### Attack-side inputs

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

#### Other attack settings

- Clan Castle
- Siege Machine

### Defense-side inputs

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

## Machine learning pipeline

1. Generate thousands of synthetic battles.
2. Convert attack and defense configurations into tabular features.
3. Train a `RandomForestRegressor` to predict `win_probability`.
4. Score a user-selected army against a target base.
5. Search candidate strategies and rank the strongest options.

## Important project note

This is a synthetic-data project. The target label `win_probability` is generated by the rule-based simulator, so the ML model is learning to approximate a simulation function rather than discovering patterns from real gameplay logs. That is acceptable for the project goal, but it should be understood as a learned approximation of the simulator, not proof of real-world game performance.

The machine-learning layer is included to demonstrate a full predictive pipeline on synthetic data: dataset generation, feature engineering, model training, evaluation, explanation artifacts, and recommendation. In a real production system with access to real battle logs, the same structure could be retrained on observed outcomes instead of simulator-generated labels.

## Project structure

- `battle_simulator.py`  
  Synthetic data generation, schema definitions, and battle scoring logic.

- `train_model.py`  
  Dataset creation, model training, evaluation, feature importance export, and artifact generation.

- `recommend_strategy.py`  
  Candidate search, ranking, and recommendation reasoning.

- `app.py`  
  Flask application for the website and API.

- `templates/base.html`  
  Shared page shell and step navigation.

- `templates/attack.html`  
  Attack setup page.

- `templates/defence.html`  
  Defence setup page.

- `templates/results.html`  
  Results page for predictions, strategy comparison, and model insights.

- `static/style.css`  
  UI styling.

- `static/app.js`  
  Frontend interactions and chart rendering.

- `tests/test_app.py`  
  Basic application tests.

- `artifacts/metrics.json`  
  Stored evaluation results from the latest model training run.

- `artifacts/feature_importance.csv`  
  Ranked feature importance values from the trained Random Forest model.

## Dataset

The dataset is generated locally by running `python3 train_model.py`.

It contains:

- attack composition features
- detailed defense structure features
- aggregate defense pressure features
- `win_probability` as the target variable

Current dataset summary:

- `7000` simulated battles
- `87` total columns
- target column: `win_probability`

## Model and evaluation

The project uses a `RandomForestRegressor` to estimate battle success probability.

Current held-out test results:

- `MAE = 0.0008`
- `RMSE = 0.0036`
- `R² = 0.3227`

Generated artifacts after training:

- `artifacts/model.joblib`
- `artifacts/metrics.json`
- `artifacts/feature_importance.csv`

If these artifacts are missing, the project can regenerate them locally by running `python3 train_model.py`. The application also supports a load-or-train flow when the saved model is unavailable.

## Website

The web app allows you to:

- build the attack setup in a separate step
- configure the defending base in a separate step
- use base presets for quick testing
- estimate the probability of success for a chosen attack
- compare your current army with stronger recommended strategies
- inspect structured summaries of selected attack and defense inputs
- view model metrics, feature importance, and recommendation reasons directly in the results page

Local URL:

- [http://127.0.0.1:5000/attack](http://127.0.0.1:5000/attack)

## API

Endpoints:

- `GET /health`
- `POST /api/predict`

The API expects a complete army-and-base payload using the same field names as the web form. Invalid or incomplete payloads return `400`.

Example:

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"base_level":15,"clan_castle":"cc_yeti","siege_machine":"log_launcher","troop_barbarian":10,"troop_archer":12,"troop_wizard":8,"troop_goblin":0,"troop_giant":6,"troop_wall_breaker":4,"troop_balloon":6,"troop_healer":4,"troop_dragon":4,"troop_pekka":3,"troop_baby_dragon":0,"troop_miner":8,"troop_electro_dragon":0,"troop_yeti":0,"troop_dragon_rider":0,"troop_electro_titan":0,"troop_root_rider":0,"troop_thrower":0,"troop_meteor_golem":0,"troop_minion":0,"troop_hog_rider":6,"troop_valkyrie":0,"troop_golem":0,"troop_witch":4,"troop_lava_hound":0,"troop_bowler":0,"troop_ice_golem":0,"troop_apprentice_warden":0,"troop_headhunter":0,"troop_druid":0,"troop_furnace":0,"spell_lightning":1,"spell_heal":2,"spell_rage":2,"spell_jump":1,"spell_freeze":2,"spell_clone":0,"spell_invisibility":0,"spell_recall":0,"spell_revive":0,"spell_totem":0,"spell_poison":0,"spell_earthquake":0,"spell_haste":0,"spell_skeleton":0,"spell_bat":0,"spell_overgrowth":0,"spell_ice_block":0,"hero_barbarian_king":80,"hero_archer_queen":82,"hero_minion_prince":55,"hero_grand_warden":60,"hero_royal_champion":32,"hero_dragon_duke":30,"pet_lassi":7,"pet_electro_owl":8,"pet_mighty_yak":7,"pet_unicorn":9,"guardian_ground_guardian":6,"guardian_air_guardian":6,"guardian_healing_guardian":7,"defense_cannon":5,"defense_archer_tower":5,"defense_wall":7,"defense_mortar":5,"defense_air_defence":7,"defense_wizard_tower":6,"defense_air_sweeper":5,"defense_hidden_tesla":6,"defense_bomb_tower":5,"defense_xbow":6,"defense_inferno_tower":6,"defense_eagle_artillery":5,"defense_scattershot":5,"defense_spell_tower":5,"defense_monolith":5,"defense_traps":5}'
```

## GitHub repository settings

Suggested repository description:

`ML project for predicting attack outcomes using synthetic data generation, Random Forest regression, and strategy recommendation.`

Suggested topics:

- `machine-learning`
- `python`
- `flask`
- `random-forest`
- `scikit-learn`
- `synthetic-data`
- `simulation`
- `predictive-modeling`
- `web-app`

## Setup

```bash
git clone https://github.com/Gillcharu/Outcome-Prediction-in-Adversarial-Simulations-using-Synthetic-Data-Generation-and-ML.git
cd Outcome-Prediction-in-Adversarial-Simulations-using-Synthetic-Data-Generation-and-ML
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

For local development, set `FLASK_SECRET_KEY` in `.env` or in your shell before starting the app. If `FLASK_ENV=production`, the app requires `FLASK_SECRET_KEY` to be set.

## Run locally

Train the model and generate the dataset:

```bash
python3 train_model.py
```

Start the app:

```bash
python3 app.py
```

Run tests:

```bash
python3 -m unittest discover -s tests
```

## Deployment

The repository already includes:

- `Dockerfile`
- `Procfile`
- `gunicorn`

Example Docker usage:

```bash
docker build -t attack-predictor .
docker run -p 5000:5000 attack-predictor
```

## GitHub polish checklist

- README with clear project framing
- CI workflow for test execution
- issue templates
- contributing guide
- license
- profile README template

## Future improvements

- stronger simulation rules with more game-specific mechanics
- smarter search strategies such as beam search or evolutionary search
- deeper explanation panels beyond global feature importance
- grouped defense filters in the UI
- deployed public demo

## Current limitations

- The simulator is intentionally lightweight and does not model full game mechanics such as troop pathing, detailed targeting priorities, or time-based DPS interactions.
- The training labels come from the rule-based simulator, so the model should be interpreted as an approximation of the simulator rather than a predictor validated on real battle logs.
- Strategy recommendation currently searches a curated candidate pool, which improves speed and consistency but is not yet a true optimization algorithm.
