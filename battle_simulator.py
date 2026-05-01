from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
import math
from types import MappingProxyType
import random
from typing import Dict, List, Mapping, Tuple

import pandas as pd


TROOP_TYPES = [
    "barbarian",
    "archer",
    "wizard",
    "goblin",
    "giant",
    "wall_breaker",
    "balloon",
    "healer",
    "dragon",
    "pekka",
    "baby_dragon",
    "miner",
    "electro_dragon",
    "yeti",
    "dragon_rider",
    "electro_titan",
    "root_rider",
    "thrower",
    "meteor_golem",
    "minion",
    "hog_rider",
    "valkyrie",
    "golem",
    "witch",
    "lava_hound",
    "bowler",
    "ice_golem",
    "apprentice_warden",
    "headhunter",
    "druid",
    "furnace",
]

SPELL_TYPES = [
    "lightning",
    "heal",
    "rage",
    "jump",
    "freeze",
    "clone",
    "invisibility",
    "recall",
    "revive",
    "totem",
    "poison",
    "earthquake",
    "haste",
    "skeleton",
    "bat",
    "overgrowth",
    "ice_block",
]

HERO_TYPES = [
    "barbarian_king",
    "archer_queen",
    "minion_prince",
    "grand_warden",
    "royal_champion",
    "dragon_duke",
]

PET_TYPES = [
    "lassi",
    "electro_owl",
    "mighty_yak",
    "unicorn",
]

GUARDIAN_TYPES = [
    "ground_guardian",
    "air_guardian",
    "healing_guardian",
]

CLAN_CASTLE_TYPES = [
    "cc_yeti",
    "cc_balloon",
    "cc_dragon",
    "cc_ice_golem",
]

SIEGE_TYPES = [
    "wall_wrecker",
    "battle_blimp",
    "stone_slammer",
    "siege_barracks",
    "log_launcher",
    "flame_flinger",
    "battle_drill",
]

DEFENSE_TYPES = [
    "cannon",
    "archer_tower",
    "wall",
    "mortar",
    "air_defence",
    "wizard_tower",
    "air_sweeper",
    "hidden_tesla",
    "bomb_tower",
    "xbow",
    "inferno_tower",
    "eagle_artillery",
    "scattershot",
    "spell_tower",
    "monolith",
    "traps",
]

TROOP_MAX_COUNTS = {
    "barbarian": 30,
    "archer": 30,
    "wizard": 20,
    "goblin": 20,
    "giant": 14,
    "wall_breaker": 12,
    "balloon": 16,
    "healer": 8,
    "dragon": 12,
    "pekka": 10,
    "baby_dragon": 10,
    "miner": 18,
    "electro_dragon": 10,
    "yeti": 10,
    "dragon_rider": 10,
    "electro_titan": 8,
    "root_rider": 8,
    "thrower": 12,
    "meteor_golem": 6,
    "minion": 24,
    "hog_rider": 18,
    "valkyrie": 14,
    "golem": 6,
    "witch": 14,
    "lava_hound": 6,
    "bowler": 12,
    "ice_golem": 8,
    "apprentice_warden": 6,
    "headhunter": 8,
    "druid": 8,
    "furnace": 6,
}

TROOP_HOUSING_SPACE = {
    "barbarian": 1,
    "archer": 1,
    "wizard": 4,
    "goblin": 1,
    "giant": 5,
    "wall_breaker": 2,
    "balloon": 5,
    "healer": 14,
    "dragon": 20,
    "pekka": 25,
    "baby_dragon": 10,
    "miner": 6,
    "electro_dragon": 30,
    "yeti": 18,
    "dragon_rider": 25,
    "electro_titan": 32,
    "root_rider": 20,
    "thrower": 6,
    "meteor_golem": 30,
    "minion": 2,
    "hog_rider": 5,
    "valkyrie": 8,
    "golem": 30,
    "witch": 12,
    "lava_hound": 30,
    "bowler": 6,
    "ice_golem": 15,
    "apprentice_warden": 20,
    "headhunter": 6,
    "druid": 16,
    "furnace": 18,
}
MAX_ARMY_CAPACITY = 320
DEFAULT_DATASET_SAMPLES = 7000

SPELL_MAX_COUNTS = {spell: 4 for spell in SPELL_TYPES}
SPELL_MAX_COUNTS.update({"clone": 2, "recall": 2, "revive": 2, "totem": 2})

HERO_MAX_LEVELS = {
    "barbarian_king": 95,
    "archer_queen": 95,
    "minion_prince": 70,
    "grand_warden": 70,
    "royal_champion": 45,
    "dragon_duke": 45,
}

PET_MAX_LEVELS = {
    "lassi": 10,
    "electro_owl": 10,
    "mighty_yak": 10,
    "unicorn": 10,
}

GUARDIAN_MAX_LEVELS = {
    "ground_guardian": 10,
    "air_guardian": 10,
    "healing_guardian": 10,
}

TROOP_AIR_POWER = {
    "archer": 0.15,
    "wizard": 0.35,
    "balloon": 1.7,
    "dragon": 1.9,
    "baby_dragon": 1.25,
    "electro_dragon": 2.2,
    "dragon_rider": 1.85,
    "minion": 0.8,
    "lava_hound": 1.25,
}

TROOP_GROUND_POWER = {
    "barbarian": 0.45,
    "archer": 0.2,
    "wizard": 1.0,
    "goblin": 0.45,
    "giant": 1.1,
    "wall_breaker": 0.7,
    "pekka": 2.2,
    "miner": 1.25,
    "yeti": 1.7,
    "electro_titan": 1.9,
    "root_rider": 1.8,
    "thrower": 1.4,
    "meteor_golem": 2.1,
    "hog_rider": 1.4,
    "valkyrie": 1.3,
    "golem": 1.8,
    "witch": 0.85,
    "bowler": 1.5,
    "ice_golem": 0.85,
    "apprentice_warden": 0.35,
    "headhunter": 0.7,
    "druid": 0.25,
    "furnace": 0.8,
}

TROOP_SUPPORT_POWER = {
    "archer": 0.15,
    "wizard": 0.25,
    "healer": 1.6,
    "baby_dragon": 0.2,
    "yeti": 0.3,
    "witch": 0.95,
    "lava_hound": 0.3,
    "ice_golem": 0.9,
    "apprentice_warden": 1.1,
    "headhunter": 0.25,
    "druid": 1.25,
    "furnace": 0.55,
}

SPELL_AIR_POWER = {
    "rage": 2.0,
    "freeze": 1.8,
    "clone": 1.8,
    "haste": 1.6,
    "invisibility": 1.1,
    "bat": 1.0,
    "overgrowth": 0.8,
    "ice_block": 0.9,
}

SPELL_GROUND_POWER = {
    "heal": 1.9,
    "rage": 1.7,
    "jump": 1.4,
    "freeze": 1.3,
    "earthquake": 1.5,
    "skeleton": 0.8,
    "overgrowth": 1.0,
    "totem": 1.0,
    "revive": 1.2,
}

SPELL_SUPPORT_POWER = {
    "heal": 1.3,
    "rage": 1.0,
    "recall": 0.9,
    "revive": 1.3,
    "poison": 0.7,
    "invisibility": 1.2,
    "ice_block": 1.0,
}

DEFENSE_AIR_WEIGHTS = {
    "archer_tower": 1.2,
    "air_defence": 2.3,
    "wizard_tower": 1.1,
    "air_sweeper": 1.6,
    "hidden_tesla": 1.1,
    "xbow": 1.0,
    "inferno_tower": 1.2,
    "eagle_artillery": 1.0,
    "scattershot": 1.15,
    "spell_tower": 0.9,
    "monolith": 1.1,
    "traps": 0.8,
}

DEFENSE_GROUND_WEIGHTS = {
    "cannon": 0.95,
    "archer_tower": 0.7,
    "mortar": 1.2,
    "wizard_tower": 1.2,
    "hidden_tesla": 0.9,
    "bomb_tower": 1.3,
    "xbow": 1.0,
    "inferno_tower": 1.3,
    "eagle_artillery": 1.2,
    "scattershot": 1.6,
    "spell_tower": 1.0,
    "monolith": 1.7,
    "traps": 1.0,
}

DEFENSE_WALL_WEIGHTS = {
    "wall": 2.5,
    "cannon": 0.4,
    "xbow": 0.5,
    "monolith": 0.8,
}

DEFENSE_SPLASH_WEIGHTS = {
    "mortar": 1.7,
    "wizard_tower": 1.8,
    "bomb_tower": 1.6,
    "eagle_artillery": 1.4,
    "scattershot": 2.0,
    "spell_tower": 1.4,
}


def make_zero_map(keys: List[str]) -> Dict[str, int]:
    return {key: 0 for key in keys}


def merge_setup(keys: List[str], updates: Dict[str, int]) -> Dict[str, int]:
    values = make_zero_map(keys)
    values.update(updates)
    return values


def enforce_army_capacity(troops: Dict[str, int], capacity: int = MAX_ARMY_CAPACITY) -> Dict[str, int]:
    adjusted = dict(troops)
    total_capacity = sum(adjusted[name] * TROOP_HOUSING_SPACE[name] for name in TROOP_TYPES)
    if total_capacity <= capacity:
        return adjusted

    ranked = sorted(
        TROOP_TYPES,
        key=lambda name: (
            adjusted[name] * TROOP_HOUSING_SPACE[name],
            TROOP_HOUSING_SPACE[name],
            adjusted[name],
        ),
        reverse=True,
    )
    while total_capacity > capacity:
        changed = False
        for troop in ranked:
            if adjusted[troop] <= 0:
                continue
            adjusted[troop] -= 1
            total_capacity -= TROOP_HOUSING_SPACE[troop]
            changed = True
            if total_capacity <= capacity:
                break
        if not changed:
            break
    return adjusted


@dataclass(frozen=True)
class AttackConfig:
    troops: Mapping[str, int]
    spells: Mapping[str, int]
    heroes: Mapping[str, int]
    pets: Mapping[str, int]
    guardians: Mapping[str, int]
    clan_castle: str
    siege_machine: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "troops", MappingProxyType(dict(self.troops)))
        object.__setattr__(self, "spells", MappingProxyType(dict(self.spells)))
        object.__setattr__(self, "heroes", MappingProxyType(dict(self.heroes)))
        object.__setattr__(self, "pets", MappingProxyType(dict(self.pets)))
        object.__setattr__(self, "guardians", MappingProxyType(dict(self.guardians)))


@dataclass(frozen=True)
class BaseConfig:
    base_level: int
    anti_air_defense: float
    splash_defense: float
    wall_strength: float
    inferno_strength: float
    trap_pressure: float
    ground_pressure: float
    defenses: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "defenses", MappingProxyType(dict(self.defenses)))


def aggregate_base_metrics(defenses: Dict[str, int]) -> Dict[str, float]:
    return {
        "anti_air_defense": sum(defenses.get(name, 0) * weight for name, weight in DEFENSE_AIR_WEIGHTS.items()),
        "ground_pressure": sum(defenses.get(name, 0) * weight for name, weight in DEFENSE_GROUND_WEIGHTS.items()),
        "splash_defense": sum(defenses.get(name, 0) * weight for name, weight in DEFENSE_SPLASH_WEIGHTS.items()),
        "wall_strength": sum(defenses.get(name, 0) * weight for name, weight in DEFENSE_WALL_WEIGHTS.items()),
        "inferno_strength": (
            defenses.get("inferno_tower", 0) * 2.2
            + defenses.get("monolith", 0) * 1.8
            + defenses.get("xbow", 0) * 1.0
            + defenses.get("spell_tower", 0) * 1.1
        ),
        "trap_pressure": defenses.get("traps", 0) * 2.3 + defenses.get("bomb_tower", 0) * 0.7,
    }


def random_attack_config(rng: random.Random) -> AttackConfig:
    troops = {name: rng.randint(0, TROOP_MAX_COUNTS[name]) for name in TROOP_TYPES}
    troops = enforce_army_capacity(troops)
    spells = {name: rng.randint(0, SPELL_MAX_COUNTS[name]) for name in SPELL_TYPES}
    heroes = {name: rng.randint(0, HERO_MAX_LEVELS[name]) for name in HERO_TYPES}
    pets = {name: rng.randint(0, PET_MAX_LEVELS[name]) for name in PET_TYPES}
    guardians = {name: rng.randint(0, GUARDIAN_MAX_LEVELS[name]) for name in GUARDIAN_TYPES}
    return AttackConfig(
        troops=troops,
        spells=spells,
        heroes=heroes,
        pets=pets,
        guardians=guardians,
        clan_castle=rng.choice(CLAN_CASTLE_TYPES),
        siege_machine=rng.choice(SIEGE_TYPES),
    )


def random_base_config(rng: random.Random) -> BaseConfig:
    defenses = {name: rng.randint(1, 10) for name in DEFENSE_TYPES}
    metrics = aggregate_base_metrics(defenses)
    return BaseConfig(
        base_level=rng.randint(9, 17),
        anti_air_defense=metrics["anti_air_defense"],
        ground_pressure=metrics["ground_pressure"],
        splash_defense=metrics["splash_defense"],
        wall_strength=metrics["wall_strength"],
        inferno_strength=metrics["inferno_strength"],
        trap_pressure=metrics["trap_pressure"],
        defenses=defenses,
    )


def attack_style_scores(attack: AttackConfig) -> Tuple[float, float, float]:
    air_power = sum(attack.troops[name] * weight for name, weight in TROOP_AIR_POWER.items())
    air_power += sum(attack.spells[name] * weight for name, weight in SPELL_AIR_POWER.items())

    ground_power = sum(attack.troops[name] * weight for name, weight in TROOP_GROUND_POWER.items())
    ground_power += sum(attack.spells[name] * weight for name, weight in SPELL_GROUND_POWER.items())

    support_power = sum(attack.troops[name] * weight for name, weight in TROOP_SUPPORT_POWER.items())
    support_power += sum(attack.spells[name] * weight for name, weight in SPELL_SUPPORT_POWER.items())

    hero_boost = sum(attack.heroes.values()) * 0.05
    air_power += attack.pets["electro_owl"] * 0.8 + attack.guardians["air_guardian"] * 0.8
    ground_power += (
        attack.pets["mighty_yak"] * 0.8
        + attack.guardians["ground_guardian"] * 0.8
        + attack.pets["lassi"] * 0.45
    )
    support_power += (
        hero_boost
        + attack.pets["unicorn"] * 1.0
        + attack.guardians["healing_guardian"] * 1.0
        + attack.pets["lassi"] * 0.15
    )

    return air_power, ground_power, support_power


def attack_phase_profile(attack: AttackConfig) -> Dict[str, float]:
    air_power, ground_power, support_power = attack_style_scores(attack)

    wall_breakers = attack.troops["wall_breaker"] * 1.25
    jump_pressure = attack.spells["jump"] * 2.0 + attack.spells["earthquake"] * 1.5
    siege_breach = {
        "wall_wrecker": 5.0,
        "log_launcher": 4.6,
        "battle_drill": 4.8,
        "siege_barracks": 2.4,
        "battle_blimp": 1.8,
        "stone_slammer": 1.6,
        "flame_flinger": 1.2,
    }.get(attack.siege_machine, 1.0)

    air_troop_mass = sum(attack.troops[name] for name in TROOP_AIR_POWER)
    ground_troop_mass = sum(attack.troops.get(name, 0) for name in TROOP_GROUND_POWER)
    splash_vulnerable_mass = (
        attack.troops["barbarian"]
        + attack.troops["archer"]
        + attack.troops["wizard"]
        + attack.troops["balloon"]
        + attack.troops["minion"]
        + attack.troops["witch"] * 1.3
        + attack.spells["bat"] * 3.0
        + attack.spells["skeleton"] * 2.3
    )

    sustain = (
        support_power
        + attack.spells["heal"] * 2.1
        + attack.spells["rage"] * 0.8
        + attack.spells["freeze"] * 0.6
        + attack.pets["unicorn"] * 0.85
        + attack.guardians["healing_guardian"] * 0.95
        + attack.troops["healer"] * 1.3
        + attack.troops["druid"] * 1.0
    )
    burst = (
        air_power * 0.3
        + ground_power * 0.34
        + attack.spells["rage"] * 1.8
        + attack.spells["freeze"] * 1.4
        + attack.spells["clone"] * 1.2
        + attack.troops["headhunter"] * 0.7
    )
    hero_dive = (
        attack.heroes["barbarian_king"] * 0.045
        + attack.heroes["archer_queen"] * 0.05
        + attack.heroes["royal_champion"] * 0.08
        + attack.heroes["grand_warden"] * 0.04
        + attack.heroes["dragon_duke"] * 0.05
        + attack.heroes["minion_prince"] * 0.04
    )
    pathing = (
        siege_breach
        + wall_breakers
        + jump_pressure
        + attack.troops["root_rider"] * 0.65
        + attack.troops["hog_rider"] * 0.22
        + attack.troops["miner"] * 0.24
    )
    trap_resilience = (
        sustain * 0.35
        + attack.troops["ice_golem"] * 0.9
        + attack.troops["golem"] * 0.8
        + attack.troops["yeti"] * 0.45
        + attack.spells["heal"] * 1.1
        + attack.spells["freeze"] * 0.45
    )
    air_ratio = air_troop_mass / max(1.0, air_troop_mass + ground_troop_mass)
    ground_ratio = ground_troop_mass / max(1.0, air_troop_mass + ground_troop_mass)

    return {
        "air_power": air_power,
        "ground_power": ground_power,
        "support_power": support_power,
        "sustain": sustain,
        "burst": burst,
        "hero_dive": hero_dive,
        "pathing": pathing,
        "trap_resilience": trap_resilience,
        "splash_vulnerable_mass": splash_vulnerable_mass,
        "air_ratio": air_ratio,
        "ground_ratio": ground_ratio,
    }


def battle_score(attack: AttackConfig, base: BaseConfig, rng: random.Random, include_noise: bool = False) -> float:
    profile = attack_phase_profile(attack)
    air_power = profile["air_power"]
    ground_power = profile["ground_power"]
    support_power = profile["support_power"]

    attack_total = air_power + ground_power + support_power
    defense_total = (
        base.anti_air_defense * 1.05
        + base.splash_defense * 0.95
        + base.wall_strength * 1.15
        + base.inferno_strength * 1.2
        + base.trap_pressure * 0.9
        + base.ground_pressure * 0.24
        + base.base_level * 6.0
    )

    score = 46.0
    score += attack_total * 0.34
    score -= defense_total * 0.58

    anti_air_gap = air_power - (base.anti_air_defense * 0.84)
    wall_break_gap = profile["pathing"] - (base.wall_strength * 0.36)
    sustain_gap = profile["sustain"] - (base.splash_defense * 0.4 + base.trap_pressure * 0.28)
    core_gap = profile["burst"] + profile["hero_dive"] - (base.inferno_strength * 0.5 + base.base_level * 0.24)
    trap_gap = profile["trap_resilience"] - base.trap_pressure * 0.42

    score += anti_air_gap * (0.18 + profile["air_ratio"] * 0.08)
    score += wall_break_gap * (0.22 + profile["ground_ratio"] * 0.04)
    score += sustain_gap * 0.16
    score += core_gap * 0.14
    score += trap_gap * 0.12

    if profile["air_ratio"] > 0.62 and base.defenses.get("air_sweeper", 0) >= 7:
        score -= (base.defenses["air_sweeper"] - 6) * 1.1
    if profile["ground_ratio"] > 0.58 and base.defenses.get("wall", 0) >= 7 and wall_break_gap < 0:
        score -= (base.defenses["wall"] - 6) * 0.9
    if profile["splash_vulnerable_mass"] > 24 and base.splash_defense >= 55:
        score -= (profile["splash_vulnerable_mass"] - 24) * 0.16
    if profile["hero_dive"] > 12 and attack.spells["invisibility"] > 0:
        score += 1.8
    if profile["air_ratio"] > 0.55 and attack.spells["freeze"] + attack.spells["haste"] >= 3:
        score += 1.4
    if profile["ground_ratio"] > 0.55 and attack.spells["jump"] + attack.spells["earthquake"] >= 2:
        score += 1.3
    if attack.troops["headhunter"] >= 2 and base.defenses.get("monolith", 0) >= 6:
        score += 1.0

    if attack.siege_machine == "wall_wrecker":
        score += max(0.0, 18.0 - base.wall_strength) * 1.1
    elif attack.siege_machine == "log_launcher":
        score += max(0.0, 20.0 - base.wall_strength - base.inferno_strength * 0.25) * 0.9
    elif attack.siege_machine == "battle_blimp":
        score += air_power * 0.22 - base.anti_air_defense * 0.2
    elif attack.siege_machine == "stone_slammer":
        score += air_power * 0.25 - base.anti_air_defense * 0.18
    elif attack.siege_machine == "siege_barracks":
        score += ground_power * 0.15 + support_power * 0.08
    elif attack.siege_machine == "flame_flinger":
        score += support_power * 0.12 + max(0.0, 14.0 - base.ground_pressure * 0.08)
    elif attack.siege_machine == "battle_drill":
        score += ground_power * 0.16 + max(0.0, 14.0 - base.wall_strength * 0.3)

    if attack.clan_castle == "cc_balloon":
        score += air_power * 0.12 - base.anti_air_defense * 0.12
    elif attack.clan_castle == "cc_dragon":
        score += air_power * 0.13 - base.anti_air_defense * 0.1
    elif attack.clan_castle == "cc_yeti":
        score += ground_power * 0.12
    elif attack.clan_castle == "cc_ice_golem":
        score += support_power * 0.1 + attack.spells["freeze"] * 0.8

    if include_noise:
        score += rng.uniform(-8.0, 8.0)
    # Centering at 50.0 and scaling by 11.0 keeps the synthetic probabilities
    # spread across a useful midrange instead of collapsing toward 0 or 1.
    probability = 1.0 / (1.0 + math.exp(-(score - 50.0) / 11.0))
    return max(0.0, min(1.0, probability))


def flatten_attack_config(attack: AttackConfig) -> Dict[str, int | str]:
    row: Dict[str, int | str] = {}
    for troop in TROOP_TYPES:
        row[f"troop_{troop}"] = attack.troops[troop]
    for spell in SPELL_TYPES:
        row[f"spell_{spell}"] = attack.spells[spell]
    for hero in HERO_TYPES:
        row[f"hero_{hero}"] = attack.heroes[hero]
    for pet in PET_TYPES:
        row[f"pet_{pet}"] = attack.pets[pet]
    for guardian in GUARDIAN_TYPES:
        row[f"guardian_{guardian}"] = attack.guardians[guardian]
    row["clan_castle"] = attack.clan_castle
    row["siege_machine"] = attack.siege_machine
    return row


def flatten_base_config(base: BaseConfig) -> Dict[str, int | float]:
    row: Dict[str, int | float] = {
        "base_level": base.base_level,
        "anti_air_defense": base.anti_air_defense,
        "ground_pressure": base.ground_pressure,
        "splash_defense": base.splash_defense,
        "wall_strength": base.wall_strength,
        "inferno_strength": base.inferno_strength,
        "trap_pressure": base.trap_pressure,
    }
    for defense in DEFENSE_TYPES:
        row[f"defense_{defense}"] = base.defenses.get(defense, 0)
    return row


def generate_dataset(num_samples: int = DEFAULT_DATASET_SAMPLES, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    rows: List[Dict[str, int | float | str]] = []
    for _ in range(num_samples):
        attack = random_attack_config(rng)
        base = random_base_config(rng)
        row: Dict[str, int | float | str] = {}
        row.update(flatten_attack_config(attack))
        row.update(flatten_base_config(base))
        row["win_probability"] = battle_score(attack, base, rng, include_noise=False)
        rows.append(row)
    return pd.DataFrame(rows)


def candidate_attacks() -> List[AttackConfig]:
    hero_defaults = {
        "barbarian_king": 88,
        "archer_queen": 90,
        "minion_prince": 60,
        "grand_warden": 65,
        "royal_champion": 38,
        "dragon_duke": 35,
    }
    pet_defaults = {"lassi": 8, "electro_owl": 8, "mighty_yak": 7, "unicorn": 9}
    guardian_defaults = {"ground_guardian": 7, "air_guardian": 7, "healing_guardian": 8}

    troop_options = [
        enforce_army_capacity(merge_setup(TROOP_TYPES, {"electro_dragon": 8, "dragon_rider": 6, "balloon": 10, "baby_dragon": 4, "minion": 8})),
        enforce_army_capacity(merge_setup(TROOP_TYPES, {"yeti": 8, "electro_titan": 4, "root_rider": 6, "wizard": 8, "healer": 4, "apprentice_warden": 2, "wall_breaker": 6})),
        enforce_army_capacity(merge_setup(TROOP_TYPES, {"miner": 14, "hog_rider": 12, "healer": 4, "wizard": 6, "headhunter": 3, "ice_golem": 2})),
        enforce_army_capacity(merge_setup(TROOP_TYPES, {"pekka": 5, "bowler": 10, "witch": 6, "golem": 2, "ice_golem": 2, "wall_breaker": 6})),
        enforce_army_capacity(merge_setup(TROOP_TYPES, {"dragon": 8, "balloon": 12, "lava_hound": 2, "minion": 10, "baby_dragon": 3})),
        enforce_army_capacity(merge_setup(TROOP_TYPES, {"root_rider": 8, "valkyrie": 8, "wizard": 8, "healer": 3, "apprentice_warden": 2})),
        enforce_army_capacity(merge_setup(TROOP_TYPES, {"golem": 3, "witch": 10, "bowler": 8, "ice_golem": 2, "wall_breaker": 6})),
        enforce_army_capacity(merge_setup(TROOP_TYPES, {"miner": 12, "hog_rider": 10, "headhunter": 4, "healer": 4, "apprentice_warden": 2})),
    ]

    spell_options = [
        merge_setup(SPELL_TYPES, {"rage": 3, "freeze": 3, "clone": 1, "haste": 1}),
        merge_setup(SPELL_TYPES, {"heal": 3, "rage": 2, "freeze": 1, "jump": 1}),
        merge_setup(SPELL_TYPES, {"invisibility": 2, "freeze": 2, "rage": 2, "recall": 1}),
        merge_setup(SPELL_TYPES, {"lightning": 3, "freeze": 2, "rage": 1, "earthquake": 1}),
        merge_setup(SPELL_TYPES, {"heal": 2, "jump": 2, "rage": 2, "poison": 1}),
    ]

    attacks: List[AttackConfig] = []
    for troops, spells, clan_castle, siege_machine in product(
        troop_options,
        spell_options,
        CLAN_CASTLE_TYPES,
        SIEGE_TYPES,
    ):
        attacks.append(
            AttackConfig(
                troops=troops,
                spells=spells,
                heroes=hero_defaults,
                pets=pet_defaults,
                guardians=guardian_defaults,
                clan_castle=clan_castle,
                siege_machine=siege_machine,
            )
        )
    return attacks


def mutate_attack_config(attack: AttackConfig, variant_index: int) -> AttackConfig:
    troops = dict(attack.troops)
    spells = dict(attack.spells)
    pets = dict(attack.pets)
    guardians = dict(attack.guardians)

    troop_patterns = [
        [("wall_breaker", 2), ("freeze", 1), ("electro_dragon", -1), ("dragon_rider", 1)],
        [("root_rider", 1), ("jump", 1), ("heal", -1), ("wizard", 2)],
        [("hog_rider", 2), ("freeze", 1), ("miner", -2), ("headhunter", 1)],
        [("balloon", 2), ("haste", 1), ("minion", 2), ("clone", 1)],
        [("witch", 2), ("ice_golem", 1), ("poison", 1), ("rage", -1)],
        [("yeti", 1), ("earthquake", 1), ("wall_breaker", 2), ("heal", -1)],
    ]
    pattern = troop_patterns[variant_index % len(troop_patterns)]
    for name, delta in pattern:
        if name in troops:
            troops[name] = max(0, min(TROOP_MAX_COUNTS[name], troops[name] + delta))
        elif name in spells:
            spells[name] = max(0, min(SPELL_MAX_COUNTS[name], spells[name] + delta))
    troops = enforce_army_capacity(troops)

    pet_cycle = ["electro_owl", "mighty_yak", "unicorn", "lassi"]
    guardian_cycle = ["air_guardian", "ground_guardian", "healing_guardian"]
    pets[pet_cycle[variant_index % len(pet_cycle)]] = min(10, pets[pet_cycle[variant_index % len(pet_cycle)]] + 1)
    guardians[guardian_cycle[variant_index % len(guardian_cycle)]] = min(
        10,
        guardians[guardian_cycle[variant_index % len(guardian_cycle)]] + 1,
    )

    return AttackConfig(
        troops=troops,
        spells=spells,
        heroes=dict(attack.heroes),
        pets=pets,
        guardians=guardians,
        clan_castle=attack.clan_castle,
        siege_machine=attack.siege_machine,
    )
