import unittest

from battle_simulator import DEFENSE_TYPES, candidate_attacks, generate_dataset
from battle_simulator import BaseConfig
from recommend_strategy import load_metrics, recommend_for_base


class MLPipelineTestCase(unittest.TestCase):
    def test_generated_dataset_has_expected_schema(self):
        df = generate_dataset(num_samples=25, seed=11)
        self.assertEqual(len(df), 25)
        self.assertIn("win_probability", df.columns)
        self.assertIn("troop_barbarian", df.columns)
        self.assertIn("spell_rage", df.columns)
        self.assertIn("hero_barbarian_king", df.columns)
        self.assertIn("defense_cannon", df.columns)
        self.assertIn("defense_monolith", df.columns)

    def test_candidate_attacks_exist(self):
        candidates = candidate_attacks()
        self.assertGreater(len(candidates), 0)
        self.assertTrue(any(candidate.siege_machine for candidate in candidates))

    def test_recommendation_returns_ranked_predictions(self):
        base = BaseConfig(
            base_level=15,
            anti_air_defense=10.0,
            splash_defense=10.0,
            wall_strength=10.0,
            inferno_strength=10.0,
            trap_pressure=10.0,
            defenses={name: 5 for name in DEFENSE_TYPES},
        )
        ranked = recommend_for_base(base)
        self.assertIn("predicted_win_probability", ranked.columns)
        self.assertGreater(len(ranked), 0)

    def test_metrics_are_available(self):
        metrics = load_metrics()
        self.assertIn("mae", metrics)
        self.assertIn("rmse", metrics)
        self.assertIn("r2", metrics)


if __name__ == "__main__":
    unittest.main()
