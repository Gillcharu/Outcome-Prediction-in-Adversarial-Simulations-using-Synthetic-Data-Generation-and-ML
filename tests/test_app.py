import unittest

from app import app


class AppRoutesTestCase(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_home_page_loads(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Predict the best attack before the battle starts.", response.data)

    def test_api_predict_returns_analysis(self):
        payload = {
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
            "trap_pressure": 5,
        }
        response = self.client.post("/api/predict", json=payload)
        body = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertIn("prediction", body)
        self.assertIn("recommendation", body)
        self.assertIn("top_recommendations", body)
        self.assertIn("chart_data", body)


if __name__ == "__main__":
    unittest.main()
