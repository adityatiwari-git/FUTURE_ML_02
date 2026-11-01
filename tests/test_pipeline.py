import unittest
import pandas as pd
from src.train import train_and_evaluate
from src.preprocessing import clean_text
from src.priority import assign_priority


class TextCleaningTests(unittest.TestCase):
    def test_normalizes_case_and_urls(self):
        self.assertEqual(clean_text("VPN ERROR https://example.com"), "vpn error")

    def test_removes_redacted_placeholders(self):
        self.assertNotIn("name", clean_text("Hello [NAME], contact [EMAIL]"))


class ModelTrainingTests(unittest.TestCase):
    def test_compares_three_classifiers_on_a_holdout(self):
        train = pd.DataFrame({
            "text": ["vpn access error", "reset password account", "install software app",
                     "email service broken", "shared folder permission", "server outage critical"] * 3,
            "category": ["Access", "Account", "Software", "Email", "Files", "Outage"] * 3,
        })
        test = pd.DataFrame({
            "text": ["cannot access vpn", "password reset issue", "install new software",
                     "email is not working", "folder access permission", "production server down"],
            "category": ["Access", "Account", "Software", "Email", "Files", "Outage"],
        })
        results, best_name, model, actual, predicted = train_and_evaluate(train, test)
        self.assertEqual(len(results), 3)
        self.assertIn(best_name, results)
        self.assertEqual(len(actual), len(predicted))


class PriorityTests(unittest.TestCase):
    def test_outage_is_high(self):
        self.assertEqual(assign_priority("Production server down for all users")["priority"], "High")

    def test_is_down_outage_is_high(self):
        self.assertEqual(assign_priority("Production server is down")["priority"], "High")

    def test_information_request_is_low(self):
        self.assertEqual(assign_priority("How do I find the user guide?")["priority"], "Low")

    def test_ambiguous_ticket_defaults_to_medium(self):
        self.assertEqual(assign_priority("Please check this ticket")["priority"], "Medium")


if __name__ == "__main__":
    unittest.main()
