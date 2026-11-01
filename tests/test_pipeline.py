import unittest
from src.preprocessing import clean_text
from src.priority import assign_priority


class TextCleaningTests(unittest.TestCase):
    def test_normalizes_case_and_urls(self):
        self.assertEqual(clean_text("VPN ERROR https://example.com"), "vpn error")

    def test_removes_redacted_placeholders(self):
        self.assertNotIn("name", clean_text("Hello [NAME], contact [EMAIL]"))


class PriorityTests(unittest.TestCase):
    def test_outage_is_high(self):
        self.assertEqual(assign_priority("Production server down for all users")["priority"], "High")

    def test_information_request_is_low(self):
        self.assertEqual(assign_priority("How do I find the user guide?")["priority"], "Low")

    def test_ambiguous_ticket_defaults_to_medium(self):
        self.assertEqual(assign_priority("Please check this ticket")["priority"], "Medium")


if __name__ == "__main__":
    unittest.main()
