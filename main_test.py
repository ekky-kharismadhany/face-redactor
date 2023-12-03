import unittest

from main import app

class TestMain(unittest.TestCase):
    def setUp(self):
        app.testing = True
        self.app = app.test_client()

    def test_index(self):
        response = self.app.get("/")
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()
