import unittest
from package.module import example_function

class TestExample(unittest.TestCase):
    def test_example(self):
        self.assertEqual(example_function(), 42)

if __name__ == "__main__":
    unittest.main()
