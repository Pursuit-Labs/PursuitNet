import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import pursuitnet as pn
import time

class TestRandomProcesses(unittest.TestCase):

    def test_random_number_generation(self):
        print("Testing seeding consistency:")
        pn.random.seed(42)
        first_run = pn.random.print_random_number()
        second_run = pn.random.print_random_number()

        pn.random.seed(42)
        third_run = pn.random.print_random_number()
        fourth_run = pn.random.print_random_number()

        self.assertEqual(first_run, third_run, "Seeding does not produce consistent results in random.")
        self.assertEqual(second_run, fourth_run, "Seeding does not produce consistent results in random.")

    def test_module_consistency(self):
        print("Testing module consistency with seeding:")
        seed_value = int(time.time())
        pn.random.seed(seed_value)
        random_number = pn.random.print_random_number()
        nn_number = pn.nn.print_random_number()
        optim_number = pn.optim.print_random_number()

        pn.random.seed(seed_value)
        self.assertEqual(random_number, pn.random.print_random_number(), "Random module seed consistency failed.")
        self.assertEqual(nn_number, pn.nn.print_random_number(), "NN module seed consistency failed.")
        self.assertEqual(optim_number, pn.optim.print_random_number(), "Optim module seed consistency failed.")

    def test_without_seeding(self):
        print("Testing behavior without explicit seeding:")
        initial_random = pn.random.print_random_number()
        initial_nn = pn.nn.print_random_number()
        initial_optim = pn.optim.print_random_number()

        new_random = pn.random.print_random_number()
        new_nn = pn.nn.print_random_number()
        new_optim = pn.optim.print_random_number()

        self.assertNotEqual(initial_random, new_random, "Random module should produce different values on subsequent runs without reseeding.")
        self.assertNotEqual(initial_nn, new_nn, "NN module should produce different values on subsequent runs without reseeding.")
        self.assertNotEqual(initial_optim, new_optim, "Optim module should produce different values on subsequent runs without reseeding.")

if __name__ == "__main__":
    unittest.main()
