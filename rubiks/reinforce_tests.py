import unittest
from reconstruction import CubeReconstructor
from reinforce import CubieEnforcer

class TestCubieEnforcer(unittest.TestCase):
    def setUp(self):
        # Fully solved cube input
        self.input_faces = [
            ("W", "WWWWWWWWW"),
            ("Y", "YYYYYYYYY"),
            ("B", "BBBBBBBBB"),
            ("G", "GGGGGGGGG"),
            ("R", "RRRRRRRRR"),
            ("O", "OOOOOOOOO")
        ]

        self.reconstructor = CubeReconstructor(self.input_faces)
        self.assertTrue(self.reconstructor.reconstruct())  # Should succeed

    def test_single_corner_enforcement(self):
        constraints = [("WRB", "WRB")]  # This should be already valid in solved cube
        enforcer = CubieEnforcer(self.reconstructor, constraints)
        self.assertTrue(enforcer.enforce_constraints())

        final_state = enforcer.get_final_state()
        self.assertIsNotNone(final_state)
        self.assertEqual(final_state["W"][6], "W")
        self.assertEqual(final_state["R"][2], "R")
        self.assertEqual(final_state["B"][0], "B")

    def test_invalid_corner(self):
        constraints = [("WRG", "YYY")]  # Invalid constraint (Y not part of WRG)
        enforcer = CubieEnforcer(self.reconstructor, constraints)
        self.assertFalse(enforcer.enforce_constraints())

    def test_multiple_constraints(self):
        constraints = [("WRB", "WRB"), ("YBO", "YBO"), ("RG", "RG")]
        enforcer = CubieEnforcer(self.reconstructor, constraints)
        self.assertTrue(enforcer.enforce_constraints())

        final_state = enforcer.get_final_state()
        self.assertIsNotNone(final_state)
        self.assertEqual(final_state["W"][6], "W")
        self.assertEqual(final_state["B"][0], "B")
        self.assertEqual(final_state["R"][2], "R")

    def test_conflicting_constraints(self):
        constraints = [("WRB", "WRB"), ("WRB", "YRG")]  # Cannot place same cubie in two spots
        enforcer = CubieEnforcer(self.reconstructor, constraints)
        self.assertFalse(enforcer.enforce_constraints())

if __name__ == "__main__":
    unittest.main()
