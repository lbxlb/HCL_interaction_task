"""Boundary checks for task events; no MuJoCo dependency."""
import unittest

from button_press_state import ButtonPressState, PressCriteria


class PressStateTests(unittest.TestCase):
    def test_spike_and_proximity_do_not_press(self):
        state = ButtonPressState()
        state.update(0, 0.0002, True)
        state.update(0.01, 0, False)
        for t in (0.02, 0.04, 0.06):
            state.update(t, 0.0002, False)
        self.assertEqual(state.press_count, 0)

    def test_hysteresis_release_debounce_and_rearming(self):
        state = ButtonPressState()
        state.update(0, 0.0002, True)
        self.assertEqual(state.update(0.02, 0.0002, True)['event'], 'pressed')
        for t, q, contact in ((0.03, 0.0001, True), (0.04, 0.0002, True),
                              (0.05, 0, False), (0.06, 0.0001, False),
                              (0.08, 0, True)):
            self.assertIsNone(state.update(t, q, contact)['event'])
        state.update(0.10, 0, False)
        self.assertEqual(state.update(0.12, 0, False)['event'], 'released')
        state.update(0.14, 0.0002, True)
        self.assertEqual(state.update(0.16, 0.0002, True)['event'], 'pressed')
        self.assertEqual((state.press_count, state.release_count), (2, 1))
        state.reset()
        self.assertEqual(state.update(0, 0, False)['press_count'], 0)

    def test_invalid_input_and_configuration(self):
        for kwargs in ({'release_position_m': 0.0002}, {'press_hold_s': 0},
                       {'release_hold_s': float('nan')}):
            with self.assertRaises(ValueError):
                PressCriteria(**kwargs)
        state = ButtonPressState()
        state.update(0, 0, False)
        for t, q, contact in ((0, 0, False), (-1, 0, False),
                              (1, float('nan'), False), (1, 0, 1)):
            with self.assertRaises(ValueError):
                state.update(t, q, contact)
        self.assertEqual(state.update(0.002, 0, False)['phase'], 'ready')


if __name__ == '__main__':
    unittest.main()
