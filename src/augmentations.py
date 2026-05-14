import numpy as np
import random


def apply_temporal_jitter(signal, max_shift=50):
    """Shifts the signal dynamically along the time axis"""
    shift_amount = random.randint(-max_shift, max_shift)
    if shift_amount == 0:
        return signal

    jittered_signal = np.zeros_like(signal)
    if shift_amount > 0:
        jittered_signal[shift_amount:, :] = signal[:-shift_amount, :]
    else:
        shift_amount = abs(shift_amount)
        jittered_signal[:-shift_amount, :] = signal[shift_amount:, :]

    return jittered_signal


def apply_random_lead_masking(signal, mask_prob=0.5):
    """Zeroes out 1 to 2 random leads"""
    if random.random() < mask_prob:
        num_leads_to_mask = random.choice([1, 2])
        lead_to_mask = random.sample(range(12), num_leads_to_mask)

        for lead in lead_to_mask:
            signal[:, lead] = 0.0

    return signal
