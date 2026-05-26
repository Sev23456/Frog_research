"""
Basal Ganglia action selection and movement gating.

Models:
- Direct pathway (premotor cortex в†’ striatum в†’ SNr в†’ thalamus): FACILITATES movement
- Indirect pathway (premotor cortex в†’ striatum в†’ GPe в†’ STN в†’ SNr): INHIBITS movement
- Dopamine modulation: enhances direct, suppresses indirect
- Thalamic relay: gates motor commands based on winner
"""

import numpy as np
from Frog_predator_neuro_dual_fast.core.biological_neuron import LIFNeuron, PyramidalNeuron, FastSpikingInterneuron


class BasalGanglia:
    """
    Action selection circuit using archetypal basal ganglia architecture.
    
    Enables smooth decision-making by:
    1. Requiring confidence threshold before execution
    2. Arbitrating between competing motives
    3. Providing hysteresis via recurrent connections
    """
    
    def __init__(self):
        # Medium Spiny Neurons (striatum): dopamine-sensitive
        self.msn_direct = [PyramidalNeuron() for _ in range(8)]      # D1 receptors (Go, excited by DA)
        self.msn_indirect = [FastSpikingInterneuron() for _ in range(8)]  # D2 receptors (Stop, inhibited by DA)
        
        # Subthalamic nucleus (STN): glutamatergic excitation
        self.stn = [PyramidalNeuron() for _ in range(4)]
        
        # Substantia nigra pars reticulata (SNr): GABAergic, tonically active
        self.snr = [FastSpikingInterneuron() for _ in range(4)]
        
        # Thalamic relay: pass signals if SNr inhibition is low
        self.thalamic_relay = [PyramidalNeuron() for _ in range(4)]
        
        # State
        self.dopamine_level = 0.5  # Will be updated from brain
        self.last_action_signal = 0.0  # For hysteresis
        self.decision_confidence = 0.0  # How much we're committing to action
        self.time_since_decision = 0.0  # Track decision stability
        
    def select_action(self, cortical_input, dopamine_level, can_inhibit=True):
        """
        Arbitrate between competing drives and gate motor output.
        
        Args:
            cortical_input: np.array of shape (8,) with competing action signals (0-1)
            dopamine_level: float (0-1), affects D1/D2 balance
            can_inhibit: if False, always allow movement (emergency override)
        
        Returns:
            dict with:
                'gating_signal': float (0-1), 0 = blocked, 1 = execute
                'confidence': float (0-1), how committed to action
                'direct_activity': float, direct pathway output
                'indirect_activity': float, indirect pathway output
        """
        self.dopamine_level = dopamine_level
        
        # Direct pathway: D1 neurons EXCITED by dopamine
        # Cortex в†’ Striatum (D1) в†’ SNr (inhibit) в†’ Thalamus (disinhibit)
        direct_gain = 1.0 + dopamine_level * 1.5  # DA enhances direct path
        for idx, neuron in enumerate(self.msn_direct):
            input_signal = cortical_input[idx] if idx < len(cortical_input) else 0.0
            neuron.integrate(1.0, input_signal * direct_gain * 40.0, dopamine_level)
        
        direct_activity = float(np.mean([n.activity_level() for n in self.msn_direct]))
        
        # Indirect pathway: D2 neurons INHIBITED by dopamine
        # Cortex в†’ Striatum (D2) в†’ GPe (inhibit) в†’ STN (excite) в†’ SNr (inhibit thalamus more)
        indirect_gain = 0.4 * (1.0 - dopamine_level * 0.8)  # DA suppresses indirect path
        for idx, neuron in enumerate(self.msn_indirect):
            input_signal = cortical_input[idx] if idx < len(cortical_input) else 0.0
            neuron.integrate(1.0, max(0.0, input_signal * indirect_gain * 20.0))
        
        indirect_activity = float(np.mean([n.activity_level() for n in self.msn_indirect]))
        
        # STN: glutamatergic, drives SNr
        for neuron in self.stn:
            # Receives indirect pathway input and cortical feedback
            stn_input = indirect_activity * 30.0 + (1.0 - direct_activity) * 10.0
            neuron.integrate(1.0, stn_input, 0.8)
        
        stn_activity = float(np.mean([n.activity_level() for n in self.stn]))
        
        # SNr: tonically active, inhibits thalamus and superior colliculus
        # Less inhibition when direct pathway active, more when indirect active
        basal_inhibition = 20.0 + stn_activity * 25.0 - direct_activity * 50.0
        for neuron in self.snr:
            neuron.integrate(1.0, basal_inhibition)
        
        snr_activity = float(np.mean([n.activity_level() for n in self.snr]))
        
        # Thalamic relay: disinhibited by reduced SNr activity
        # Acts as a gate: if SNr is quiet, thalamus relays motor commands
        thalamic_input = max(0.0, 40.0 - snr_activity * 8.0)  # Inverse relationship
        for neuron in self.thalamic_relay:
            neuron.integrate(1.0, thalamic_input, 0.85)
        
        thalamic_activity = float(np.mean([n.activity_level() for n in self.thalamic_relay]))
        
        # Compute confidence: do we have strong direct > indirect?
        net_drive = direct_activity - indirect_activity * 0.8
        competition = abs(direct_activity - indirect_activity)
        
        # Hysteresis: if we were already committing, require less signal to continue
        commitment_bonus = self.last_action_signal * 0.30  # Original balanced value
        adjusted_drive = net_drive + commitment_bonus
        
        # Confidence decreases when pathways compete
        confidence = max(0.0, np.tanh(adjusted_drive * 2.0) - competition * 0.5)
        
        # Only gate if confident AND can_inhibit
        if can_inhibit:
            gating_signal = confidence * thalamic_activity / (40.0 + 1e-6)
        else:
            gating_signal = 0.9 + confidence * 0.1  # Always high confidence override
        
        # Apply hysteresis: smooth gating decisions  
        alpha = 0.20  # 20% new decision, 80% memory (balanced)
        gating_signal = alpha * gating_signal + (1.0 - alpha) * self.last_action_signal
        
        self.last_action_signal = gating_signal
        self.decision_confidence = float(confidence)
        self.time_since_decision += 1.0
        
        return {
            'gating_signal': float(np.clip(gating_signal, 0.0, 1.0)),
            'confidence': float(np.clip(confidence, 0.0, 1.0)),
            'direct_activity': float(direct_activity),
            'indirect_activity': float(indirect_activity),
            'net_drive': float(net_drive),
            'snr_inhibition': float(snr_activity),
            'thalamic_relay': float(thalamic_activity),
            'is_halted': float(gating_signal) < 0.25,
            'is_committed': float(gating_signal) > 0.75,
        }
    
    def reset_decision(self):
        """Reset decision state (e.g., when reward received)."""
        self.last_action_signal = 0.0
        self.time_since_decision = 0.0
        self.decision_confidence = 0.0
    
    def all_neurons(self):
        """Return all neurons for caching."""
        return (
            self.msn_direct + self.msn_indirect +
            self.stn + self.snr + self.thalamic_relay
        )


