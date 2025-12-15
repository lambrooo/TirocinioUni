import numpy as np
import torch

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

class ActiveInferenceAgent:
    def __init__(self, efe_mode='full'):
        """
        Initializes the Robust Active Inference Agent.
        
        Args:
            efe_mode: EFE calculation mode. Options:
                - 'full': Use both epistemic (entropy) and pragmatic (utility) terms
                - 'epistemic_only': Use only epistemic term (minimize entropy)
                - 'pragmatic_only': Use only pragmatic term (maximize utility)
        
        Generative Model Structure:
        
        Factors (Hidden States):
        0. Temperature (3): Low, Optimal, High
        1. Motor Temp (2): Safe, Overheating
        2. Load (2): Low, High
        3. Temp Sensor Health (2): Reliable, Spoofed
        4. Motor Sensor Health (2): Reliable, Spoofed
        
        Modalities (Observations):
        0. Obs Temp (3): Low, Opt, High
        1. Obs Motor (2): Safe, Overheat
        2. Obs Load (2): Low, High
        3. Obs Cyber (2): Safe, Alert (from external defense)
        
        Controls (Actions):
        0. Thermal (3): Cool, Maintain, Heat
        1. Load (3): Decrease, Maintain, Increase
        2. Verify (3): Wait, Verify Temp, Verify Motor
        """
        # --- EFE Mode ---
        valid_modes = ['full', 'epistemic_only', 'pragmatic_only']
        if efe_mode not in valid_modes:
            raise ValueError(f"efe_mode must be one of {valid_modes}, got '{efe_mode}'")
        self.efe_mode = efe_mode
        # --- Dimensions ---
        self.num_factors = 5
        self.num_states = [3, 2, 2, 2, 2]
        
        self.num_modalities = 4
        self.num_obs = [3, 2, 2, 2]

        self.num_controls = [3, 3, 3] # Thermal, Load, Verify

        # --- Matrices ---
        self.A = self.initialize_A_matrix() # Likelihood P(o|s)
        self.B = self.initialize_B_matrix() # Transition P(s'|s,a)
        self.C = self.initialize_C_matrix() # Preferences P(o)
        self.D = self.initialize_D_matrix() # Prior P(s)

        # --- Internal State ---
        self.qs = [d.copy() for d in self.D]
        self.prev_action = [1, 1, 0] # Initial actions (Maintain, Maintain, Wait)

    def initialize_A_matrix(self):
        """
        Likelihood mapping: P(Observation | State)
        
        Crucial for Robustness:
        - If Sensor Health is Reliable: Obs ~ State (Identity)
        - If Sensor Health is Spoofed: Obs ~ Fixed Value (e.g., Low)
        """
        A = []
        
        # --- Modality 0: Obs Temperature ---
        # Depends on Factor 0 (Temp) and Factor 3 (Temp Sensor Health)
        # We need a tensor mapping A[0][Obs, Temp, TempHealth]
        # But pymdp usually assumes A[modality] is a tensor over ALL factors or specific ones.
        # For simplicity in this custom implementation, we'll assume A is factorized or we handle the logic manually?
        # No, standard Active Inference uses tensors.
        # Let's construct the full tensor for this modality relative to its relevant factors.
        # To keep it simple without a library, we will assume independence where possible, 
        # but here dependence is key.
        
        # We will implement A as a list of matrices/tensors.
        # Since we are writing the inference engine manually (infer_state), we can define how A is structured.
        # Let's define A[modality] as a function of relevant states for clarity in this custom code,
        # OR we stick to the standard: A[modality] has shape [Num_Obs, Num_States_F1, Num_States_F2...]
        # Let's use the standard tensor approach for the relevant factors.
        
        # Modality 0 (Obs Temp) depends on Factor 0 (Temp) and Factor 3 (Temp Health)
        # Shape: [3 (Obs), 3 (Temp), 2 (Health)]
        A_temp = np.zeros((3, 3, 2))
        
        # Case 1: Reliable (Health=0) -> Identity
        A_temp[:, :, 0] = np.eye(3) * 0.9 + 0.033
        
        # Case 2: Spoofed (Health=1) -> Biased to Low (Index 0) to induce heating
        # If spoofed, we observe Low (0) regardless of true temp
        A_temp[:, :, 1] = np.array([
            [0.9, 0.9, 0.9], # High prob of observing Low
            [0.05, 0.05, 0.05],
            [0.05, 0.05, 0.05]
        ])
        A.append(A_temp)

        # --- Modality 1: Obs Motor ---
        # Depends on Factor 1 (Motor) and Factor 4 (Motor Health)
        # Shape: [2 (Obs), 2 (Motor), 2 (Health)]
        A_motor = np.zeros((2, 2, 2))
        
        # Case 1: Reliable (Health=0) -> Identity
        A_motor[:, :, 0] = np.eye(2) * 0.9 + 0.05
        
        # Case 2: Spoofed (Health=1) -> Biased to Safe (Index 0) to hide overheating
        A_motor[:, :, 1] = np.array([
            [0.9, 0.9], # Always observe Safe
            [0.1, 0.1]
        ])
        A.append(A_motor)

        # --- Modality 2: Obs Load ---
        # Depends on Factor 2 (Load) - Assume reliable for now (or add Load Health later)
        # Shape: [2 (Obs), 2 (Load)]
        A_load = np.eye(2) * 0.95 + 0.025
        A.append(A_load)

        # --- Modality 3: Obs Cyber ---
        # Depends on Factor 3 (Temp Health) and Factor 4 (Motor Health)
        # Shape: [2 (Obs), 2 (TempH), 2 (MotorH)]
        # Alert (1) if ANY is Spoofed (1)
        A_cyber = np.zeros((2, 2, 2))
        
        for th in range(2):
            for mh in range(2):
                if th == 1 or mh == 1:
                    # Attack present -> Alert
                    A_cyber[:, th, mh] = [0.1, 0.9]
                else:
                    # Safe -> Safe
                    A_cyber[:, th, mh] = [0.9, 0.1]
        A.append(A_cyber)

        return A

    def initialize_B_matrix(self):
        """
        Transition likelihood: P(Next State | Current State, Action)
        """
        B = []

        # --- Factor 0: Temperature ---
        # Actions: 0=Cool, 1=Maintain, 2=Heat (Control 0)
        B_temp = np.zeros((3, 3, 3))
        # Cool: Shift distribution left (High->Opt, Opt->Low)
        B_temp[:, :, 0] = [[1.0, 0.8, 0.0], [0.0, 0.2, 0.9], [0.0, 0.0, 0.1]]
        # Maintain: Stay
        B_temp[:, :, 1] = [[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.0, 0.1, 0.9]]
        # Heat: Shift right (Low->Opt, Opt->High)
        B_temp[:, :, 2] = [[0.1, 0.0, 0.0], [0.9, 0.2, 0.0], [0.0, 0.8, 1.0]]
        B.append(B_temp)

        # --- Factor 1: Motor Temp ---
        # Actions: 0=Decrease Load, 1=Maintain, 2=Increase Load (Control 1)
        # Physics: High Load increases Motor Temp. Low Load decreases it.
        B_motor = np.zeros((2, 2, 3))
        # Decrease Load -> Cools down
        B_motor[:, :, 0] = [[1.0, 0.6], [0.0, 0.4]]
        # Maintain -> Stable
        B_motor[:, :, 1] = [[0.9, 0.1], [0.1, 0.9]]
        # Increase Load -> Heats up
        B_motor[:, :, 2] = [[0.7, 0.0], [0.3, 1.0]]
        B.append(B_motor)

        # --- Factor 2: Load ---
        # Actions: Control 1
        B_load = np.zeros((2, 2, 3))
        # Decrease
        B_load[:, :, 0] = [[1.0, 0.9], [0.0, 0.1]]
        # Maintain
        B_load[:, :, 1] = [[1.0, 0.0], [0.0, 1.0]]
        # Increase
        B_load[:, :, 2] = [[0.1, 0.0], [0.9, 1.0]]
        B.append(B_load)

        # --- Factor 3: Temp Sensor Health ---
        # Actions: Control 2 (Verify)
        # 0=Wait, 1=Verify Temp, 2=Verify Motor
        B_temp_h = np.zeros((2, 2, 3))
        # Wait/Verify Motor: State persists (with small chance of attack starting/ending)
        B_temp_h[:, :, 0] = [[0.99, 0.05], [0.01, 0.95]] # Sticky attacks
        B_temp_h[:, :, 2] = [[0.99, 0.05], [0.01, 0.95]]
        # Verify Temp: Resets to Reliable (Agent trusts verification clears doubt/fixes it)
        # Or rather, verification reveals truth. In this model, we assume verification RESTORES trust/fixes.
        B_temp_h[:, :, 1] = [[1.0, 1.0], [0.0, 0.0]] 
        B.append(B_temp_h)

        # --- Factor 4: Motor Sensor Health ---
        # Actions: Control 2 (Verify)
        B_motor_h = np.zeros((2, 2, 3))
        # Wait/Verify Temp
        B_motor_h[:, :, 0] = [[0.99, 0.05], [0.01, 0.95]]
        B_motor_h[:, :, 1] = [[0.99, 0.05], [0.01, 0.95]]
        # Verify Motor
        B_motor_h[:, :, 2] = [[1.0, 1.0], [0.0, 0.0]]
        B.append(B_motor_h)

        return B

    def initialize_C_matrix(self):
        """
        Preferences: P(Observation).
        """
        C = []
        # Temp: Prefer Optimal
        C.append(np.array([0.0, 4.0, 0.0]))
        # Motor: Prefer Safe
        C.append(np.array([4.0, -2.0]))
        # Load: Prefer High
        C.append(np.array([0.0, 2.0]))
        # Cyber: Prefer Safe
        C.append(np.array([2.0, -2.0]))
        return C

    def initialize_D_matrix(self):
        """
        Prior belief about initial states.
        """
        D = []
        D.append(np.array([0.1, 0.8, 0.1])) # Temp
        D.append(np.array([0.9, 0.1]))      # Motor
        D.append(np.array([0.8, 0.2]))      # Load
        D.append(np.array([0.9, 0.1]))      # Temp Health (Reliable)
        D.append(np.array([0.9, 0.1]))      # Motor Health (Reliable)
        return D

    def infer_state(self, observations):
        """
        Perception: Minimize VFE.
        Custom implementation for non-standard A matrix shapes.
        """
        new_qs = []
        
        # --- Factor 0: Temp ---
        # Depends on Obs Temp (Mod 0) via A[0] which links F0 and F3
        # We need to marginalize over F3 (Temp Health) to get likelihood for F0
        # Likelihood F0 = sum_F3 ( P(o|F0, F3) * qs[F3] )
        
        # Get Obs Indices
        o_temp, o_motor, o_load, o_cyber = observations

        # 1. Likelihoods
        # L_Temp(F0) = A[0][o_temp, :, :] . qs[3]
        L_temp_F0 = np.dot(self.A[0][o_temp, :, :], self.qs[3])
        
        # L_Motor(F1) = A[1][o_motor, :, :] . qs[4]
        L_motor_F1 = np.dot(self.A[1][o_motor, :, :], self.qs[4])
        
        # L_Load(F2) = A[2][o_load, :]
        L_load_F2 = self.A[2][o_load, :]
        
        # L_Health_Temp(F3)
        # Part 1: From Obs Temp -> A[0][o_temp, :, :].T . qs[0]
        L_HT_from_Temp = np.dot(self.A[0][o_temp, :, :].T, self.qs[0])
        # Part 2: From Obs Cyber -> A[3][o_cyber, :, :]. marginalize F4
        # A[3] shape: [2, 2(F3), 2(F4)]
        # sum_F4 ( A[3][o_cyber, :, F4] * qs[4][F4] )
        L_HT_from_Cyber = np.dot(self.A[3][o_cyber, :, :], self.qs[4])
        L_HT_F3 = L_HT_from_Temp * L_HT_from_Cyber

        # L_Health_Motor(F4)
        # Part 1: From Obs Motor
        L_HM_from_Motor = np.dot(self.A[1][o_motor, :, :].T, self.qs[1])
        # Part 2: From Obs Cyber
        # sum_F3 ( A[3][o_cyber, F3, :] * qs[3][F3] )
        L_HM_from_Cyber = np.dot(self.A[3][o_cyber, :, :].T, self.qs[3])
        L_HM_F4 = L_HM_from_Motor * L_HM_from_Cyber

        Likelihoods = [L_temp_F0, L_motor_F1, L_load_F2, L_HT_F3, L_HM_F4]

        # 2. Priors & Posterior
        for f in range(self.num_factors):
            # Determine control index for this factor
            if f == 0: c_idx = 0 # Thermal
            elif f in [1, 2]: c_idx = 1 # Load
            else: c_idx = 2 # Verify (affects both healths)
            
            action = self.prev_action[c_idx]
            
            # Prior: B * qs_prev
            prior_prob = np.dot(self.B[f][:, :, action], self.qs[f])
            
            # Posterior (Log domain)
            post_log = np.log(Likelihoods[f] + 1e-16) + np.log(prior_prob + 1e-16)
            new_qs.append(softmax(post_log))
            
        self.qs = new_qs
        return self.qs

    def plan_action(self):
        """
        Action Selection: Minimize EFE.
        """
        selected_actions = []

        # Control 0: Thermal (Affects F0)
        G_0 = self.calculate_G(factor_indices=[0], control_idx=0)
        selected_actions.append(self.sample_action(G_0))

        # Control 1: Load (Affects F1, F2)
        G_1 = self.calculate_G(factor_indices=[1, 2], control_idx=1)
        selected_actions.append(self.sample_action(G_1))

        # Control 2: Verify (Affects F3, F4)
        G_2 = self.calculate_G(factor_indices=[3, 4], control_idx=2)
        selected_actions.append(self.sample_action(G_2))

        self.prev_action = selected_actions
        return selected_actions

    def calculate_G(self, factor_indices, control_idx):
        num_actions = self.num_controls[control_idx]
        G = np.zeros(num_actions)

        for a in range(num_actions):
            total_value = 0
            for f in factor_indices:
                # Predict Next State
                qs_next = np.dot(self.B[f][:, :, a], self.qs[f])
                
                # Predict Observation (Simplified - assuming direct mapping for EFE)
                # For complex A, this is hard. We approximate.
                # We use the entropy of qs_next as Epistemic value directly.
                # And we check preference of expected state (if we mapped C to states).
                # But C is on observations.
                
                # Let's map C to states for simplicity in this robust model
                # C_state = C * A (approx)
                # Or just use the entropy of qs_next (Epistemic) + Utility of qs_next (Pragmatic)
                
                # Epistemic: H(qs_next) - We want high entropy? No, we want to resolve uncertainty.
                # Actually, Epistemic Value in EFE is about reducing parameter uncertainty or state uncertainty.
                # Actions that lead to "Verify" (Reset Health to Reliable) reduce entropy of Health.
                # If Health is 0.5/0.5, Entropy is high. Verify -> 1.0/0.0, Entropy low.
                # So we want to MINIMIZE expected entropy of next state (Information Gain).
                # Wait: Information Gain = H(Prior) - H(Posterior).
                # We want actions that result in Low Entropy Beliefs.
                
                entropy = -np.sum(qs_next * np.log(qs_next + 1e-16))
                
                # Pragmatic: How good is qs_next?
                # We need to map qs_next to observations to check against C.
                # This is complex with our coupled A.
                # Heuristic:
                # If f=0 (Temp): Prefer Opt (1).
                # If f=1 (Motor): Prefer Safe (0).
                # If f=3,4 (Health): Prefer Reliable (0).
                
                utility = 0
                if f == 0: utility = qs_next[1] * 2.0 # Opt
                elif f == 1: utility = qs_next[0] * 2.0 # Safe
                elif f == 2: utility = qs_next[1] * 1.0 # High Load
                elif f in [3, 4]: utility = qs_next[0] * 3.0 # Reliable
                
                # G = - (Utility + Epistemic_Bonus)
                # We want to minimize G.
                # Utility is good (negative G).
                # High Entropy state is BAD (positive G) -> We want low entropy.
                
                # But "Verify" action reduces entropy of Health.
                # So Verify -> Low Entropy -> Low G -> Selected.
                
                # Apply EFE mode
                if self.efe_mode == 'full':
                    total_value += utility - entropy  # Both epistemic and pragmatic
                elif self.efe_mode == 'epistemic_only':
                    total_value += -entropy  # Only epistemic (minimize entropy)
                elif self.efe_mode == 'pragmatic_only':
                    total_value += utility  # Only pragmatic (maximize utility)
            
            # --- Opportunity Cost for Verification ---
            # If this is the Verification control (idx 2) and action is Verify (1 or 2)
            if control_idx == 2 and a > 0:
                # Penalty: Represents lost production (Low Load vs High Load)
                # High Load Utility ~ 2.0 (from C matrix)
                # Low Load Utility ~ 0.0
                # So we lose ~2.0 units of utility per step of verification.
                # We add a penalty to the Value (reduce it).
                opportunity_cost = 2.0 
                total_value -= opportunity_cost

            G[a] = total_value

        return G

    def sample_action(self, G_values):
        probs = softmax(G_values * 5.0) # High precision to force good actions
        return np.random.choice(len(G_values), p=probs)

    def discretize_observation(self, temp, motor_temp, load, attack_detected):
        if temp < 40: obs_temp = 0
        elif temp <= 70: obs_temp = 1
        else: obs_temp = 2

        obs_motor = 0 if motor_temp < 80 else 1
        obs_load = 0 if load < 30 else 1
        obs_cyber = 1 if attack_detected else 0

        return [obs_temp, obs_motor, obs_load, obs_cyber]

    def map_action_to_control(self, actions):
        thermal_vals = [-5.0, 0.0, 5.0]
        temp_action = thermal_vals[actions[0]]

        load_vals = [-5.0, 0.0, 5.0]
        load_action = load_vals[actions[1]]

        # Verify: 0=Wait, 1=Verify Temp, 2=Verify Motor
        # Simulation only supports verifying Motor for now (or generic).
        # We map 1 or 2 to Verify=True.
        verify_action = actions[2] > 0

        return temp_action, load_action, verify_action

    def step(self, temp_reading, motor_temp_reading, load_reading, attack_detected=False):
        obs_indices = self.discretize_observation(temp_reading, motor_temp_reading, load_reading, attack_detected)
        self.infer_state(obs_indices)
        actions = self.plan_action()
        return self.map_action_to_control(actions)

if __name__ == "__main__":
    agent = ActiveInferenceAgent()
    print("Robust Agent initialized.")
    # Test Step
    t, l, v = agent.step(85.0, 90.0, 40.0, False)
    print(f"Step: {t}, {l}, {v}")
