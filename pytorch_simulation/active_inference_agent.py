import numpy as np
import torch


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class ActiveInferenceAgent:
    def __init__(self, efe_mode="full", precision=5.0):
        """
        Initialize the Active Inference agent.

        Args:
            efe_mode: EFE calculation mode. Options:
                - 'full': Use both epistemic (entropy) and pragmatic (utility) terms
                - 'epistemic_only': Use only epistemic term (minimize entropy)
                - 'pragmatic_only': Use only pragmatic term (maximize utility)
            precision: Inverse temperature for action selection (default 5.0).
                - Low (1-2): More exploration, random actions
                - Medium (5): Balanced
                - High (10+): More exploitation, deterministic actions

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
        valid_modes = ["full", "epistemic_only", "pragmatic_only"]
        if efe_mode not in valid_modes:
            raise ValueError(f"efe_mode must be one of {valid_modes}, got '{efe_mode}'")
        self.efe_mode = efe_mode
        self.precision = precision
        # --- Dimensions ---
        self.num_factors = 5
        self.num_states = [3, 2, 2, 2, 2]

        self.num_modalities = 4
        self.num_obs = [3, 2, 2, 2]

        self.num_controls = [3, 3, 3]  # Thermal, Load, Verify

        # --- Matrices ---
        self.A = self.initialize_A_matrix()  # Likelihood P(o|s)
        self.B = self.initialize_B_matrix()  # Transition P(s'|s,a)
        self.C = self.initialize_C_matrix()  # Preferences P(o)
        self.D = self.initialize_D_matrix()  # Prior P(s)

        # --- Internal State ---
        self.qs = [d.copy() for d in self.D]
        self.prev_action = [1, 1, 0]  # Initial actions (Maintain, Maintain, Wait)

    def initialize_A_matrix(self):
        """
        Likelihood mapping P(observation | state).

        Sensor-health factors modulate observation reliability:
        - reliable sensors preserve the state-to-observation mapping
        - spoofed sensors bias observations toward attacker-selected values
        """
        A = []

        # --- Modality 0: Obs Temperature ---
        # Explicit tensors are used because each modality depends on a limited
        # subset of factors rather than on the full joint state.
        A_temp = np.zeros((3, 3, 2))

        # Case 1: Reliable (Health=0) -> Identity
        A_temp[:, :, 0] = np.eye(3) * 0.9 + 0.033

        # Case 2: Spoofed (Health=1) -> Biased to Low (Index 0) to induce heating
        # If spoofed, we observe Low (0) regardless of true temp
        A_temp[:, :, 1] = np.array(
            [
                [0.9, 0.9, 0.9],  # High prob of observing Low
                [0.05, 0.05, 0.05],
                [0.05, 0.05, 0.05],
            ]
        )
        A.append(A_temp)

        # --- Modality 1: Obs Motor ---
        # Depends on Factor 1 (Motor) and Factor 4 (Motor Health)
        # Shape: [2 (Obs), 2 (Motor), 2 (Health)]
        A_motor = np.zeros((2, 2, 2))

        # Case 1: Reliable (Health=0) -> Identity
        A_motor[:, :, 0] = np.eye(2) * 0.9 + 0.05

        # Case 2: Spoofed (Health=1) -> Biased to Safe (Index 0) to hide overheating
        A_motor[:, :, 1] = np.array(
            [
                [0.9, 0.9],  # Always observe Safe
                [0.1, 0.1],
            ]
        )
        A.append(A_motor)

        # --- Modality 2: Obs Load ---
        # Load observations are modeled as reliable in the current formulation.
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
        B_temp_h[:, :, 0] = [[0.99, 0.05], [0.01, 0.95]]  # Sticky attacks
        B_temp_h[:, :, 2] = [[0.99, 0.05], [0.01, 0.95]]
        # Verification is modeled as a reset of the corresponding sensor-health factor.
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
        D.append(np.array([0.1, 0.8, 0.1]))  # Temp
        D.append(np.array([0.9, 0.1]))  # Motor
        D.append(np.array([0.8, 0.2]))  # Load
        D.append(np.array([0.9, 0.1]))  # Temp Health (Reliable)
        D.append(np.array([0.9, 0.1]))  # Motor Health (Reliable)
        return D

    def get_B_matrix(self):
        """Return the transition model for inspection utilities."""
        return self.B

    def infer_state(self, observations):
        """
        Perception: Minimize VFE.
        Custom implementation for non-standard A matrix shapes.
        """
        new_qs = []

        # Observation likelihoods are marginalized over the coupled health factors.

        # Get Obs Indices
        o_temp, o_motor, o_load, o_cyber = observations

        # 1. Likelihoods
        L_temp_F0 = np.dot(self.A[0][o_temp, :, :], self.qs[3])

        L_motor_F1 = np.dot(self.A[1][o_motor, :, :], self.qs[4])

        L_load_F2 = self.A[2][o_load, :]

        # L_Health_Temp(F3)
        L_HT_from_Temp = np.dot(self.A[0][o_temp, :, :].T, self.qs[0])
        L_HT_from_Cyber = np.dot(self.A[3][o_cyber, :, :], self.qs[4])
        L_HT_F3 = L_HT_from_Temp * L_HT_from_Cyber

        # L_Health_Motor(F4)
        L_HM_from_Motor = np.dot(self.A[1][o_motor, :, :].T, self.qs[1])
        L_HM_from_Cyber = np.dot(self.A[3][o_cyber, :, :].T, self.qs[3])
        L_HM_F4 = L_HM_from_Motor * L_HM_from_Cyber

        Likelihoods = [L_temp_F0, L_motor_F1, L_load_F2, L_HT_F3, L_HM_F4]

        # 2. Priors & Posterior
        for f in range(self.num_factors):
            # Determine control index for this factor
            if f == 0:
                c_idx = 0  # Thermal
            elif f in [1, 2]:
                c_idx = 1  # Load
            else:
                c_idx = 2  # Verify (affects both healths)

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
        Action selection with an EFE-derived value proxy.
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

                # The score is a compact proxy for Expected Free Energy:
                # low entropy captures epistemic value, while selected state
                # preferences capture pragmatic value. Higher scores are better.
                entropy = -np.sum(qs_next * np.log(qs_next + 1e-16))

                utility = 0
                if f == 0:
                    utility = qs_next[1] * 2.0  # Opt
                elif f == 1:
                    utility = qs_next[0] * 2.0  # Safe
                elif f == 2:
                    utility = qs_next[1] * 1.0  # High Load
                elif f in [3, 4]:
                    utility = qs_next[0] * 3.0  # Reliable

                if self.efe_mode == "full":
                    total_value += utility - entropy  # Both epistemic and pragmatic
                elif self.efe_mode == "epistemic_only":
                    total_value += -entropy  # Only epistemic (minimize entropy)
                elif self.efe_mode == "pragmatic_only":
                    total_value += utility  # Only pragmatic (maximize utility)

            if control_idx == 2 and a > 0:
                # Verification temporarily reduces production and is therefore penalized.
                opportunity_cost = 2.0
                total_value -= opportunity_cost

            G[a] = total_value

        return G

    def sample_action(self, G_values):
        probs = softmax(G_values * self.precision)
        return np.random.choice(len(G_values), p=probs)

    def discretize_observation(self, temp, motor_temp, load, attack_detected):
        if temp < 40:
            obs_temp = 0
        elif temp <= 70:
            obs_temp = 1
        else:
            obs_temp = 2

        obs_motor = 0 if motor_temp < 80 else 1
        obs_load = 0 if load < 30 else 1
        obs_cyber = 1 if attack_detected else 0

        return [obs_temp, obs_motor, obs_load, obs_cyber]

    def map_action_to_control(self, actions):
        thermal_vals = [-5.0, 0.0, 5.0]
        temp_action = thermal_vals[actions[0]]

        load_vals = [-5.0, 0.0, 5.0]
        load_action = load_vals[actions[1]]

        # The current simulator exposes verification as a boolean action.
        verify_action = actions[2] > 0

        return temp_action, load_action, verify_action

    def step(
        self, temp_reading, motor_temp_reading, load_reading, attack_detected=False
    ):
        obs_indices = self.discretize_observation(
            temp_reading, motor_temp_reading, load_reading, attack_detected
        )
        self.infer_state(obs_indices)
        actions = self.plan_action()
        return self.map_action_to_control(actions)


class AdaptiveActiveInferenceAgent(ActiveInferenceAgent):
    """
    Active Inference Agent that learns transition dynamics (B matrix) online.

    On long simulations, this agent should outperform the fixed-model agent
    as it learns the true dynamics of the environment.

    Supports multiple learning rate schedules for thesis comparison:
    - 'constant': Fixed learning rate throughout
    - 'decay': Exponential decay (lr = lr_0 * decay_rate^step)
    - 'warmup': Linear warmup then constant
    - 'cosine': Cosine annealing
    - 'adaptive': Adjusts based on prediction error
    """

    def __init__(
        self,
        efe_mode="full",
        precision=5.0,
        learning_rate=0.01,
        lr_schedule="constant",
        lr_decay_rate=0.9999,
        lr_warmup_steps=100,
        lr_min=0.001,
        lr_cosine_max_steps=5000,
    ):
        """
        Initialize adaptive agent with learning capabilities.

        Args:
            efe_mode: EFE calculation mode ('full', 'epistemic_only', 'pragmatic_only')
            precision: Inverse temperature for action selection
            learning_rate: Initial/base learning rate (0.001-0.1 recommended)
            lr_schedule: Learning rate schedule ('constant', 'decay', 'warmup', 'cosine', 'adaptive')
            lr_decay_rate: Decay rate for 'decay' schedule (default 0.9999)
            lr_warmup_steps: Number of warmup steps for 'warmup' schedule
            lr_min: Minimum learning rate for scheduled approaches
            lr_cosine_max_steps: Max steps for cosine annealing cycle (default 5000)
        """
        super().__init__(efe_mode=efe_mode, precision=precision)

        # Learning rate configuration
        self.base_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.lr_schedule = lr_schedule
        self.lr_decay_rate = lr_decay_rate
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_min = lr_min
        self.lr_cosine_max_steps = lr_cosine_max_steps

        # Experience counts for learning B matrix
        # For each factor: counts[next_state, current_state, action]
        self.B_counts = []
        for f in range(self.num_factors):
            n_states = self.num_states[f]
            if f == 0:
                n_actions = self.num_controls[0]  # Thermal
            elif f in [1, 2]:
                n_actions = self.num_controls[1]  # Load
            else:
                n_actions = self.num_controls[2]  # Verify
            # Initialize with small pseudo-counts to avoid division by zero
            self.B_counts.append(np.ones((n_states, n_states, n_actions)) * 0.1)

        # Store initial B matrix for comparison
        self.B_initial = [b.copy() for b in self.B]

        # Store previous state estimates for learning
        self.prev_qs = None
        self.prev_action = [1, 1, 0]

        # Tracking for analysis
        self.total_updates = 0
        self.current_step = 0

        # Learning history for thesis analysis
        self.learning_history = {
            "step": [],
            "learning_rate": [],
            "prediction_error": [],
            "b_matrix_change": [],
            "b_matrix_entropy": [],
            "model_divergence": [],  # KL divergence from initial model
        }

        # For adaptive LR
        self.recent_errors = []
        self.error_window = 50

    def _update_learning_rate(self):
        """Update learning rate based on schedule."""
        step = self.current_step

        if self.lr_schedule == "constant":
            return  # Keep base_learning_rate

        elif self.lr_schedule == "decay":
            # Exponential decay
            self.learning_rate = max(
                self.lr_min, self.base_learning_rate * (self.lr_decay_rate**step)
            )

        elif self.lr_schedule == "warmup":
            # Linear warmup then constant
            if step < self.lr_warmup_steps:
                self.learning_rate = self.base_learning_rate * (
                    step / max(1, self.lr_warmup_steps)
                )
            else:
                self.learning_rate = self.base_learning_rate

        elif self.lr_schedule == "cosine":
            # Cosine annealing
            self.learning_rate = self.lr_min + 0.5 * (
                self.base_learning_rate - self.lr_min
            ) * (1 + np.cos(np.pi * step / self.lr_cosine_max_steps))

        elif self.lr_schedule == "adaptive":
            # Adjust based on recent prediction errors
            if len(self.recent_errors) >= self.error_window:
                recent_avg = np.mean(self.recent_errors[-self.error_window :])
                older_avg = (
                    np.mean(
                        self.recent_errors[-2 * self.error_window : -self.error_window]
                    )
                    if len(self.recent_errors) >= 2 * self.error_window
                    else recent_avg
                )

                # If error is decreasing, reduce LR (converging)
                # If error is increasing or stagnant, increase LR (need more learning)
                if recent_avg < older_avg * 0.95:  # Improving
                    self.learning_rate = max(self.lr_min, self.learning_rate * 0.99)
                elif recent_avg > older_avg * 1.05:  # Degrading
                    self.learning_rate = min(
                        self.base_learning_rate * 2, self.learning_rate * 1.01
                    )

    def _compute_prediction_error(self, prev_qs, curr_qs, actions):
        """Compute prediction error for learning metrics."""
        total_error = 0.0

        for f in range(self.num_factors):
            if f == 0:
                action = actions[0]
            elif f in [1, 2]:
                action = actions[1]
            else:
                action = actions[2]

            # What we predicted
            predicted_qs = np.dot(self.B[f][:, :, action], prev_qs[f])

            # What we observed
            actual_qs = curr_qs[f]

            # KL divergence as prediction error
            kl_div = np.sum(
                actual_qs * np.log((actual_qs + 1e-10) / (predicted_qs + 1e-10))
            )
            total_error += max(0, kl_div)  # Ensure non-negative

        return total_error / self.num_factors

    def _compute_model_divergence(self):
        """Compute KL divergence from initial B matrix (model change)."""
        total_kl = 0.0
        count = 0

        for f in range(self.num_factors):
            for a in range(self.B[f].shape[2]):
                for s in range(self.B[f].shape[1]):
                    p = self.B[f][:, s, a]
                    q = self.B_initial[f][:, s, a]
                    kl = np.sum(p * np.log((p + 1e-10) / (q + 1e-10)))
                    total_kl += max(0, kl)
                    count += 1

        return total_kl / count if count > 0 else 0.0

    def _compute_b_matrix_change(self, prev_B):
        """Compute how much B matrix changed in this step."""
        total_change = 0.0

        for f in range(self.num_factors):
            change = np.abs(self.B[f] - prev_B[f]).mean()
            total_change += change

        return total_change / self.num_factors

    def _record_learning_metrics(self, pred_error, prev_B):
        """Record metrics for learning curve analysis."""
        self.learning_history["step"].append(self.current_step)
        self.learning_history["learning_rate"].append(self.learning_rate)
        self.learning_history["prediction_error"].append(pred_error)
        self.learning_history["b_matrix_change"].append(
            self._compute_b_matrix_change(prev_B)
        )
        self.learning_history["b_matrix_entropy"].append(
            np.mean(
                [
                    np.mean(-np.sum(self.B[f] * np.log(self.B[f] + 1e-10), axis=0))
                    for f in range(self.num_factors)
                ]
            )
        )
        self.learning_history["model_divergence"].append(
            self._compute_model_divergence()
        )

    def step(
        self, temp_reading, motor_temp_reading, load_reading, attack_detected=False
    ):
        """
        Extended step that also learns from experience.
        """
        # Update learning rate based on schedule
        self._update_learning_rate()

        # Discretize current observation
        obs_indices = self.discretize_observation(
            temp_reading, motor_temp_reading, load_reading, attack_detected
        )

        # Store previous beliefs before updating
        prev_qs_copy = (
            [q.copy() for q in self.qs] if self.prev_qs is None else self.prev_qs
        )

        # Store previous B for change tracking
        prev_B = [b.copy() for b in self.B]

        # Standard perception
        self.infer_state(obs_indices)

        # Compute prediction error before learning
        pred_error = self._compute_prediction_error(
            prev_qs_copy, self.qs, self.prev_action
        )
        self.recent_errors.append(pred_error)

        # Learn from transition (previous state -> current state)
        self._update_B_matrix(prev_qs_copy, self.qs, self.prev_action)

        # Store current beliefs for next learning step
        self.prev_qs = [q.copy() for q in self.qs]

        # Record learning history every 10 steps (for efficiency)
        if self.current_step % 10 == 0:
            self._record_learning_metrics(pred_error, prev_B)

        self.current_step += 1

        # Plan action
        actions = self.plan_action()
        self.prev_action = actions

        return self.map_action_to_control(actions)

    def _update_B_matrix(self, prev_qs, curr_qs, actions):
        """
        Update B matrix based on observed transition.

        Uses soft assignment: weight the update by probability of being in each state.
        """
        for f in range(self.num_factors):
            # Determine which action controls this factor
            if f == 0:
                action = actions[0]  # Thermal
            elif f in [1, 2]:
                action = actions[1]  # Load
            else:
                action = actions[2]  # Verify

            # Outer product gives expected count: P(s_prev) * P(s_curr)
            # This is a soft assignment of the transition
            expected_transition = np.outer(curr_qs[f], prev_qs[f])

            # Add to counts with current learning rate
            self.B_counts[f][:, :, action] += (
                expected_transition * self.learning_rate * 10
            )

            # Normalize to get new B matrix (column-stochastic)
            for a in range(self.B_counts[f].shape[2]):
                for s in range(self.B_counts[f].shape[1]):
                    col_sum = self.B_counts[f][:, s, a].sum()
                    if col_sum > 0:
                        self.B[f][:, s, a] = self.B_counts[f][:, s, a] / col_sum

        self.total_updates += 1

    def get_learning_stats(self):
        """Return comprehensive learning statistics for analysis."""
        return {
            "total_updates": self.total_updates,
            "current_step": self.current_step,
            "learning_rate": self.learning_rate,
            "base_learning_rate": self.base_learning_rate,
            "lr_schedule": self.lr_schedule,
            "B_matrix_entropy": [
                np.mean(-np.sum(self.B[f] * np.log(self.B[f] + 1e-10), axis=0))
                for f in range(self.num_factors)
            ],
            "model_divergence": self._compute_model_divergence(),
            "avg_prediction_error": np.mean(self.recent_errors[-100:])
            if self.recent_errors
            else 0.0,
        }

    def get_learning_history_df(self):
        """Return learning history as pandas DataFrame for plotting."""
        import pandas as pd

        return pd.DataFrame(self.learning_history)

    def reset_learning(self):
        """Reset learning state (useful for experiments)."""
        self.B = [b.copy() for b in self.B_initial]
        self.B_counts = []
        for f in range(self.num_factors):
            n_states = self.num_states[f]
            if f == 0:
                n_actions = self.num_controls[0]
            elif f in [1, 2]:
                n_actions = self.num_controls[1]
            else:
                n_actions = self.num_controls[2]
            self.B_counts.append(np.ones((n_states, n_states, n_actions)) * 0.1)

        self.total_updates = 0
        self.current_step = 0
        self.learning_rate = self.base_learning_rate
        self.recent_errors = []
        self.learning_history = {k: [] for k in self.learning_history.keys()}

    def save_model(self, filepath):
        """
        Save the learned B matrix and agent state to disk.

        Enables transfer learning: train on one scenario, deploy on another.

        Args:
            filepath: Path to save the model (will add .npz extension)

        Returns:
            str: Path to the saved file
        """
        import os

        # Ensure .npz extension
        if not filepath.endswith(".npz"):
            filepath = filepath + ".npz"

        # Create directory if needed
        os.makedirs(
            os.path.dirname(filepath) if os.path.dirname(filepath) else ".",
            exist_ok=True,
        )

        # Prepare data to save
        save_data = {
            # Learned B matrix (main model)
            "B_0": self.B[0],
            "B_1": self.B[1],
            "B_2": self.B[2],
            "B_3": self.B[3],
            "B_4": self.B[4],
            # Experience counts for continued learning
            "B_counts_0": self.B_counts[0],
            "B_counts_1": self.B_counts[1],
            "B_counts_2": self.B_counts[2],
            "B_counts_3": self.B_counts[3],
            "B_counts_4": self.B_counts[4],
            # Agent state
            "total_updates": np.array([self.total_updates]),
            "current_step": np.array([self.current_step]),
            "learning_rate": np.array([self.learning_rate]),
            "base_learning_rate": np.array([self.base_learning_rate]),
            # Config
            "efe_mode": np.array([self.efe_mode], dtype=object),
            "precision": np.array([self.precision]),
            "lr_schedule": np.array([self.lr_schedule], dtype=object),
        }

        np.savez(filepath, **save_data)
        print(f"Model saved to {filepath}")
        return filepath

    def load_model(self, filepath, continue_learning=True):
        """
        Load a previously saved B matrix and agent state.

        Args:
            filepath: Path to the saved model (.npz file)
            continue_learning: If True, keep B_counts for continued learning.
                             If False, only load B matrix (frozen model).

        Returns:
            dict: Metadata about the loaded model
        """
        # Ensure .npz extension
        if not filepath.endswith(".npz"):
            filepath = filepath + ".npz"

        data = np.load(filepath, allow_pickle=True)

        # Load B matrix
        self.B = [
            data["B_0"],
            data["B_1"],
            data["B_2"],
            data["B_3"],
            data["B_4"],
        ]

        if continue_learning:
            # Load experience counts for continued learning
            self.B_counts = [
                data["B_counts_0"],
                data["B_counts_1"],
                data["B_counts_2"],
                data["B_counts_3"],
                data["B_counts_4"],
            ]
            self.total_updates = int(data["total_updates"][0])
            self.current_step = int(data["current_step"][0])
        else:
            # Reconstruct pseudo-counts from the loaded transition model.
            self.B_counts = []
            for f in range(self.num_factors):
                n_states = self.num_states[f]
                if f == 0:
                    n_actions = self.num_controls[0]
                elif f in [1, 2]:
                    n_actions = self.num_controls[1]
                else:
                    n_actions = self.num_controls[2]
                # Initialize counts from loaded B to maintain distribution
                self.B_counts.append(self.B[f] * 10 + 0.1)
            self.total_updates = 0
            self.current_step = 0

        metadata = {
            "filepath": filepath,
            "total_updates": int(data["total_updates"][0]),
            "original_step": int(data["current_step"][0]),
            "efe_mode": str(data["efe_mode"][0]),
            "precision": float(data["precision"][0]),
            "lr_schedule": str(data["lr_schedule"][0]),
            "continue_learning": continue_learning,
        }

        print(f"Model loaded from {filepath}")
        print(f"  - Original training steps: {metadata['original_step']}")
        print(f"  - Continue learning: {continue_learning}")

        return metadata

    def get_b_matrix_snapshot(self):
        """
        Get a snapshot of the current B matrix for visualization.

        Returns:
            dict: Dictionary with B matrix data for each factor
        """
        factor_names = ["Temperature", "Motor", "Load", "Temp_Health", "Motor_Health"]
        action_names = [
            ["Cool", "Maintain", "Heat"],  # Factor 0
            ["Decrease", "Maintain", "Increase"],  # Factor 1
            ["Decrease", "Maintain", "Increase"],  # Factor 2
            ["Wait", "Verify_Temp", "Verify_Motor"],  # Factor 3
            ["Wait", "Verify_Temp", "Verify_Motor"],  # Factor 4
        ]
        state_names = [
            ["Low", "Optimal", "High"],  # Factor 0
            ["Safe", "Overheating"],  # Factor 1
            ["Low", "High"],  # Factor 2
            ["Reliable", "Spoofed"],  # Factor 3
            ["Reliable", "Spoofed"],  # Factor 4
        ]

        snapshot = {}
        for f in range(self.num_factors):
            snapshot[factor_names[f]] = {
                "matrix": self.B[f].copy(),
                "initial_matrix": self.B_initial[f].copy(),
                "change": self.B[f] - self.B_initial[f],
                "actions": action_names[f],
                "states": state_names[f],
            }

        return snapshot


if __name__ == "__main__":
    agent = ActiveInferenceAgent()
    print("Active Inference agent initialized.")
    t, l, v = agent.step(85.0, 90.0, 40.0, False)
    print(f"Static step output: {t}, {l}, {v}")

    adaptive_agent = AdaptiveActiveInferenceAgent(learning_rate=0.01)
    print("Adaptive Active Inference agent initialized.")
    t, l, v = adaptive_agent.step(85.0, 90.0, 40.0, False)
    print(f"Adaptive step output: {t}, {l}, {v}")
