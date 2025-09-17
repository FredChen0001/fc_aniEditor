import math
import torch



class CoToHelper():
    """
    Dynamically updates the `cotodrop` flags for LoRA adapters based on training progress.
    """

    def __init__(self, adapter_modules, adpater_name='default', initial_p=0.1, final_p=1.0, stage1_ratio=0.75):
        self.loras = adapter_modules
        self.initial_p = initial_p
        self.final_p = final_p
        self.stage1_ratio = stage1_ratio
        self.total_steps = None
        self.adapter_name = adpater_name

    def on_train_begin(self, total_steps):
        self.total_steps = total_steps
        self.update_dropout_rate(self.initial_p)

    def on_step_end(self, step):
        end_step = math.ceil(self.total_steps * self.stage1_ratio)
        rate = self.initial_p + (self.final_p - self.initial_p) * (step / end_step)
        self.update_dropout_rate(min(rate, self.final_p))

    def update_dropout_rate(self, rate):
        count = len(self.loras)
        random_tensor = generate_random_tensor(count, rate)
        for i, lora in enumerate(self.loras):
            lora.scaling[self.adapter_name] = float(random_tensor[i].item() <= rate)

def generate_random_tensor(n, k):
    """Generate a tensor of length `n` with at least one value ≤ k."""
    while True:
        t = torch.rand(n)
        if not torch.all(t > k):
            return t