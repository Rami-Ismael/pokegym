from dataclasses import dataclass

@dataclass
class Reward:
    reward: float
    done: bool
    info: dict

    def __post_init__(self):
        self.reward = float(self.reward)
        self.done = bool(self.done)
        self.info = dict(self.info)
        #print(f"Reward: {self.reward}, Done: {self.done}, Info: {self.info}")