class EnvironmentClock:

    def __init__(self, time_horizon: int) -> None:
        self.t = 0
        self.t_end = time_horizon
        self.fin = False

    def tick(self) -> None:
        self.t += 1
        if self.t >= self.t_end:
            self.fin = True

    def reset(self) -> None:
        self.t = 0
        self.fin = False

    def is_at_time_horizon(self) -> bool:
        return self.fin
