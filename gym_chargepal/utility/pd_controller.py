from __future__ import annotations


class PDController:

    def __init__(self, kp: float = 0.0, kd: float = 0.0) -> None:
        """ One dimension proportional, derivative controller

        Args:
            kp: Proportional gain. Defaults to 0.0.
            kd: Derivative gain. Defaults to 0.0.
        """
        self.kp = abs(kp)
        self.kd = abs(kd)
        self.prev_error: float | None = None  # Previous controller error

    def reset(self) -> None:
        """ Reset controller """
        self.prev_error = None

    def update(self, error: float, period: float) -> float:
        """ Controller update step.

        Args:
            error: Controller error
            period: Controller update duration [sec]

        Returns:
            Controller output/system input
        """
        if period > 0.0:
            if self.prev_error is None:
                self.prev_error = error
            result = self.kp * error + self.kd * (error - self.prev_error) / period
            self.prev_error = error
        else:
            result = 0.0
        return result
