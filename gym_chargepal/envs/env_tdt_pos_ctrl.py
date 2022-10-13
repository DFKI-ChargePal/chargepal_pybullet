# global
import numpy as np

# local
from gym_chargepal.envs.env import Environment
from gym_chargepal.worlds.world_tdt import WorldTopDownTask
from gym_chargepal.bullet.ik_solver import IKSolver
from gym_chargepal.bullet.joint_position_motor_control import JointPositionMotorControl
from gym_chargepal.sensors.sensor_plug import PlugSensor
from gym_chargepal.sensors.sensor_socket import SocketSensor
from gym_chargepal.controllers.controller_tcp_pos import TcpPositionController
from gym_chargepal.eval.eval_dist import EvalDistance

# mypy
import numpy.typing as npt
from typing import Any, Callable, Dict, Tuple 


class EnvironmentTcpPositionCtrlTdt(Environment):
    """ Environment with cartesian position controller - Task: Top down task """

    def __init__(self, **kwargs: Dict[str, Any]):
        # Update environment configuration
        config_env = {} if 'config_env' not in kwargs else kwargs['config_env']
        super().__init__(config_env)

        # extract component hyperparameter from kwargs
        extract_config: Callable[[str], Dict[str, Any]] = lambda name: {} if name not in kwargs else kwargs[name]
        config_eval = extract_config('config_eval')
        config_world = extract_config('config_world')
        config_ik_solver = extract_config('config_ik_solver')
        config_control_interface = extract_config('config_control_interface')
        config_low_level_control = extract_config('config_low_level_control')
        config_plug_sensor = {} if 'config_plug_sensor' not in kwargs else kwargs['config_plug_sensor']
        config_socket_sensor = {} if 'config_socket_sensor' not in kwargs else kwargs['config_socket_sensor']

        # start configuration in world coordinates
        self.pos_world_0 = tuple(
            [sum(ep) for ep in zip(self.hyperparams['tgt_config_pos'], self.hyperparams['start_config_pos'])]
            )
        self.ang_world_0 = tuple(
            [sum(ep) for ep in zip(self.hyperparams['tgt_config_ang'], self.hyperparams['start_config_ang'])]
            )
        # reset variance
        self.pos_var_0 = self.hyperparams['reset_variance'][0]
        self.ang_var_0 = self.hyperparams['reset_variance'][1]
        # resolve cross references
        config_low_level_control['plug_lin_config'] = self.pos_world_0
        config_low_level_control['plug_ang_config'] = self.ang_world_0
        # render option can be enabled with render() function
        self.is_render = False
        self.toggle_render_mode = False
        # components
        self.world = WorldTopDownTask(config_world)
        self.ik_solver = IKSolver(config_ik_solver, self.world)
        self.control_interface = JointPositionMotorControl(config_control_interface, self.world)
        self.plug_sensor = PlugSensor(config_plug_sensor, self.world)
        self.socket_sensor = SocketSensor(config_socket_sensor, self.world)
        self.low_level_control = TcpPositionController(
            config_low_level_control,
            self.ik_solver,
            self.control_interface,
            self.plug_sensor
        )
        self.eval = EvalDistance(config_eval, self.world, self.clock)
        # logging
        self.error_pos = np.inf
        self.error_ang = np.inf

    def reset(self) -> npt.NDArray[np.float32]:
        # reset environment
        self.clock.reset()
        if self.toggle_render_mode:
            self.world.disconnect()
            self.toggle_render_mode = False
        # reset robot by default joint configuration
        self.world.reset(render=self.is_render)
        # get start joint configuration by inverse kinematic
        pos_0 = tuple(np.array(self.pos_world_0) + np.array(self.pos_var_0) * self.rs.randn(3))  # type: ignore
        ang_0 = tuple(np.array(self.ang_world_0) + np.array(self.ang_var_0) * self.rs.randn(3))  # type: ignore
        ori_0 = self.world.bullet_client.getQuaternionFromEuler(ang_0)
        joint_config_0 = self.ik_solver.solve((pos_0, ori_0))
        # reset robot again
        self.world.reset(joint_config_0)
        # update sensors states
        self.update_sensors(target_sensor=True)
        return self.get_obs()
    
    def step(self, action: npt.NDArray[np.float32]) -> Tuple[npt.NDArray[np.float32], float, bool, Dict[Any, Any]]:
        """ Execute environment/simulation step. """
        # apply action
        self.low_level_control.update(action=np.array(action))
        # step simulation
        self.world.step(render=self.is_render)
        self.clock.tick()
        # update states
        self.update_sensors()
        obs = self.get_obs()
        # evaluate environment
        done = self.eval.is_done()
        reward = self.eval.calc_reward(
            pos_ee=self.plug_sensor.get_pos(),
            ori_ee=self.plug_sensor.get_ori(),
            pos_tg=self.socket_sensor.get_pos(),
            ori_tg=self.socket_sensor.get_ori()
        )
        info = self.compose_info()
        return obs, reward, done, info

    def render(self, mode: str = "human") -> None:
        self.toggle_render_mode = True if not self.is_render else False
        self.is_render = True

    def close(self) -> None:
        self.world.disconnect()

    def update_sensors(self, target_sensor: bool=False) -> None:
        if target_sensor:
            self.socket_sensor.update()
        self.plug_sensor.update()

    def get_obs(self) -> npt.NDArray[np.float32]:
        tgt_pos = np.array(self.socket_sensor.get_pos())
        plg_pos = np.array(self.plug_sensor.get_pos())
        dif_pos: Tuple[float, ...] = tuple(tgt_pos - plg_pos)

        tgt_ori = self.socket_sensor.get_ori()
        plg_ori = self.plug_sensor.get_ori()
        dif_ori = self.world.bullet_client.getDifferenceQuaternion(plg_ori, tgt_ori)
        obs = np.array((dif_pos + dif_ori), dtype=np.float32)

        tgt_ori_ = np.array(tgt_ori)
        plg_ori_ = np.array(plg_ori)
        self.error_pos = np.sqrt(np.sum(np.square(dif_pos)))
        self.error_ang = np.arccos(np.clip((2 * (tgt_ori_.dot(plg_ori_))**2 - 1), -1.0, 1.0))
        return obs

    def compose_info(self) -> Dict[str, Any]:
        info = {
            'error_pos': self.error_pos,
            'error_ang': self.error_ang,
            'solved': self.eval.is_solved(self.error_pos, self.error_ang),
        }
        return info
