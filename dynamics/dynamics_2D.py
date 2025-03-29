import math
import torch
from dynamics import Dynamics

class PlanarRobot2D(Dynamics):
    def __init__(self, obstacle_radius: float = 0.5, velocity: float = 1.0):
        self.obstacle_radius = obstacle_radius  # Obstacle radius at origin
        self.velocity = velocity  # Constant velocity of the robot
        
        super().__init__(
            loss_type='brt_hjivi', set_mode='avoid',
            state_dim=2, input_dim=3, control_dim=1, disturbance_dim=0,
            state_mean=[0, 0],  # [x, y]
            state_var=[2, 2],
            value_mean=0.25,
            value_var=0.5,
            value_normto=0.02,
            deepreach_model="exact",
        )
    
    def state_test_range(self):
        return [
            [-2, 2], #px
            [-2, 2], #py
        ]
    
    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        # wrapped_state[..., 2] = (wrapped_state[..., 2] + math.pi) % (2 * math.pi) - math.pi
        return wrapped_state
    
    def dsdt(self, state, control):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = self.velocity * torch.cos(control[..., 0])  # dx/dt
        dsdt[..., 1] = self.velocity * torch.sin(control[..., 0])  # dy/dt
        return dsdt
    
    def boundary_fn(self, state):
        return torch.norm(state, dim=-1) - self.obstacle_radius
    
    def sample_target_state(self, num_samples):
        raise NotImplementedError
    
    def hamiltonian(self, state, dvds):
        return self.velocity * torch.norm(dvds, dim=-1)
    
    def optimal_control(self, state, dvds):
        theta_star = torch.atan2(dvds[..., 1], dvds[..., 0])
        return theta_star.unsqueeze(-1)
    
    def optimal_disturbance(self, state, dvds):
        raise NotImplementedError
    
    def plot_config(self):
        return {
            'state_slices': [0, 0],
            'state_labels': ['x', 'y'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
        }
