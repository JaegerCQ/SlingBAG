import torch
from torch.nn import Parameter

class SlidingBallModel(torch.nn.Module):
    def __init__(self, xyz=None, pressure_0=None, radius=None): 
        super(SlidingBallModel, self).__init__()
        self._xyz = Parameter(torch.tensor(xyz, dtype=torch.float32) if xyz is not None else torch.empty(0))
        self._pressure_0 = Parameter(torch.tensor(pressure_0, dtype=torch.float32) if pressure_0 is not None else torch.empty(0))
        self._radius = Parameter(torch.tensor(radius, dtype=torch.float32) if radius is not None else torch.empty(0))
        self._is_destroyed = False

    def initialize_attributes(self, xyz, pressure_0, radius):
        """初始化球的属性值"""
        self._xyz = Parameter(torch.tensor(xyz, dtype=torch.float32, requires_grad=True, device=torch.device('cuda:0')))
        self._pressure_0 = Parameter(torch.tensor(pressure_0, dtype=torch.float32, requires_grad=True, device=torch.device('cuda:0')))
        self._radius = Parameter(torch.tensor(radius, dtype=torch.float32, requires_grad=True, device=torch.device('cuda:0')))

    def get_attributes(self):
        """访问球的当前属性值"""
        if self._is_destroyed:
            return {'xyz': torch.empty(0), 'pressure_0': torch.empty(0), 'radius': torch.empty(0)}
        return {
            'xyz': self._xyz.clone(),
            'pressure_0': self._pressure_0.clone(),
            'radius': self._radius.clone()
        }
    
    def adaptive_density_optimization(self, pressure_threshold, radius_max_threshold, radius_min_threshold, boundaries,gradient_direction=None):
        """球的自适应密度优化"""
        if self._pressure_0 < pressure_threshold or self._radius < radius_min_threshold or self.is_out_of_bounds(boundaries):
            self._destroy()
            return None
        elif self._radius > radius_max_threshold:
            # 使用传入的梯度方向，确保它是单位向量
            if gradient_direction is not None:
                gradient_direction = self.normalize_gradient(gradient_direction)
            
            # 分裂为两个球
            new_radius = self._radius / 2
            self._radius = Parameter(new_radius)

            # 生成一个新球，位置略微偏移，其他属性相同
            new_xyz_offset = gradient_direction * self._radius.item() if gradient_direction is not None else torch.tensor([self._radius.item(), 0, 0], dtype=torch.float32, device=self._xyz.device)
            new_xyz = self._xyz + new_xyz_offset
            if self.is_out_of_bounds(boundaries, new_xyz):
                return None
            
            new_ball = self.__class__()
            new_ball.initialize_attributes(new_xyz.detach().cpu().numpy(), self._pressure_0.item(), new_radius.item())
            return new_ball

    def clone_along_gradient(self, gradient_direction, boundaries,res):
        """沿着位置梯度方向进行复制"""
        if not torch.is_tensor(gradient_direction):
            gradient_direction = torch.tensor(gradient_direction, dtype=torch.float32, device=self._xyz.device)
        
        # 归一化梯度方向为单位向量
        gradient_direction = self.normalize_gradient(gradient_direction)

        new_xyz = self._xyz + gradient_direction*res
        if self.is_out_of_bounds(boundaries, new_xyz):
            return None
        
        new_ball = self.__class__()
        new_ball.initialize_attributes(new_xyz.detach().cpu().numpy(), self._pressure_0.item(), self._radius.item())
        return new_ball

    def normalize_gradient(self, gradient_direction):
        """归一化梯度为单位向量"""
        norm = torch.norm(gradient_direction)
        if norm != 0:
            return gradient_direction / norm
        return gradient_direction

    def _destroy(self):
        """标记实例对象为销毁状态"""
        self._is_destroyed = True

    def is_out_of_bounds(self, boundaries, xyz=None):
        """检查球是否超出边界"""
        if xyz is None:
            xyz = self._xyz
        x, y, z = xyz
        x_min, x_max, y_min, y_max, z_min, z_max = boundaries
        return not (x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max)