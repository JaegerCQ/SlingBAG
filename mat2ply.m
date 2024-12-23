%% 清理环境变量
clc; clear;

%% 加载 liver 数据
load('data\source.mat'); % 确保加载的 liver 变量存在

%% 设置输出文件夹
output_folder = 'data';
% 如果文件夹不存在，则创建文件夹
if ~exist(output_folder, 'dir')
mkdir(output_folder);
end

%% 初始化用于临时存储数据
phantom = source;

% 自动获取尺寸
[liver_size_x, liver_size_y, liver_size_z] = size(phantom);

%% 提取非零点云信息
% 获取非零点位置和强度
non_zero_indices = find(phantom > 0);
[x_idx, y_idx, z_idx] = ind2sub(size(phantom), non_zero_indices);
intensity_values = phantom(non_zero_indices);

%% 转换为物理坐标（单位：米）
scale = 0.1e-3; % 每像素实际尺寸
x_phys = (x_idx - liver_size_x / 2) * scale;
y_phys = (y_idx - liver_size_y / 2) * scale;
z_phys = (z_idx - liver_size_z / 2) * scale;
radius = ones(size(x_phys)) * 0.1e-3; % 定义点云默认尺寸

%% 创建输出 .ply 文件
ply_filename = fullfile(output_folder, 'phantom_for_simulation.ply');

% 打开文件，检查是否成功
fileID = fopen(ply_filename, 'w');
if fileID == -1
    error('无法创建或打开文件：%s。请检查路径权限或文件是否被占用。', ply_filename);
end

% 写入 PLY 文件头
fprintf(fileID, 'ply\n');
fprintf(fileID, 'format ascii 1.0\n');
fprintf(fileID, 'element vertex %d\n', numel(x_phys));
fprintf(fileID, 'property float x\n');
fprintf(fileID, 'property float y\n');
fprintf(fileID, 'property float z\n');
fprintf(fileID, 'property float intensity\n');
fprintf(fileID, 'property float radius\n');
fprintf(fileID, 'end_header\n');

% 写入点云数据
data = [x_phys, y_phys, z_phys, intensity_values, radius];
fprintf(fileID, '%.9f %.9f %.9f %.9f %.9f\n', data');

% 关闭文件
fclose(fileID);

%% 显示操作完成信息
disp('PLY 文件已保存：');
disp(ply_filename);