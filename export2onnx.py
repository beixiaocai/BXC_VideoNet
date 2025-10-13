import torch
import torch.onnx
from model import VideoNetModel
import os


def convert_to_onnx(model_path='best_model.pth', output_path='best_model.onnx',num_classes=0, dynamic_input=True):
    print(f"开始将PyTorch模型转换为ONNX格式...")
    print(f"输入模型: {model_path}")
    print(f"输出ONNX文件: {output_path}")

    # 初始化模型
    model = VideoNetModel(num_classes=num_classes)
    
    # 加载模型权重 - 处理多GPU训练的情况
    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        # 检查是否是多GPU训练的模型
        if any(key.startswith('module.') for key in state_dict.keys()):
            # 移除DataParallel包装的键前缀
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return
    
    model.eval()
    print("模型加载成功")
    
    # 创建输入张量 - 使用与训练相同的输入尺寸
    batch_size = 1
    num_frames = 16
    height = 224
    width = 224
    dummy_input = torch.randn(batch_size, 3, num_frames, height, width)
    print(f"创建虚拟输入: 形状 {dummy_input.shape}")
    
    # 导出ONNX模型 - 优化配置
    export_kwargs = {
        'model': model,
        'args': (dummy_input,),
        'f': output_path,
        'export_params': True,
        'opset_version': 13,  # 使用较新的opset以获得更好的性能
        'do_constant_folding': True,
        'input_names': ['input'],
        'output_names': ['output'],
        'verbose': False
    }
    
    # 如果启用动态输入
    if dynamic_input:
        dynamic_axes = {
            'input': {
                0: 'batch_size',  # 批次维度
                2: 'frames'       # 时间维度
            },
            'output': {
                0: 'batch_size'   # 批次维度
            }
        }
        export_kwargs['dynamic_axes'] = dynamic_axes
        print("启用动态输入维度: batch_size 和 frames")
    
    # 执行导出
    try:
        torch.onnx.export(**export_kwargs)
        print(f"ONNX模型成功保存到 {output_path}")
        
        # 验证导出的模型
        verify_onnx_model(output_path)
    except Exception as e:
        print(f"导出ONNX模型时出错: {e}")


def verify_onnx_model(onnx_path):
    """验证导出的ONNX模型格式是否正确"""
    try:
        import onnx
        # 加载ONNX模型
        model = onnx.load(onnx_path)
        # 检查模型格式
        onnx.checker.check_model(model)
        print("ONNX模型验证成功，格式正确")
        
        # 打印模型信息
        print(f"模型输入: {[input.name for input in model.graph.input]}")
        print(f"模型输出: {[output.name for output in model.graph.output]}")
        print(f"模型包含 {len(model.graph.node)} 个节点")
    except ImportError:
        print("未安装ONNX库，跳过模型验证")
    except Exception as e:
        print(f"ONNX模型验证失败: {e}")


if __name__ == '__main__':
    model_path = "models_20251012001_ucf50/best_model_epoch242_aac0.9314.pth"
    output_path = "models_20251012001_ucf50/best_model_epoch242_aac0.9314.onnx"
    # UCF50数据集的50个动作类别名称 （根据实际数据集修改）
    class_names = ['BaseballPitch', 'Basketball', 'BenchPress', 'Biking', 'Billiards', 'BreastStroke', 'CleanAndJerk', 'Diving', 'Drumming', 'Fencing', 'GolfSwing', 'HighJump', 'HorseRace', 'HorseRiding', 'HulaHoop', 'JavelinThrow', 'JugglingBalls', 'JumpingJack', 'JumpRope', 'Kayaking', 'Lunges', 'MilitaryParade', 'Mixing', 'Nunchucks', 'PizzaTossing', 'PlayingGuitar', 'PlayingPiano', 'PlayingTabla', 'PlayingViolin', 'PoleVault', 'PommelHorse', 'PullUps', 'Punch', 'PushUps', 'RockClimbingIndoor', 'RopeClimbing', 'Rowing', 'SalsaSpin', 'SkateBoarding', 'Skiing', 'Skijet', 'SoccerJuggling', 'Swing', 'TaiChi', 'TennisSwing', 'ThrowDiscus', 'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog', 'YoYo']
    print(f"class_names: {len(class_names)}")
    num_classes = len(class_names)

    """
    
    mo --input_model models_20251012001_ucf50/best_model_epoch242_aac0.9314.onnx  --output_dir models_20251012001_ucf50/best_model_epoch242_aac0.9314_ov_model
    
    """

    # 提供命令行配置选项
    convert_to_onnx(
        model_path=model_path,
        output_path=output_path,
        num_classes=num_classes,
        dynamic_input=False
    )
    
    print("\n转换完成！")