CONFIG = {
    'kill_action': 60,  # 吃子达到设定，则平局
    'dirichlet': 0.15,  #alphazero论文，走子位置越多，数值越小  # 国际象棋，0.3；日本将棋，0.15；围棋，0.03
    'play_out': 2000,  # 每次移动的模拟次数
    'c_puct': 5,  # puct 中u的权重
    'buffer_size': 100000, # 经验池大小，与内存有关
    'pytorch_model_path': 'current_policy.pkl',   # 模型路径
    'use_frame': 'pytorch',  # paddle or pytorch根据自己的环境进行切换
    'train_data_buffer_path': 'train_data_buffer.pkl',   # 数据容器的路径
    'batch_size': 512,  # 训练的batch大小,根据显存
    'kl_trag': 0.02,   # kl散度控制
    'epochs': 5,  # 每次更新的train_step数量
    'game_batch_num': 3000,  # 训练更新的次数


}
