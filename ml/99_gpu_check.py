import torch

if torch.cuda.is_available():
    print("GPUが使用可能です。")
    num_devices = torch.cuda.device_count()
    print(f"利用可能なGPUの数: {num_devices}")
    for i in range(num_devices):
        print("-" * 40)
        print(f"GPU {i}:")
        # デバイス名
        device_name = torch.cuda.get_device_name(i)
        print(f"  デバイス名: {device_name}")
        # GPUの詳細なプロパティ
        props = torch.cuda.get_device_properties(i)
        print(f"  コンピュート・キャパビリティ: {props.major}.{props.minor}")
        # 総メモリ (GB単位に変換)
        total_memory = props.total_memory / (1024**3)
        print(f"  総メモリ: {total_memory:.2f} GB")
        # SM（Streaming Multiprocessor）の数
        print(f"  マルチプロセッサ数: {props.multi_processor_count}")
        # その他、props 内にはクロック速度やメモリバス幅などの情報も含まれます。
else:
    print("GPUが使用できません。")
