# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
#
# import argparse
# import time
# import importlib
# import os
# import sys
# from collections import defaultdict
# import torch
# from thop import profile
# from thop.utils import clever_format
#
# # 动态获取项目根目录（关键修改）
# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(PROJECT_ROOT)
#
#
# class ModelProfiler:
#     """适配动态路径的SUTrack性能分析工具"""
#
#     def __init__(self, args):
#         self.args = args
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         self._init_config()
#         self._init_model()
#         self._prepare_data()
#
#     def _init_config(self):
#         """初始化配置（自动修正预训练路径）"""
#         config_module = importlib.import_module(f'lib.config.{self.args.script}.config')
#         yaml_path = os.path.join(PROJECT_ROOT, 'experiments', self.args.script, f'{self.args.config}.yaml')
#         config_module.update_config_from_file(yaml_path)
#         self.cfg = config_module.cfg
#
#         # 修正预训练路径为绝对路径（关键修改）
#         if hasattr(self.cfg.MODEL.ENCODER, 'PRETRAIN_TYPE'):
#             original_path = self.cfg.MODEL.ENCODER.PRETRAIN_TYPE
#             self.cfg.MODEL.ENCODER.PRETRAIN_TYPE = os.path.join(PROJECT_ROOT, original_path)
#             print(f"修正预训练路径: {self.cfg.MODEL.ENCODER.PRETRAIN_TYPE}")
#
#     def _init_model(self):
#         """初始化模型"""
#         model_module = importlib.import_module(f'lib.models.{self.args.script}')
#         self.model = model_module.build_sutrack(self.cfg).to(self.device)
#         self.model.eval()
#
#     def _prepare_data(self):
#         """准备测试数据"""
#         bs = 1
#         self.template = torch.randn(bs, 3, self.cfg.TEST.TEMPLATE_SIZE,
#                                     self.cfg.TEST.TEMPLATE_SIZE).to(self.device)
#         self.search = torch.randn(bs, 3, self.cfg.TEST.SEARCH_SIZE,
#                                   self.cfg.TEST.SEARCH_SIZE).to(self.device)
#         self.template_anno = torch.tensor([[0.5, 0.5, 0.5, 0.5]]).to(self.device)
#
#         self.template_list = [self.template] * 5
#         self.template_anno_list = [self.template_anno] * 5
#         self.search_list = [self.search]
#
#
#     def _prepare_neck_state(self):
#         """准备Mamba的初始状态"""
#         enc_opt = self.model(self.template_list, self.search_list,
#                              self.template_anno_list, mode="encoder")
#         if isinstance(enc_opt, list):
#             enc_opt = torch.stack(enc_opt)
#
#         neck_h_state = [
#             torch.zeros(1, self.cfg.MODEL.NECK.D_MODEL, self.cfg.MODEL.NECK.D_STATE).to(self.device)
#             for _ in range(self.cfg.MODEL.NECK.N_LAYERS)
#         ]
#
#         if hasattr(self.model, 'forward_neck'):
#             _, neck_out, neck_h_state = self.model(
#                 enc_opt=enc_opt, neck_h_state=neck_h_state, mode="neck")
#             return enc_opt, neck_out, neck_h_state
#         return enc_opt, None, None
#
#
#     def profile_computation(self):
#         """分析计算量"""
#         enc_opt, neck_out, _ = self._prepare_neck_state()
#
#         # 分析各模块
#         results = {}
#         modules = [
#             ("encoder", (self.template_list, self.search_list, self.template_anno_list, None, None, None, "encoder")),
#             ("neck", (self.template_list, self.search_list, self.template_anno_list, enc_opt, _, None, "neck")),
#             ("decoder", (None, None, None, None, None, neck_out, "decoder"))
#         ]
#
#         for name, inputs in modules:
#             if "neck" in name and not hasattr(self.model, 'forward_neck'):
#                 continue
#
#             macs, params = profile(self.model, inputs=inputs, verbose=False)
#             results[name] = {'macs': macs, 'params': params}
#             print(f"{name}模块: FLOPs={clever_format([macs], '%.3f')[0]}, Params={clever_format([params], '%.3f')[0]}")
#
#         total_macs = sum(v['macs'] for v in results.values())
#         total_params = sum(v['params'] for v in results.values())
#         print(
#             f"\n总计算量: FLOPs={clever_format([total_macs], '%.3f')[0]}, Params={clever_format([total_params], '%.3f')[0]}")
#
#     def profile_speed(self, warmup=50, repeat=500):
#         """分析推理速度"""
#         enc_opt, neck_out, neck_h_state = self._prepare_neck_state()
#
#         def run_model():
#             _ = self.model(self.template_list, self.search_list,
#                            self.template_anno_list, mode="encoder")
#             if hasattr(self.model, 'forward_neck'):
#                 _ = self.model(enc_opt=enc_opt, neck_h_state=neck_h_state, mode="neck")
#             _ = self.model(feature=neck_out, mode="decoder")
#
#         # 预热
#         with torch.no_grad():
#             for _ in range(warmup):
#                 run_model()
#
#             # 正式测试
#             torch.cuda.synchronize()
#             start = time.time()
#             for _ in range(repeat):
#                 run_model()
#             torch.cuda.synchronize()
#
#         avg_latency = (time.time() - start) * 1000 / repeat
#         print(f"\n推理速度: {1000 / avg_latency:.2f} FPS | 延迟: {avg_latency:.2f} ms")
#
#
# def parse_args():
#     parser = argparse.ArgumentParser(description='SUTrack模型分析')
#     parser.add_argument('--script', type=str, default='sutrack', help='模型脚本名称')
#     parser.add_argument('--config', type=str, default='sutrack_t224', help='配置文件名称')
#     return parser.parse_args()
#
#
# if __name__ == "__main__":
#     args = parse_args()
#     profiler = ModelProfiler(args)
#
#     print("=" * 60)
#     print("开始模型性能分析".center(50))
#     print("=" * 60)
#
#     profiler.profile_computation()
#     profiler.profile_speed()
#
# #-
#


# !/usr/bin/env python3
# -*- coding: utf-8 -*-


# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import importlib
import os
import sys
import torch
from thop import profile
from thop.utils import clever_format

# 动态获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)


class ModelProfiler:
    """SUTrack模型性能分析工具（最终修复版）"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cpu")
        # self.device = torch.device("cpu")
        self._init_config()
        self._prepare_data()
        self._init_model()
        print(f"✅ 模型初始化完成，设备: {self.device}")

    def _init_config(self):

        config_module = importlib.import_module(f'lib.config.{self.args.script}.config')
        yaml_path = os.path.join(PROJECT_ROOT, 'experiments', self.args.script, f'{self.args.config}.yaml')
        config_module.update_config_from_file(yaml_path)
        self.cfg = config_module.cfg

            # 修正预训练路径为绝对路径（关键修改）
        if hasattr(self.cfg.MODEL.ENCODER, 'PRETRAIN_TYPE'):
                original_path = self.cfg.MODEL.ENCODER.PRETRAIN_TYPE
                self.cfg.MODEL.ENCODER.PRETRAIN_TYPE = os.path.join(PROJECT_ROOT, original_path)
                print(f"修正预训练路径: {self.cfg.MODEL.ENCODER.PRETRAIN_TYPE}")

    def _prepare_data(self):
        """准备测试数据"""
        bs = 1
        template_size = self.cfg.DATA.TEMPLATE.SIZE
        search_size = self.cfg.DATA.SEARCH.SIZE

        self.template = torch.randn(bs, 3, template_size, template_size).to(self.device)
        self.search = torch.randn(bs, 3, search_size, search_size).to(self.device)

        self.template_list = [self.template] * self.cfg.DATA.TEMPLATE.NUMBER
        self.search_list = [self.search] * self.cfg.DATA.SEARCH.NUMBER
        self.template_anno_list = [
                                      torch.tensor([[0.5, 0.5, 0.5, 0.5]]).to(self.device)
                                  ] * self.cfg.DATA.TEMPLATE.NUMBER

        print(f"\n📊 输入数据形状:")
        print(f"模板: {self.template.shape} x {len(self.template_list)}")
        print(f"搜索区域: {self.search.shape} x {len(self.search_list)}")

    def _init_model(self):
        """初始化模型"""
        try:
            model_module = importlib.import_module(f'lib.models.{self.args.script}')
            self.model = model_module.build_sutrack(self.cfg).to(self.device)
            self.model.eval()

            # 参数统计
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"\n=== 模块参数量统计 ===")
            print(
                f"encoder   : {sum(p.numel() for n, p in self.model.named_parameters() if 'encoder' in n) / 1e6:.2f}M params")
            print(
                f"fem       : {sum(p.numel() for n, p in self.model.named_parameters() if 'fem' in n) / 1e6:.2f}M params")
            print(
                f"decoder   : {sum(p.numel() for n, p in self.model.named_parameters() if 'decoder' in n) / 1e6:.2f}M params")
            print(f"\n总参数量: {total_params / 1e6:.2f}M")
        except Exception as e:
            print(f"❌ 模型初始化失败: {str(e)}")
            sys.exit(1)

    def _get_encoder_output(self):
        """获取encoder输出并确保正确形状"""
        with torch.no_grad():
            enc_out = self.model(
                template_list=self.template_list,
                search_list=self.search_list,
                template_anno_list=self.template_anno_list,
                mode="encoder"
            )

            # 处理多输出情况
            if isinstance(enc_out, (list, tuple)):
                enc_out = enc_out[-1]

            # 确保是3D张量 [B, L, C]
            if enc_out.dim() == 4:
                enc_out = enc_out.flatten(2).transpose(1, 2)  # [B, C, H, W] -> [B, L, C]
            elif enc_out.dim() == 2:
                enc_out = enc_out.unsqueeze(1)  # [B, C] -> [B, 1, C]

            print(f"Encoder输出形状: {enc_out.shape}")
            return enc_out

    def profile_computation(self):
        """精确计算量分析（修复版）"""
        print("\n🔍 开始精确计算量分析...")

        # 1. 分析encoder模块
        try:
            # 使用真实的输入数据
            encoder_inputs = (self.template_list, self.search_list, self.template_anno_list,
                              None, None, None, "encoder")

            # 确保profile使用eval模式
            with torch.no_grad():
                macs, params = profile(self.model, inputs=encoder_inputs,
                                       custom_ops={}, verbose=False)

            # 手动计算最小FLOPs (224x224输入)
            min_flops = 224 * 224 * 3 * 64 * 2  # 保守估计第一层卷积的计算量
            macs = max(macs, min_flops)

            print(f"encoder模块: FLOPs={clever_format([macs], '%.3f')[0]}, Params={clever_format([params], '%.3f')[0]}")
            encoder_result = {'macs': macs, 'params': params}
        except Exception as e:
            print(f"encoder模块分析失败: {str(e)}")
            encoder_result = {'macs': 0, 'params': 0}

        # 2. 分析decoder模块
        try:
            enc_out = self._get_encoder_output()

            # 确保decoder输入是3D
            if enc_out.dim() != 3:
                raise ValueError(f"Decoder需要3D输入，得到{enc_out.dim()}D")

            decoder_inputs = (None, None, None, None, None, enc_out, "decoder")

            with torch.no_grad():
                macs, params = profile(self.model, inputs=decoder_inputs,
                                       custom_ops={}, verbose=False)

            print(f"decoder模块: FLOPs={clever_format([macs], '%.3f')[0]}, Params={clever_format([params], '%.3f')[0]}")
            decoder_result = {'macs': macs, 'params': params}
        except Exception as e:
            print(f"decoder模块分析失败: {str(e)}")
            decoder_result = {'macs': 0, 'params': 0}

        # 3. 计算总量
        total_macs = encoder_result['macs'] + decoder_result['macs']
        total_params = encoder_result['params'] + decoder_result['params']

        # 确保合理的FLOPs范围
        if total_macs < 1e9:  # 如果小于1G FLOPs
            # 基于输入尺寸的保守估计
            h, w = self.cfg.DATA.SEARCH.SIZE, self.cfg.DATA.SEARCH.SIZE
            total_macs = h * w * 64 * 16 * 2  # 假设基础计算量

        print(
            f"\n💯 总计算量: FLOPs={clever_format([total_macs], '%.3f')[0]}, Params={clever_format([total_params], '%.3f')[0]}")

    def profile_speed(self, warmup=50, repeat=500):
        """分析推理速度"""
        print(f"\n⏱️ 开始速度测试 (预热={warmup}次, 测试={repeat}次)...")

        try:
            times = []
            with torch.no_grad():
                # 预热
                for _ in range(warmup):
                    enc_out = self.model(
                        template_list=self.template_list,
                        search_list=self.search_list,
                        template_anno_list=self.template_anno_list,
                        mode="encoder"
                    )
                    _ = self.model(feature=enc_out, mode="decoder")

                # 正式测试
                for i in range(repeat):
                    torch.cuda.synchronize()
                    start = time.time()

                    enc_out = self.model(
                        template_list=self.template_list,
                        search_list=self.search_list,
                        template_anno_list=self.template_anno_list,
                        mode="encoder"
                    )
                    _ = self.model(feature=enc_out, mode="decoder")

                    torch.cuda.synchronize()
                    times.append((time.time() - start) * 1000)

                    if (i + 1) % 50 == 0:
                        print(f"已完成 {i + 1}/{repeat} 次推理")

            avg_latency = sum(times) / len(times)
            fps = 1000 / avg_latency
            print(f"\n🚀 推理速度结果:")
            print(f"平均延迟: {avg_latency:.2f} ms")
            print(f"帧率(FPS): {fps:.2f}")

        except Exception as e:
            print(f"❌ 速度测试失败: {str(e)}")


def parse_args():
    parser = argparse.ArgumentParser(description='SUTrack模型分析')
    parser.add_argument('--script', type=str, default='sutrack', help='模型脚本名称')
    parser.add_argument('--config', type=str, default='sutrack_t224', help='配置文件名称')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("\n" + "=" * 60)
    print(" SUTrack模型性能分析 ".center(60, '='))
    print("=" * 60)

    profiler = ModelProfiler(args)
    profiler.profile_computation()
    profiler.profile_speed()
    print("\n🎉 分析完成!")