import torch
import torch
import torch.nn as nn
# from nnunetv2.nets.AHDCNet import HDC_Net_D_A
# from nnunetv2.nets.TDPCNet import TDPC_Net
# # from nnunetv2.nets.LE_BTS import TDPC_Net
# from nnunetv2.nets.superlightnet import NormalU_Net
#
# from nnunetv2.nets.LWCTrans import LWCTrans
# from nnunetv2.nets.VT_Unet import SwinTransformerSys3D
# from nnunetv2.nets.UXNet.network_backbone import UXNET
# from nnunetv2.nets.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
# from nnunetv2.nets.SegFormer3D import SegFormer3D
# from nnunetv2.nets.SlimUNETR.SlimUNETR import SlimUNETR
# from nnunetv2.nets.AHDCNet import HDC_Net_D_A
# from nnunetv2.nets.swinunetr import SwinUNETR
# from monai.networks.nets import ViTAutoEnc
# from nnunetv2.nets.Unet import UNet

# from ADHDC_MSSC_invertRes_DWT import HDC_Net_D_A
# from AHDCNet_MSSC_invertRes import HDC_Net_D_A
# from AHDCNet_MSSC import HDC_Net_D_A
# from ADHDC_MSSC_invertRes_DWT_CorrT import HDC_Net_D_A
# from TDPC_pool import TDPC_Net
# from TDPC_avgpool import TDPC_Net
# from TDPC_LightTransformer_cosin import TDPC_Net
# from VT_Unet import SwinTransformerSys3D
# from SegFormer3D import SegFormer3D
# from UXNet.network_backbone import UXNET
# from TDPCNet import TDPC_Net
# from nnunetv2.nets.ADHDC_CSAFormer import HDC_Net_D_A
# from nnunetv2.nets.ADHDC_AMDA import HDC_Net_D_A
# from LWCTrans import LWCTrans
# from nnunetv2.nets.ADHDC_DWT import HDC_Net_D_A
# from AHDCNet import HDC_Net_D_A
# from TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
# from nnunetv2.nets.superlightnet import NormalU_Net
# from nnunetv2.nets.Unet_invertRes_AMDA import UNet3D_Constant32
# from nnunetv2.nets.Unet_next import UNet3D_Constant32
# from nnunetv2.nets.Unet_MSSC import UNet3D_Constant32
# from nnunetv2.nets.Unet_DyHWT import UNet3D_Constant32
# from nnunetv2.nets.Unet_CSAFormer import UNet3D_Constant32
# from nnunetv2.nets.Unet_DyHWT_CSAFormer  import UNet3D_Constant32
# from nnunetv2.nets.LE_BTS_HDC_DyFS_Conv_ablation import LE_BTS
# from nnunetv2.nets.LE_BTS_HDC_Conv import LE_BTS_HDC_Conv
# from nnunetv2.nets.LE_BTS import TDPC_Net
# from nnunetv2.nets.LE_BTS_noDyFS import LE_BTS
# from nnunetv2.nets.LEBTS64 import TDPC_Net
# from nnunetv2.nets.SlimUNETR.SlimUNETR import SlimUNETR
# from nnunetv2.nets.Unet_invertRes_noAMDA import UNet3D_Constant32
# from nnunetv2.nets.Unet_next import UNet3D_Constant32
# from nnunetv2.nets.Unet_MSSC import UNet3D_Constant32
# from nnunetv2.nets.Unet_DyHWT import UNet3D_Constant32
# from nnunetv2.nets.Unet_CSAFormer import UNet3D_Constant32
# from nnunetv2.nets.Unet_MSSC_DyHWT import UNet3D_Constant32
# from nnunetv2.nets.Unet_MSSC_CSAFormer import UNet3D_Constant32
from nnunetv2.nets.Unet_DyHWT_CSAFormer  import UNet3D_Constant32
# from nnunetv2.nets.LEBTS_dyhwt_gpu import LE_BTS
# from nnunetv2.nets.LEBTS_MSSC_ablation import LE_BTS_Ablation
import torch
# from nnunetv2.nets.LE_BTS0923_StageWiseMSCG import LE_BTS
# from nnunetv2.nets.Unet_invertRes_noAMDA import UNet3D_Constant32
# from nnunetv2.nets.LEBTS_MSCG_nofusion import LE_BTS
# ============================================================
# 1. Import your model
# ============================================================
# from nnunetv2.nets.TDPCNet import TDPC_Net


# ============================================================
# 2. Parameter counting using native PyTorch
# ============================================================
def count_parameters(model):
    trainable_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    return trainable_params, total_params


# ============================================================
# 3. FLOPs calculation using fvcore
# ============================================================
def count_flops(model, x):
    from fvcore.nn import FlopCountAnalysis

    model.eval()

    with torch.no_grad():
        flops = FlopCountAnalysis(model, x)

    total_flops = flops.total()

    return total_flops, flops


# ============================================================
# 4. Main
# ============================================================
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("Model Complexity Test")
    print("=" * 70)

    print(f"Device: {device}")

    # --------------------------------------------------------
    # Input size
    # --------------------------------------------------------
    batch_size = 1
    channels = 4
    depth = 128
    height = 128
    width = 128

    x = torch.randn(
        batch_size,
        channels,
        depth,
        height,
        width
    ).to(device)

    print(
        f"Input shape: "
        f"{batch_size} × {channels} × "
        f"{depth} × {height} × {width}"
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------
    # model = LE_BTS(
    #     4,      # input channels
    #     3,      # output channels
    #     32,     # base channels
    # ).to(device)
    model = UNet3D_Constant32(in_channels=4, num_classes=3).to(device)
    # model = SlimUNETR(
    #     in_channels=4,
    #     out_channels=3,
    #     embed_dim=96,
    #     embedding_dim=64,
    #     channels=(24, 48, 60),
    #     blocks=(1, 2, 3, 3),
    #     heads=(1, 2, 4, 4),
    #     r=(4, 2, 2, 1),
    #     # distillation=False,
    #     # dropout=0.3,
    # ).to(device)

    # model = HDC_Net_D_A().to(device)
    # model = TDPC_Net().to(device)
    # model = NormalU_Net(depths_unidirectional='small', class_nums=3).to(device)
    # model = LWCTrans(4, 3, conv_kernel_size=((3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)),
    #                   pool_op_kernel_sizes=((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 1))).to(device)
    # model = SegFormer3D().to(device)
    # model = SlimUNETR(
    #     in_channels=4,
    #     out_channels=3,
    #     embed_dim=96,
    #     embedding_dim=128,
    #     channels=(24, 48, 60),
    #     blocks=(1, 2, 3, 2),
    #     heads=(1, 2, 4, 4),
    #     r=(4, 2, 2, 1),
    #     # distillation=False,
    #     # dropout=0.3,
    # ).to(device)

    # model = UNet(4, 3, channels_peizhi, deep_supervision=False, ds_layer=4, **blocks_kwargs).to(self.device)
    # print(self.num_input_channels)
    # print(len(self.label_manager.all_labels))
    # self.network = UNet3D_Constant32(self.num_input_channels,3,32).to(self.device)
    # self.network = create_model(self.num_input_channels,len(self.label_manager.all_labels),32,name='TDPCNet').to(self.device)

    # self.network = self.build_network_architecture(self.plans_manager, self.dataset_json,
    #                                                self.configuration_manager,
    #                                                self.num_input_channels,
    #                                                enable_deep_supervision=False).to(self.device)
    # self.network = TDPC_Net(self.num_input_channels, 3, 48).to(self.device)
    # self.network = TDPC_Net(self.num_input_channels, 3,32).to(self.device)
    # model = SwinUNETR(
    #     img_size=(128, 128, 128),
    #     num_heads= (2, 4, 8, 8),
    #     in_channels=4,
    #     out_channels=3,
    #     drop_rate = 0.2,
    #     attn_drop_rate = 0.2,
    #     dropout_path_rate = 0.2,
    #     use_checkpoint = False).to(device)
    # self.network = NormalU_Net(depths_unidirectional='small',class_nums=3).to(self.device)
    # self.network = ViTWrapper(ViTAutoEnc(img_size=(160, 160, 160),patch_size=16,in_channels=4,out_channels=3,num_layers=11,num_heads=16,
    #     hidden_size=512,
    #     mlp_dim=2048
    #     )).to(self.device)
    # self.network = HDC_Net_D_A().to(self.device) #消融实验
    # model = UNETR_PP(in_channels=4,
    # out_channels=3,
    # feature_size= 16,
    # hidden_size= 256,
    # num_heads= 4,
    # pos_embed = "perceptron",
    # norm_name = "instance",
    # dropout_rate = 0.0,
    # depths=[3, 3, 3, 3],
    # dims=[32, 64, 128, 256],
    # conv_op=nn.Conv3d,
    # do_ds=False).to(device) #不使用深度监督
    # self.network = LWCTrans(self.num_input_channels, 3,
    #                         conv_kernel_size=((3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)),
    #                         pool_op_kernel_sizes=((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 1))
    #                         ).to(self.device)
    # self.network = SwinTransformerSys3D(
    #     img_size=(128, 128, 128),
    #     patch_size=(4, 4, 4),
    #     in_chans=self.num_input_channels,
    #     num_classes=3,
    #     embed_dim=96,
    #     depths=[2, 2, 2, 1],
    #     depths_decoder=[1, 2, 2, 2],
    #     num_heads=[3, 6, 12, 24],
    #     window_size=(7, 7, 7),
    #     mlp_ratio=4.,
    #     qkv_bias=True,
    #     qk_scale=None,
    #     drop_rate=0.,
    #     attn_drop_rate=0.,
    #     drop_path_rate=0.1,
    #     norm_layer=nn.LayerNorm,
    #     patch_norm=True,
    #     use_checkpoint=False,
    #     frozen_stages=-1,
    #     final_upsample="expand_first",
    # ).to(self.device)
    # self.network = UXNET(out_chans=5).to(self.device)
    # self.network= TransBTS(dataset='brats', _conv_repr=True, _pe_type="learned").to(self.device)
    # self.network = SegFormer3D().to(self.device)
    # model= SlimUNETR(
    #     in_channels=4,
    #     out_channels=3,
    #     embed_dim=96,
    #     embedding_dim=128,
    #     channels=(24, 48, 60),
    #     blocks=(1, 2, 3, 2),
    #     heads=(1, 2, 4, 4),
    #     r=(4, 2, 2, 1),
    #     # distillation=False,
    #     # dropout=0.3,
    # ).to(device)
    # model = LE_BTS_HDC_Conv(4,3,32).to(device)
    # model = TransBTS(dataset='brats', _conv_repr=True, _pe_type="learned").to(device)
    # model= LWCTrans(4, 3,
    #                         conv_kernel_size=((3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)),
    #                         pool_op_kernel_sizes=((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 1))
    #                         ).to(device)
    # model = SwinTransformerSys3D().to(device)
    # model = SegFormer3D(num_classes=5).to(device)
    # model = UXNET(out_chans=5).to(device)

    model.eval()

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------
    trainable_params, total_params = count_parameters(model)

    print("\n" + "-" * 70)
    print("Parameter Count")
    print("-" * 70)

    print(f"Trainable parameters : {trainable_params:,}")
    print(f"Total parameters     : {total_params:,}")

    print(
        f"Trainable parameters : "
        f"{trainable_params / 1e6:.6f} M"
    )

    print(
        f"Total parameters     : "
        f"{total_params / 1e6:.6f} M"
    )

    # --------------------------------------------------------
    # FLOPs
    # --------------------------------------------------------
    print("\n" + "-" * 70)
    print("FLOPs")
    print("-" * 70)

    total_flops, flop_analysis = count_flops(model, x)

    print(f"Total FLOPs : {total_flops:,}")
    print(f"FLOPs       : {total_flops / 1e9:.6f} G")

    # --------------------------------------------------------
    # Optional: FLOPs by operator
    # --------------------------------------------------------
    print("\n" + "-" * 70)
    print("FLOPs by operator")
    print("-" * 70)

    try:
        print(flop_analysis.by_operator())
    except Exception as e:
        print(f"Unable to print operator statistics: {e}")

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    print(
        f"Trainable Params : "
        f"{trainable_params / 1e6:.3f} M"
    )

    print(
        f"Total Params     : "
        f"{total_params / 1e6:.3f} M"
    )

    print(
        f"FLOPs            : "
        f"{total_flops / 1e9:.2f} G"
    )

    print("=" * 70)







