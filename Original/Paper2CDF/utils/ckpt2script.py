from model.df_net import DFNet, CDFNet
import torch
from utils.config_loader import CFG


cfg_file_path = "../experiments/df_L50.yml"
cfg = CFG.read_from_yaml(cfg_file_path)
print("Load configuration from {}".format(cfg_file_path))
model = CDFNet(cfg)
model = model.to(torch.device("cuda"))
# if cfg.general.pretrained_weight:
#     pretrained_model_path = cfg.general.pretrained_weight
#     print("load weight from:{}".format(pretrained_model_path))
#     snapshot = torch.load(pretrained_model_path)
#     model.load_state_dict(snapshot.pop("model"), strict=False)
# else:
# print("do not use pretrained weight")
model.init_weights()

# device = torch.device("cuda:0")
model = model.cuda()
model.eval()
torch.backends.cudnn.benchmark = True
example_rgb = torch.ones((1, 3, 224, 224)).cuda()
example_mask = torch.ones((1, 1, 224, 224)).cuda()
R = torch.ones((1, 3, 3)).cuda()
t = torch.ones((1, 3, 1)).cuda()
# export_module = torch.jit.script(model)
# torch.jit.save(export_module, "../pretrained/df_L50_cpp_gpu.pth")
example_scale = torch.ones((1, 1)).cuda()
example_r = torch.Tensor([1.0]).cuda()
traced_script_module = torch.jit.trace(model, example_inputs=(example_rgb, example_rgb,
                                                              R, t,
                                                              example_r, example_r, example_r, example_r, example_r))
traced_script_module.save("../pretrained/df_L50_cpp_gpu.pth")

