import torch
import torch.nn.functional as F


from .kernels import gaussian_multiple_channels


class REKD(torch.nn.Module):
    def __init__(self, args, device):
        super(REKD, self).__init__()
        from e2cnn import gspaces
        from e2cnn import nn

        self.pyramid_levels = 3
        self.factor_scaling = args.factor_scaling_pyramid

        # Smooth Gausian Filter
        num_channels = 1  ## gray scale image
        self.gaussian_avg = gaussian_multiple_channels(num_channels, 1.5)

        r2_act = gspaces.Rot2dOnR2(N=args.group_size)

        self.feat_type_in = nn.FieldType(
            r2_act, num_channels * [r2_act.trivial_repr]
        )  ## input 1 channels (gray scale image)

        feat_type_out1 = nn.FieldType(r2_act, args.dim_first * [r2_act.regular_repr])
        feat_type_out2 = nn.FieldType(r2_act, args.dim_second * [r2_act.regular_repr])
        feat_type_out3 = nn.FieldType(r2_act, args.dim_third * [r2_act.regular_repr])

        feat_type_ori_est = nn.FieldType(r2_act, [r2_act.regular_repr])

        self.block1 = nn.SequentialModule(
            nn.R2Conv(
                self.feat_type_in, feat_type_out1, kernel_size=5, padding=2, bias=False
            ),
            nn.InnerBatchNorm(feat_type_out1),
            nn.ReLU(feat_type_out1, inplace=True),
        )
        self.block2 = nn.SequentialModule(
            nn.R2Conv(
                feat_type_out1, feat_type_out2, kernel_size=5, padding=2, bias=False
            ),
            nn.InnerBatchNorm(feat_type_out2),
            nn.ReLU(feat_type_out2, inplace=True),
        )
        self.block3 = nn.SequentialModule(
            nn.R2Conv(
                feat_type_out2, feat_type_out3, kernel_size=5, padding=2, bias=False
            ),
            nn.InnerBatchNorm(feat_type_out3),
            nn.ReLU(feat_type_out3, inplace=True),
        )

        self.ori_learner = nn.SequentialModule(
            nn.R2Conv(
                feat_type_out3, feat_type_ori_est, kernel_size=1, padding=0, bias=False
            )  ## Channel pooling by 8*G -> 1*G conv.
        )
        self.softmax = torch.nn.Softmax(dim=1)

        self.gpool = nn.GroupPooling(feat_type_out3)
        self.last_layer_learner = torch.nn.Sequential(
            torch.nn.BatchNorm2d(num_features=args.dim_third * self.pyramid_levels),
            torch.nn.Conv2d(
                in_channels=args.dim_third * self.pyramid_levels,
                out_channels=1,
                kernel_size=1,
                bias=True,
            ),
            torch.nn.ReLU(inplace=True),  ## clamp to make the scores positive values.
        )

        self.dim_third = args.dim_third
        self.group_size = args.group_size
        self.exported = False

    def export(self):
        from e2cnn import nn

        for name, module in dict(self.named_modules()).copy().items():
            if isinstance(module, nn.EquivariantModule):
                # print(name, "--->", module)
                module = module.export()
                setattr(self, name, module)

        self.exported = True

    def forward(self, input_data):
        features_key, features_o = self.compute_features(input_data)

        return features_key, features_o

    def compute_features(self, input_data):
        B, _, H, W = input_data.shape

        for idx_level in range(self.pyramid_levels):
            with torch.no_grad():
                input_data_resized = self._resize_input_image(
                    input_data, idx_level, H, W
                )

            if H > 2500 or W > 2500:
                features_t, features_o = self._forwarding_networks_divide_grid(
                    input_data_resized
                )
            else:
                features_t, features_o = self._forwarding_networks(input_data_resized)

            features_t = F.interpolate(
                features_t, size=(H, W), align_corners=True, mode="bilinear"
            )
            features_o = F.interpolate(
                features_o, size=(H, W), align_corners=True, mode="bilinear"
            )

            if idx_level == 0:
                features_key = features_t
                features_ori = features_o
            else:
                features_key = torch.cat([features_key, features_t], axis=1)
                features_ori = torch.add(features_ori, features_o)

        features_key = self.last_layer_learner(features_key)
        features_ori = self.softmax(features_ori)

        return features_key, features_ori

    def _forwarding_networks(self, input_data_resized):
        from e2cnn import nn

        # wrap the input tensor in a GeometricTensor (associate it with the input type)
        features_t = (
            nn.GeometricTensor(input_data_resized, self.feat_type_in)
            if not self.exported
            else input_data_resized
        )

        ## Geometric tensor feed forwarding
        features_t = self.block1(features_t)
        features_t = self.block2(features_t)
        features_t = self.block3(features_t)

        ## orientation pooling
        features_o = self.ori_learner(features_t)  ## self.cpool
        features_o = features_o.tensor if not self.exported else features_o

        ## keypoint pooling
        features_t = self.gpool(features_t)
        features_t = features_t.tensor if not self.exported else features_t

        return features_t, features_o

    def _forwarding_networks_divide_grid(self, input_data_resized):
        ## for inference time high resolution image. # spatial grid 4
        B, _, H_resized, W_resized = input_data_resized.shape
        features_t = torch.zeros(B, self.dim_third, H_resized, W_resized).cuda()
        features_o = torch.zeros(B, self.group_size, H_resized, W_resized).cuda()
        h_divide = 2
        w_divide = 2
        for idx in range(h_divide):
            for jdx in range(w_divide):
                ## compute the start and end spatial index
                h_start = H_resized // h_divide * idx
                w_start = W_resized // w_divide * jdx
                h_end = H_resized // h_divide * (idx + 1)
                w_end = W_resized // w_divide * (jdx + 1)
                ## crop the input image
                input_data_divided = input_data_resized[
                    :, :, h_start:h_end, w_start:w_end
                ]
                features_t_temp, features_o_temp = self._forwarding_networks(
                    input_data_divided
                )
                ## take into the values.
                features_t[:, :, h_start:h_end, w_start:w_end] = features_t_temp
                features_o[:, :, h_start:h_end, w_start:w_end] = features_o_temp

        return features_t, features_o

    def _resize_input_image(self, input_data, idx_level, H, W):
        if idx_level == 0:
            input_data_smooth = input_data
        else:
            ## (7,7) size gaussian kernel.
            input_data_smooth = F.conv2d(
                input_data, self.gaussian_avg.to(input_data.device), padding=[3, 3]
            )

        target_resize = (
            int(H / (self.factor_scaling**idx_level)),
            int(W / (self.factor_scaling**idx_level)),
        )

        input_data_resized = F.interpolate(
            input_data_smooth, size=target_resize, align_corners=True, mode="bilinear"
        )

        input_data_resized = self.local_norm_image(input_data_resized)

        return input_data_resized

    def local_norm_image(self, x, k_size=65, eps=1e-10):
        pad = int(k_size / 2)

        x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        x_mean = F.avg_pool2d(
            x_pad, kernel_size=[k_size, k_size], stride=[1, 1], padding=0
        )  ## padding='valid'==0
        x2_mean = F.avg_pool2d(
            torch.pow(x_pad, 2.0),
            kernel_size=[k_size, k_size],
            stride=[1, 1],
            padding=0,
        )

        x_std = torch.sqrt(torch.abs(x2_mean - x_mean * x_mean)) + eps
        x_norm = (x - x_mean) / (1.0 + x_std)

        return x_norm


def count_model_parameters(model):
    ## Count the number of learnable parameters.
    print("================ List of Learnable model parameters ================ ")
    for n, p in model.named_parameters():
        if p.requires_grad:
            print("{} {}".format(n, p.data.shape))
        else:
            print("\n\n\n None learnable params {} {}".format(n, p.data.shape))
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([torch.prod(torch.tensor(p.size())) for p in model_parameters])
    print("The number of learnable parameters : {} ".format(params.data))
    print("==================================================================== ")
