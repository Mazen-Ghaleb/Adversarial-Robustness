from torch.nn import Module, BatchNorm2d, ReLU, Conv2d, Sequential, ConvTranspose2d
from torch.nn import ModuleDict, AvgPool2d
import torch.nn.functional as F
from torch import Tensor
from collections import OrderedDict
import torch


class DenseLayer(Module):
    def __init__(self, num_input_features, growth_rate, bn_size) -> None:
        super(DenseLayer, self).__init__()

        # bottle neck
        self.layers = Sequential(
            BatchNorm2d(num_input_features),
            ReLU(inplace=True),
            Conv2d(
                in_channels=num_input_features,
                out_channels=bn_size * growth_rate,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False),
            BatchNorm2d(bn_size * growth_rate),
            ReLU(inplace=True),
            Conv2d(
                in_channels=growth_rate * bn_size,
                out_channels=growth_rate,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False)
        )

    def forward(self, x):
        if isinstance(x, Tensor):
            prev_features = torch.concat([x], 1)
        else:
            prev_features = torch.concat(x, 1)

        out = self.layers(prev_features)
        return out


class DenseBlock(ModuleDict):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate) -> None:
        super(DenseBlock, self).__init__()

        for i in range(num_layers):
            layer = DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.concat(features, 1)


class Transition(Sequential):
    def __init__(self, num_input_features, num_output_features, kernel_size=2, stride=2, padding=0):
        super(Transition, self).__init__()
        self.add_module('norm', BatchNorm2d(num_input_features))
        self.add_module('relu', ReLU(inplace=True))
        self.add_module('conv', Conv2d(num_input_features, num_output_features,
                                       kernel_size=kernel_size, stride=stride,padding=1,bias=False))
        # self.add_module('pool', Conv2d(kernel_size=2, stride=2))

class Fuse(Module):
    def __init__(self) -> None:
        super(Fuse, self).__init__()

    def forward(self, small_image, large_image):
        upscaled_image = F.interpolate(small_image, size=large_image.shape[2:], mode="bilinear")
        result_image = torch.cat((upscaled_image, large_image), dim=1)
        return result_image 

class HGD(Module):
    def __init__(
            self,
            width = 1.0,
            growth_rate=16,
            bn_size=2,
            ) -> None:
        
        super(HGD, self).__init__()
        start_channels = int(width * 64)
        self.stem = Sequential(
            BatchNorm2d(3),
            ReLU(inplace=True),
            Conv2d(in_channels=3, out_channels=start_channels,
                    kernel_size=7, stride=2, padding=3, bias=False),
            BatchNorm2d(start_channels),
            ReLU(inplace=True),
            Conv2d(in_channels=start_channels, out_channels=start_channels,
                    kernel_size=3, stride=2, padding=1, bias=False),
        )


        self.reverse_stem = Sequential(
            BatchNorm2d(start_channels),
            ReLU(inplace=True),
            ConvTranspose2d(in_channels=start_channels, out_channels=start_channels,
                   kernel_size=4, stride=2, padding=1,bias=False),

            BatchNorm2d(start_channels),
            ReLU(inplace=True),
            ConvTranspose2d(in_channels=start_channels, out_channels=start_channels,
                   kernel_size=4, stride=2, padding=1, bias=False),
        )

        self.fuse = Fuse()

        self.conv = Conv2d(in_channels=start_channels, out_channels=3, kernel_size=3, padding=1, stride=1)

        forward_path_info={
            "num_layers": [2, 3, 3, 3, 3],
            "layers_inputs": list(map(lambda x: int(x * width), [64, 64, 128, 256, 256])),
            "layers_outputs": list(map(lambda x: int(x * width), [64, 128, 256, 256, 256])), 
        }
        backward_path_info={
            "num_layers": [3, 3, 3, 2],
            "layers_inputs": list(map(lambda x: int(x * width),  [512, 512, 384, 192])),
            "layers_outputs": list(map(lambda x: int(x * width), [256, 256, 128, 64]))
        }


        for i, (num_layers, inp, out) in enumerate(
            zip(forward_path_info["num_layers"],
                forward_path_info["layers_inputs"],
                forward_path_info["layers_outputs"])):
            dense_block = DenseBlock(num_layers, inp, bn_size, growth_rate)
            if i == 0:
                kernel_size = 3
                padding = 1
                stride = 1
            else:
                kernel_size = 2
                padding = 2
                stride = 2

            transition = Transition(
                inp + num_layers * growth_rate, out,
                  padding=padding, kernel_size=kernel_size, stride=stride)
            self.add_module(f"forward_{i}", dense_block)
            self.add_module(f"forward_transition_{i}", transition)

        for i, (num_layers, inp, out) in enumerate(
            zip(backward_path_info["num_layers"],
                backward_path_info["layers_inputs"],
                backward_path_info["layers_outputs"])):

            if i == len(backward_path_info["num_layers"]) - 1:
                kernel_size = 3
                padding = 1
                stride = 1
            else:
                kernel_size = 2
                padding = 2
                stride = 2


            dense_block = DenseBlock(num_layers, inp, bn_size, growth_rate)
            transition = Transition(
                inp + num_layers * growth_rate, out,
                  padding=padding, kernel_size=kernel_size, stride=stride)
            self.add_module(f"backward_{i}", dense_block)
            self.add_module(f"backward_transition_{i}", transition)
    def forward(self, input):
        #backward path
        stem_out = self.stem(input)
        # print(stem_out.shape)
        out_forward = []
        out_forward.append(self.forward_transition_0(self.forward_0(stem_out)))
        out_forward.append(self.forward_transition_1(self.forward_1(out_forward[0])))
        out_forward.append(self.forward_transition_2(self.forward_2(out_forward[1])))
        out_forward.append(self.forward_transition_3(self.forward_3(out_forward[2])))
        out_forward.append(self.forward_transition_4(self.forward_4(out_forward[3])))
        # for out in out_forward: 
        #     print(out.shape)


        out_backward = self.fuse(out_forward[4], out_forward[3])
        # print(out_backward.shape)
        out_backward = self.backward_transition_0(self.backward_0(out_backward))
        # print(out_backward.shape, out_forward[2].shape, out_forward[3].shape)

        out_backward = self.fuse(out_backward, out_forward[2])
        # print(out_backward.shape)
        out_backward = self.backward_transition_1(self.backward_1(out_backward))
        # print(out_backward.shape)

        out_backward = self.fuse(out_backward, out_forward[1])
        # print(out_backward.shape)
        out_backward = self.backward_transition_2(self.backward_2(out_backward))

        out_backward = self.fuse(out_backward, out_forward[0])
        # print(out_backward.shape)
        out_backward = self.backward_transition_3(self.backward_3(out_backward))
        # print(out_backward.shape)

        out_backward = self.reverse_stem(out_backward)
        out_backward = self.conv(out_backward)


        return out_backward






if __name__ == "__main__":
    from torchsummary import summary
    model = HGD(width=0.5)
    input = torch.randn((1, 3, 640, 640))
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(count_parameters(model))

    print(model(input).shape)



