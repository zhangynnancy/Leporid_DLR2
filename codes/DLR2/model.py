import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable

torch.manual_seed(123)
activation = 'relu'


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class GeneralBlock2(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes):
        super(GeneralBlock2, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.fc1 = nn.Linear(inplanes, 128)
        # self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, planes)
        # self.bn2 = norm_layer(planes)

    def forward(self, x):

        if len(x.shape) > 2:
            x = torch.flatten(x, 1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)

        return out


class GeneralBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(GeneralBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        # self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        # self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        # self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        # self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class RevBlock_yinan(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(RevBlock_yinan, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        # self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.conv3 = conv3x3(inplanes, planes, stride)
        self.conv4 = conv3x3(planes, planes)
        # self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.inplanes = inplanes
        self.planes = planes

    def block_func1(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        return out

    def block_func2(self, x):
        out = self.conv3(x)
        out = self.relu(out)
        out = self.conv4(out)

        return out

    def forward(self, x):
        # print('x = ', x.shape)
        x1, x2 = torch.chunk(x, 2, dim=1)
        # print('x1 = ', x1.shape)
        # print('x2 = ', x2.shape)

        x1 = Variable(x1.contiguous())
        x2 = Variable(x2.contiguous())

        if torch.cuda.is_available():
            x1.cuda()
            x2.cuda()

        x1_ = possible_downsample(x1, self.inplanes, self.planes, self.stride, padding=1, dilation=1)
        x2_ = possible_downsample(x2, self.inplanes, self.planes, self.stride, padding=1, dilation=1)
        # print('x1_ = ', x1_.shape)
        # print('x2_ = ', x2_.shape)

        f_x2 = self.block_func1(x2_)
        y1 = f_x2 + x1_

        g_y1 = self.block_func2(y1)
        y2 = g_y1 + x2_

        y = torch.cat([y1, y2], dim=1)

        return y


def size_after_residual(size, out_channels, kernel_size, stride, padding, dilation):
    """Calculate the size of the output of the residual function
    """
    N, C_in, H_in, W_in = size

    H_out = math.floor(
        (H_in + 2*padding - dilation*(kernel_size - 1) - 1) / stride + 1
    )
    W_out = math.floor(
        (W_in + 2*padding - dilation*(kernel_size - 1) - 1) / stride + 1
    )
    return N, out_channels, H_out, W_out


def possible_downsample(x, in_channels, out_channels, stride=1, padding=1,
                        dilation=1):
    _, _, H_in, W_in = x.size()

    _, _, H_out, W_out = size_after_residual(x.size(), out_channels, 3, stride, padding, dilation)

    # Downsample image
    if H_in > H_out or W_in > W_out:
        out = F.avg_pool2d(x, 2*dilation+1, stride, padding)

    # Pad with empty channels
    if in_channels < out_channels:

        try: out
        except: out = x

        pad = Variable(torch.zeros(
            out.size(0),
            (out_channels - in_channels) // 2,
            out.size(2), out.size(3)
        ), requires_grad=True)

        if torch.cuda.is_available():
            pad = pad.cuda()

        temp = torch.cat([pad, out], dim=1)
        out = torch.cat([temp, pad], dim=1)

    # If we did nothing, add zero tensor, so the output of this function
    # depends on the input in the graph
    try: out
    except:
        injection = Variable(torch.zeros_like(x.data), requires_grad=True)

        if torch.cuda.is_available():
            injection.cuda()

        out = x + injection

    return out


class RevBlockFunction(Function):
    @staticmethod
    def residual(x, in_channels, out_channels, params, buffers, training,
                 stride=1, padding=1, dilation=1, no_activation=False):
        """Compute a pre-activation residual function.
        Args:
            x (Variable): The input variable
            in_channels (int): Number of channels of x
            out_channels (int): Number of channels of the output
        Returns:
            out (Variable): The result of the computation
        """
        out = x

        # if not no_activation:
        #     out = F.batch_norm(out, buffers[0], buffers[1], params[0],
        #                        params[1], training)
        #     out = F.relu(out)

        out = F.conv2d(out, params[-6], params[-5], stride, padding=padding,
                       dilation=dilation)

        # out = F.batch_norm(out, buffers[-2], buffers[-1], params[-4],
        #                    params[-3], training)
        out = F.relu(out)
        out = F.conv2d(out, params[-2], params[-1], stride=1, padding=1,
                       dilation=1)

        return out

    @staticmethod
    def _forward(x, in_channels, out_channels, training, stride, padding,
                 dilation, f_params, f_buffs, g_params, g_buffs,
                 no_activation=False):

        x1, x2 = torch.chunk(x, 2, dim=1)

        with torch.no_grad():
            x1 = Variable(x1.contiguous())
            x2 = Variable(x2.contiguous())

            if torch.cuda.is_available():
                x1.cuda()
                x2.cuda()

            x1_ = possible_downsample(x1, in_channels, out_channels, stride,
                                      padding, dilation)
            x2_ = possible_downsample(x2, in_channels, out_channels, stride,
                                      padding, dilation)

            f_x2 = RevBlockFunction.residual(
                x2,
                in_channels,
                out_channels,
                f_params,
                f_buffs, training,
                stride=stride,
                padding=padding,
                dilation=dilation,
                no_activation=no_activation
            )

            y1 = f_x2 + x1_

            g_y1 = RevBlockFunction.residual(
                y1,
                out_channels,
                out_channels,
                g_params,
                g_buffs,
                training
            )

            y2 = g_y1 + x2_

            y = torch.cat([y1, y2], dim=1)

            del y1, y2
            del x1, x2

        return y

    # @staticmethod
    # def _backward(output, in_channels, out_channels, f_params, f_buffs,
    #               g_params, g_buffs, training, padding, dilation, no_activation):
    #
    #     y1, y2 = torch.chunk(output, 2, dim=1)
    #     with torch.no_grad():
    #         y1 = Variable(y1.contiguous())
    #         y2 = Variable(y2.contiguous())
    #
    #         x2 = y2 - RevBlockFunction.residual(
    #             y1,
    #             out_channels,
    #             out_channels,
    #             g_params,
    #             g_buffs,
    #             training=training
    #         )
    #
    #         x1 = y1 - RevBlockFunction.residual(
    #             x2,
    #             in_channels,
    #             out_channels,
    #             f_params,
    #             f_buffs,
    #             training=training,
    #             padding=padding,
    #             dilation=dilation
    #         )
    #
    #         del y1, y2
    #         x1, x2 = x1.data, x2.data
    #
    #         x = torch.cat((x1, x2), 1)
    #     return x
    #
    # @staticmethod
    # def _grad(x, dy, in_channels, out_channels, training, stride, padding,
    #           dilation, activations, f_params, f_buffs, g_params, g_buffs,
    #           no_activation=False, storage_hooks=[]):
    #     dy1, dy2 = torch.chunk(dy, 2, dim=1)
    #
    #     x1, x2 = torch.chunk(x, 2, dim=1)
    #
    #     with torch.enable_grad():
    #         x1 = Variable(x1.contiguous(), requires_grad=True)
    #         x2 = Variable(x2.contiguous(), requires_grad=True)
    #         x1.retain_grad()
    #         x2.retain_grad()
    #
    #         if torch.cuda.is_available():
    #             x1.cuda()
    #             x2.cuda()
    #
    #         x1_ = possible_downsample(x1, in_channels, out_channels, stride,
    #                                   padding, dilation)
    #         x2_ = possible_downsample(x2, in_channels, out_channels, stride,
    #                                   padding, dilation)
    #
    #         f_x2 = RevBlockFunction.residual(
    #             x2,
    #             in_channels,
    #             out_channels,
    #             f_params,
    #             f_buffs,
    #             training=training,
    #             stride=stride,
    #             padding=padding,
    #             dilation=dilation,
    #             no_activation=no_activation
    #         )
    #
    #         y1_ = f_x2 + x1_
    #
    #         g_y1 = RevBlockFunction.residual(
    #             y1_,
    #             out_channels,
    #             out_channels,
    #             g_params,
    #             g_buffs,
    #             training=training
    #         )
    #
    #         y2_ = g_y1 + x2_
    #
    #         dd1 = torch.autograd.grad(y2_, (y1_,) + tuple(g_params), dy2,
    #                                   retain_graph=True, allow_unused=True)
    #         dy2_y1 = dd1[0]
    #         dgw = dd1[1:]
    #         dy1_plus = dy2_y1 + dy1
    #         dd2 = torch.autograd.grad(y1_, (x1, x2) + tuple(f_params), dy1_plus,
    #                                   retain_graph=True, allow_unused=True)
    #         dfw = dd2[2:]
    #
    #         dx2 = dd2[1]
    #         dx2 += torch.autograd.grad(x2_, x2, dy2, retain_graph=True, allow_unused=True)[0]
    #         dx1 = dd2[0]
    #
    #         for hook in storage_hooks:
    #             x = hook(x)
    #
    #         activations.append(x)
    #
    #         y1_.detach_()
    #         y2_.detach_()
    #         del y1_, y2_
    #         dx = torch.cat((dx1, dx2), 1)
    #
    #     return dx, dfw, dgw

    @staticmethod
    def forward(ctx, x, in_channels, out_channels, training, stride, padding,
                dilation, no_activation, activations, storage_hooks, *args):
        """Compute forward pass including boilerplate code.
        This should not be called directly, use the apply method of this class.
        Args:
            ctx (Context):                  Context object, see PyTorch docs
            x (Tensor):                     4D input tensor
            in_channels (int):              Number of channels on input
            out_channels (int):             Number of channels on output
            training (bool):                Whethere we are training right now
            stride (int):                   Stride to use for convolutions
            no_activation (bool):           Whether to compute an initial
                                            activation in the residual function
            activations (List):             Activation stack
            storage_hooks (List[Function]): Functions to apply to activations
                                            before storing them
            *args:                          Should contain all the Parameters
                                            of the module
        """

        if not no_activation:
            f_params = [Variable(x) for x in args[:8]]
            g_params = [Variable(x) for x in args[8:16]]
            f_buffs = args[16:20]
            g_buffs = args[20:]
        else:
            f_params = [Variable(x) for x in args[:6]]
            g_params = [Variable(x) for x in args[6:14]]
            f_buffs = args[14:16]
            g_buffs = args[16:]

        if torch.cuda.is_available():
            for var in f_params:
                var.cuda()
            for var in g_params:
                var.cuda()

        # if the images get smaller information is lost and we need to save the input
        _, _, H_in, W_in = x.size()
        _, _, H_out, W_out = size_after_residual(x.size(), out_channels, 3, stride, padding, dilation)
        if H_in > H_out or W_in > W_out or no_activation:
            activations.append(x)
            ctx.load_input = True
        else:
            ctx.load_input = False

        ctx.save_for_backward(*[x.data for x in f_params],
                              *[x.data for x in g_params])
        ctx.f_buffs = f_buffs
        ctx.g_buffs = g_buffs
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.training = training
        ctx.no_activation = no_activation
        ctx.storage_hooks = storage_hooks
        ctx.activations = activations
        ctx.in_channels = in_channels
        ctx.out_channels = out_channels

        y = RevBlockFunction._forward(
            x,
            in_channels,
            out_channels,
            training,
            stride,
            padding,
            dilation,
            f_params, f_buffs,
            g_params, g_buffs,
            no_activation=no_activation
        )

        return y.data

    @staticmethod
    def backward(ctx, grad_out):
        saved_tensors = list(ctx.saved_tensors)
        if not ctx.no_activation:
            f_params = [Variable(p, requires_grad=True) for p in saved_tensors[:8]]
            g_params = [Variable(p, requires_grad=True) for p in saved_tensors[8:16]]
        else:
            f_params = [Variable(p, requires_grad=True) for p in saved_tensors[:6]]
            g_params = [Variable(p, requires_grad=True) for p in saved_tensors[6:14]]

        in_channels = ctx.in_channels
        out_channels = ctx.out_channels

        # Load or reconstruct input
        if ctx.load_input:
            ctx.activations.pop()
            x = ctx.activations.pop()
        else:
            output = ctx.activations.pop()
            x = RevBlockFunction._backward(
                output,
                in_channels,
                out_channels,
                f_params, ctx.f_buffs,
                g_params, ctx.g_buffs,
                ctx.training,
                ctx.padding,
                ctx.dilation,
                ctx.no_activation
            )

        dx, dfw, dgw = RevBlockFunction._grad(
            x,
            grad_out,
            in_channels,
            out_channels,
            ctx.training,
            ctx.stride,
            ctx.padding,
            ctx.dilation,
            ctx.activations,
            f_params, ctx.f_buffs,
            g_params, ctx.g_buffs,
            no_activation=ctx.no_activation,
            storage_hooks=ctx.storage_hooks
        )

        num_buffs = 2 if ctx.no_activation else 4

        return ((dx, None, None, None, None, None, None, None, None, None) + tuple(dfw) +
                tuple(dgw) + tuple([None]*num_buffs) + tuple([None]*4))


class RevBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activations, stride=1,
                 padding=1, dilation=1, no_activation=False, storage_hooks=[]):
        super(RevBlock, self).__init__()

        self.in_channels = in_channels // 2
        self.out_channels = out_channels // 2
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.no_activation = no_activation
        self.activations = activations
        self.storage_hooks = storage_hooks

        if not no_activation:
            self.register_parameter(
                'f_bw1',
                nn.Parameter(torch.Tensor(self.in_channels))
            )
            self.register_parameter(
                'f_bb1',
                nn.Parameter(torch.Tensor(self.in_channels))
            )

        self.register_parameter(
            'f_w1',
            nn.Parameter(torch.Tensor(
                self.out_channels,
                self.in_channels,
                3, 3
            ))
        )
        self.register_parameter(
            'f_b1',
            nn.Parameter(torch.Tensor(self.out_channels))
        )
        self.register_parameter(
            'f_bw2',
            nn.Parameter(torch.Tensor(self.out_channels))
        )
        self.register_parameter(
            'f_bb2',
            nn.Parameter(torch.Tensor(self.out_channels))
        )
        self.register_parameter(
            'f_w2',
            nn.Parameter(torch.Tensor(
                self.out_channels,
                self.out_channels,
                3, 3
            ))
        )
        self.register_parameter(
            'f_b2',
            nn.Parameter(torch.Tensor(self.out_channels))
        )

        self.register_parameter(
            'g_bw1',
            nn.Parameter(torch.Tensor(self.out_channels))
        )
        self.register_parameter(
            'g_bb1',
            nn.Parameter(torch.Tensor(self.out_channels))
        )
        self.register_parameter(
            'g_w1',
            nn.Parameter(torch.Tensor(
                self.out_channels,
                self.out_channels,
                3, 3
            ))
        )
        self.register_parameter(
            'g_b1',
            nn.Parameter(torch.Tensor(self.out_channels))
        )
        self.register_parameter(
            'g_bw2',
            nn.Parameter(torch.Tensor(self.out_channels))
        )
        self.register_parameter(
            'g_bb2',
            nn.Parameter(torch.Tensor(self.out_channels))
        )
        self.register_parameter(
            'g_w2',
            nn.Parameter(torch.Tensor(
                self.out_channels,
                self.out_channels,
                3, 3
            ))
        )
        self.register_parameter(
            'g_b2',
            nn.Parameter(torch.Tensor(self.out_channels))
        )

        if not no_activation:
            self.register_buffer('f_rm1', torch.zeros(self.in_channels))
            self.register_buffer('f_rv1', torch.ones(self.in_channels))
        self.register_buffer('f_rm2', torch.zeros(self.out_channels))
        self.register_buffer('f_rv2', torch.ones(self.out_channels))

        self.register_buffer('g_rm1', torch.zeros(self.out_channels))
        self.register_buffer('g_rv1', torch.ones(self.out_channels))
        self.register_buffer('g_rm2', torch.zeros(self.out_channels))
        self.register_buffer('g_rv2', torch.ones(self.out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        f_stdv = 1 / math.sqrt(self.in_channels * 3 * 3)
        g_stdv = 1 / math.sqrt(self.out_channels * 3 * 3)

        if not self.no_activation:
            self._parameters['f_bw1'].data.uniform_()
            self._parameters['f_bb1'].data.zero_()
        self._parameters['f_w1'].data.uniform_(-f_stdv, f_stdv)
        self._parameters['f_b1'].data.uniform_(-f_stdv, f_stdv)
        self._parameters['f_w2'].data.uniform_(-g_stdv, g_stdv)
        self._parameters['f_b2'].data.uniform_(-g_stdv, g_stdv)
        self._parameters['f_bw2'].data.uniform_()
        self._parameters['f_bb2'].data.zero_()

        self._parameters['g_w1'].data.uniform_(-g_stdv, g_stdv)
        self._parameters['g_b1'].data.uniform_(-g_stdv, g_stdv)
        self._parameters['g_w2'].data.uniform_(-g_stdv, g_stdv)
        self._parameters['g_b2'].data.uniform_(-g_stdv, g_stdv)
        self._parameters['g_bw1'].data.uniform_()
        self._parameters['g_bb1'].data.zero_()
        self._parameters['g_bw2'].data.uniform_()
        self._parameters['g_bb2'].data.zero_()

        if not self.no_activation:
            self._buffers['f_rm1'].zero_()
            self._buffers['f_rv1'].fill_(1)
        self.f_rm2.zero_()
        self.f_rv2.fill_(1)

        self.g_rm1.zero_()
        self.g_rv1.fill_(1)
        self.g_rm2.zero_()
        self.g_rv2.fill_(1)

    def forward(self, x):
        return RevBlockFunction.apply(
            x,
            self.in_channels,
            self.out_channels,
            self.training,
            self.stride,
            self.padding,
            self.dilation,
            self.no_activation,
            self.activations,
            self.storage_hooks,
            *self._parameters.values(),
            *self._buffers.values(),
        )


class MLP(nn.Module):

    def __init__(self, num_users, num_items, model_args):
        super(MLP, self).__init__()
        self.dim = model_args.dim
        self.state_num = model_args.state_num
        self.input_size = model_args.dim * (model_args.state_num + 1)

        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.fc1 = nn.Linear(self.dim * 7, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 1)

        # user and item embeddings
        self.user_embeddings = nn.Embedding(num_users, self.dim)
        self.item_embeddings = nn.Embedding(num_items, self.dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, user, state, action):
        item_embs = self.item_embeddings(state).unsqueeze(1)
        item_embs = item_embs.view(-1, 5 * self.dim)
        user_emb = self.user_embeddings(user).squeeze(1)
        action_emb = self.item_embeddings(action).squeeze(1)

        x = torch.cat((user_emb, item_embs, action_emb), 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = (self.fc6(x))

        return x


class Feature(nn.Module):

    def __init__(self, num_users, num_items, model_args, dim):
        super(Feature, self).__init__()
        self.dim = dim
        self.state_num = model_args.state_num
        self.input_size = self.dim * (model_args.state_num + 1)

        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.ifrevnet = model_args.ifrevnet

        if self.ifrevnet == 1:
            print('Use RevNet')
            self.layer1 = self._rev_make_layer_yinan(RevBlock_yinan, 64, 64, 1)
            self.layer2 = self._rev_make_layer_yinan(RevBlock_yinan, 128, 128, 1)
            self.conv = conv3x3(64, 128, 2)
            self.fc = nn.Linear(128 * (self.state_num - 2) * int(self.dim / 2), self.dim)
        elif self.ifrevnet == 2:
            print('Use ConvLayers ONLY')
            self.layer1 = self._no_make_layer(GeneralBlock, 64, 1)
            self.layer2 = self._no_make_layer(GeneralBlock, 128, 1, stride=2)
            self.fc = nn.Linear(128 * (self.state_num - 2) * int(self.dim / 2), self.dim)
        elif self.ifrevnet == 3:
            print('NOT ConvLayers')
            self.layer1 = self._noconv_make_layer(GeneralBlock2, self.inplanes * (self.state_num+1) * self.dim, 256)
            self.layer2 = self._noconv_make_layer(GeneralBlock2, 256, 256)
            self.fc = nn.Linear(256, self.dim)
        else:
            print('Use ResNet')
            self.layer1 = self._res_make_layer(BasicBlock, 64, 1)
            self.layer2 = self._res_make_layer(BasicBlock, 128, 1, stride=2)
            self.fc = nn.Linear(128 * (self.state_num - 2) * int(self.dim / 2), self.dim)

        # user and item embeddings
        self.user_embeddings = nn.Embedding(num_users, self.dim)
        self.item_embeddings = nn.Embedding(num_items, self.dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _no_make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _noconv_make_layer(self, block, inplanes, planes):
        layers = []
        layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)


    def _res_make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _rev_make_layer_yinan(self, block, inplanes, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(int(inplanes/2), int(planes/2), stride, downsample, norm_layer))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _rev_make_layer(self, filters=[64,64,64], strides=[1,1], units=[1,1]):
        """
                Args:
                    units (list-like): Number of residual units in each group
                    filters (list-like): Number of filters in each unit including the
                        inputlayer, so it is one item longer than units
                    strides (list-like): Strides to use for the first units in each
                        group, same length as units
                    bottleneck (boolean): Wether to use the bottleneck residual or the
                        basic residual
                """
        self.Reversible = RevBlock
        self.layers = nn.ModuleList()
        self.activations = []
        for i, group_i in enumerate(units):
            self.layers.append(self.Reversible(
                filters[i], filters[i + 1],
                stride=strides[i],
                no_activation=True,
                activations=self.activations
            ))

            for unit in range(1, group_i):
                self.layers.append(self.Reversible(
                    filters[i + 1],
                    filters[i + 1],
                    activations=self.activations
                ))

    def forward(self, user, state):
        item_embs = self.item_embeddings(state).unsqueeze(1)
        item_embs = item_embs.view(-1, self.state_num, self.dim)
        user_emb = self.user_embeddings(user).squeeze(1)
        user_emb = user_emb.view(-1, 1, self.dim)

        x = torch.cat((user_emb, item_embs), 1)
        x = torch.unsqueeze(x, 1)
        # See note [TorchScript super()]
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu(x)
        # print('x1 = ', x.shape)
        x = self.layer1(x)
        if self.ifrevnet == 1:
            x = self.conv(x)
            x = self.relu(x)
        x = self.layer2(x)
        #
        # print('x2 = ', x.shape)
        # input('debug')
        if len(x.shape) > 2:
            x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class ActorModel(nn.Module):

    def __init__(self, dim):
        super(ActorModel, self).__init__()
        self.dim = dim
        self.fc1 = nn.Linear(self.dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, self.dim)

    def forward(self, state):
        if activation == 'selu':
            out = nn.SELU()(self.fc1(state))
            out = nn.SELU()(self.fc2(out))
            out = nn.SELU()(self.fc3(out))
        else:
            out = F.relu(self.fc1(state))
            out = F.relu(self.fc2(out))
            out = F.relu(self.fc3(out))

        out = torch.tanh(self.fc4(out))

        return out

    def initialization(self):
        for name, params in self.named_parameters():
            if name.find('weight') != -1:
                if name.find('conv') != -1:
                    nn.init.xavier_uniform_(params)
                else:
                    nn.init.normal_(params, 0, 1.0 / self.dim)
                # nn.init.normal_(params, 0, 1.0 / self.dim)
            elif name.find('bias') != -1:
                nn.init.zeros_(params)


class CriticModel(nn.Module):

    def __init__(self, dim):
        super(CriticModel, self).__init__()
        self.dim = dim
        self.fc1 = nn.Linear(self.dim * 2, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, state, action):
        out = torch.cat((state, action), 1)
        if activation == 'selu':
            out = nn.SELU()(self.fc1(out))
            out = nn.SELU()(self.fc2(out))
            out = nn.SELU()(self.fc3(out))
        else:
            out = F.relu(self.fc1(out))
            out = F.relu(self.fc2(out))
            out = F.relu(self.fc3(out))
        out = self.fc4(out)

        return out

    def initialization(self):
        for name, params in self.named_parameters():
            if name.find('weight') != -1:
                if name.find('conv') != -1:
                    nn.init.xavier_uniform_(params)
                else:
                    nn.init.normal_(params, 0, 1.0 / self.dim)
                # nn.init.normal_(params, 0, 1.0 / self.dim)
            elif name.find('bias') != -1:
                nn.init.zeros_(params)

