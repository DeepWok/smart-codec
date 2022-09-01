from .resnet import ResNet50, ResNet50Aug, ResNet50ImageNet, ResNet50ImageNetAug
from .wideresnet import WideResNet, WideResNetAug
from .shakeshake import ShakeResNet, ShakeResNetAug
from .simplenet import SimpleNet


def get_network_by_name(name, num_class=10, model_args=None):
    if name == 'resnet50':
        return ResNet50(num_classes=num_class)
    elif name == 'resnet50imagenet':
        return ResNet50ImageNet(num_classes=num_class)
    elif name == 'resnet50_aug_imagenet':
        return ResNet50ImageNetAug(num_classes=num_class, **model_args)
    elif name == 'resnet50_aug':
        if model_args is None:
            raise ValueError('model_args is required for resnet50_aug')
        return ResNet50Aug(num_class=num_class, **model_args)
    elif name == 'wresnet40_2':
        model = WideResNet(
            40, 2, 
            dropout_rate=0.0, 
            num_classes=num_class, 
            adaptive_dropouter_creator=None,
            adaptive_conv_dropouter_creator=None, 
            groupnorm=False, 
            examplewise_bn=False, 
            virtual_bn=False)
        return model
    elif name == 'wresnet':
        model = WideResNet(
            28, 10, 
            dropout_rate=0.0, 
            num_classes=num_class, 
            adaptive_dropouter_creator=None,
            adaptive_conv_dropouter_creator=None, 
            groupnorm=False, 
            examplewise_bn=False, 
            virtual_bn=False)
        return model
    elif name == 'wresnet_aug':
        model = WideResNetAug(
            28, 10, 
            dropout_rate=0.0, 
            num_classes=num_class, 
            adaptive_dropouter_creator=None,
            adaptive_conv_dropouter_creator=None, 
            groupnorm=False, 
            examplewise_bn=False, 
            virtual_bn=False,
            **model_args)
        return model
    elif name == 'shakeshake':
        model = ShakeResNet(
            26, 96, 
            num_classes=num_class)
        return model
    elif name == 'shakeshake_aug':
        model = ShakeResNetAug(
            26, 96, 
            num_classes=num_class, **model_args)
        return model
    # elif name == 'wresnet28_2':
    #     model = WideResNet(28, 2, dropout_rate=conf.get('dropout', 0.0), num_classes=num_class,
    #                        adaptive_dropouter_creator=ad_creators[0], adaptive_conv_dropouter_creator=ad_creators[1],
    #                        groupnorm=conf.get('groupnorm', False), examplewise_bn=conf.get('examplewise_bn', False),
    #                        virtual_bn=conf.get('virtual_bn', False))
    elif name == 'simplenet':
        model = SimpleNet(num_class)
        return model
    else:
        raise ValueError('Unknown network name: {}'.format(name))
