import torchvision.transforms as transforms

from models.densenet import DenseNet3

def build_model(model_name, num_classes):
    if model_name == 'densenet100':
        model = DenseNet3(100, int(num_classes))
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            # transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255), (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0)),
        ])
        return model, transform
    exit('{} model is not supported'.format(model_name))
