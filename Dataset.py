import torchvision
from PIL import Image
from torch.utils.data import Dataset


class DomainNet(Dataset):
    def __init__(self, domain_type="real", train=True, root_path="", height=224, width=224, transform=None,
                 ID_index=172, ID_domain="real"):
        """

        :param domain_type: The name of the desired Domain
        :param train: return testset if false
        :param root_path:  the root path of the dataset
        :param height: default 224
        :param width: default 224
        :param transform: whether use transform (if None than use the default set)
        :param ID_index: is the class index which you decide to be the border between ID with OOD
        :param ID_domain: to tell which domain is the ID domain
        """
        self.root_path = root_path
        self.domain_type = domain_type
        self.id_class_index = ID_index
        self.ID_domain = ID_domain
        # Get the txt file path
        name = "test.txt" if train else "train.txt"
        self.base_file_name = domain_type + "_" + name
        self.base_file_path = root_path + '/' + domain_type + '/' + self.base_file_name
        print(self.base_file_path)

        imgs = []
        with open(self.base_file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                words = line.split()
                imgs.append((words[0], int(words[1])))
            self.imgs = imgs
        self.height = height
        self.width = width
        self.transform = transform

        # Based ON Generalized ODIN
        if self.transform is None:
            self.transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.CenterCrop(self.height),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]
            )

    def __getitem__(self, item):
        path, class_label = self.imgs[item]
        path = self.root_path + '\\' + self.domain_type + '/' + path

        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        # OOD SET on semantic shift
        ood_label = 0
        if class_label <= self.id_class_index:
            ood_label = 0
        else:
            ood_label = 1

        # OOD SET on domain shift
        if self.ID_domain != self.domain_type:
            ood_label = 1

        return img, class_label, ood_label

    def __len__(self):
        return len(self.imgs)


class Cifar10_SVHN(Dataset):
    def __init__(self, data_root, train=True, download=True):
        self.ID_dataset = torchvision.datasets.CIFAR10(
            root=data_root, train=train, download=download,
        )
        self.OOD_dataset = torchvision.datasets.SVHN(
            root=data_root, split="train" if train else "test", download=download,
        )

        self.ID_len = len(self.ID_dataset)
        self.OOD_len = len(self.OOD_dataset)
        self.len = self.ID_len + self.OOD_len

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])

    def __getitem__(self, item):
        ood_label = 0

        if item <= self.ID_len:
            ood_label = 0
            img, label = self.ID_dataset[item]
        else:
            ood_label = 1
            img, label = self.OOD_dataset[item]

        return self.transform(img), label, ood_label

    def __len__(self):
        return self.len - 1


class Cifar100(Dataset):
    def __init__(self, data_root, train=True, download=True, ID_index=30):
        self.dataset = torchvision.datasets.CIFAR100(
            root=data_root, train=train, download=download,
        )
        self.ID_index = ID_index

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])

    def __getitem__(self, item):
        img, label = self.dataset[item]

        if label <= self.ID_index:
            ood_label = 0
        else:
            ood_label = 1

        return self.transform(img), label, ood_label

    def __len__(self):
        return len(self.dataset) - 1


if __name__ == '__main__':
    # trainset = DomainNet(domain_type="quickdraw", root_path=r"C:\Users\13391\Downloads\dataset")
    # img, label, ood_label = trainset[-1]
    #
    # print(img.size(), label, ood_label)
    #
    # trainset1 = Cifar10_SVHN(data_root="/tmp/public_dataset/pytorch", train=False)
    # print(trainset1[0])

    trainset2 = Cifar100(data_root="/tmp/public_dataset/pytorch", train=True)
    print(trainset2[0])
