import torch
import torchvision
from torchvision import transforms as T
import torchvision.transforms.functional as F
import PIL.Image as Image

transforms = {
    "cmnist": {
        "train":
            [
                T.ToTensor(),
                T.Normalize((0.1307,), (0.3081,)),
            ],
        "test":
            [
                T.ToTensor(),
                T.Normalize((0.1307,), (0.3081,)),
            ]
    },

    "cifar10c": {
        "train":
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ],

        "test":
            [
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
    },

    # Added for the plain-CIFAR-10 pipeline (cifar10.ipynb). Same photos as
    # cifar10c (just without the artificial spurious-correlation coloring
    # baked in), so the same normalization stats apply.
    "cifar10": {
        "train":
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ],

        "test":
            [
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
    },
}

class ColorDegrade(torch.nn.Module):
    """Convert image to grayscale.
    If the image is torch Tensor, it is expected
    to have [..., 3, H, W] shape, where ... means an arbitrary number of leading dimensions

    Args:
        num_output_channels (int): (1 or 3) number of channels desired for output image

    Returns:
        PIL Image: Grayscale version of the input.

        - If ``num_output_channels == 1`` : returned image is single channel
        - If ``num_output_channels == 3`` : returned image is 3 channel with r == g == b

    """

    def __init__(self, num_output_channels=1, ratio=0):
        super().__init__()
        torchvision.utils._log_api_usage_once(self)
        self.num_output_channels = num_output_channels
        self.ratio = ratio

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be converted to grayscale.

        Returns:
            PIL Image or Tensor: Grayscaled image.
        """

        # if ratio is 0, return grayscale
        # if ratio is 1, return full-color
        # if ratio is between 0 and 1, return weighted sum using ratio

        if self.ratio == 0:
            return torchvision.transforms.functional.rgb_to_grayscale(img, num_output_channels=self.num_output_channels)
        elif self.ratio == 1:
            return img
        else:
            grayscale = torchvision.transforms.functional.rgb_to_grayscale(img, num_output_channels=self.num_output_channels)
            return Image.blend(grayscale, img, self.ratio)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_output_channels={self.num_output_channels})"

def get_transforms(dataset,
                   blur=0,
                   color=1,
                   test=False,
                   exclude_to_tensor=False,
                   imagenet_resize=False):
    """
    Get the degredation transformation for the dataset.

    Args:
        dataset (str): the dataset name
        blur (int): the size of the Gaussian blur kernel
        color (int): the ratio of color degredation
        test (bool): if True, return the test transformation
        exclude_to_tensor (bool): if True, exclude the ToTensor transformation
        imagenet_resize (bool): if True, resize to 224x224 (after blur/color/crop,
            which stay at native 32x32 so degradation strength is unchanged) and
            renormalize with ImageNet stats, for use with ImageNet-pretrained backbones.
    """

    transform = transforms[dataset]["test" if test else "train"]

    if blur != 0:
        transform = [T.GaussianBlur(kernel_size=blur, sigma=(1, 2))] + transform

    if color != 1:
        transform = [ColorDegrade(num_output_channels=3, ratio=color)] + transform

    if imagenet_resize:
        transform = [t for t in transform if not isinstance(t, T.Normalize)]
        to_tensor_idx = next(i for i, t in enumerate(transform) if isinstance(t, T.ToTensor))
        transform = transform[:to_tensor_idx] + [T.Resize(224)] + transform[to_tensor_idx:]
        transform = transform + [T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]

    if exclude_to_tensor:
        transform = [t for t in transform if not isinstance(t, T.ToTensor)]

    return T.Compose(transform)
