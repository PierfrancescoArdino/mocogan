import os
import PIL
import functools
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from models import PatchImageDiscriminator, CategoricalVideoDiscriminator, VideoGenerator
from data import VideoFolderDataset, ImageDataset, VideoDataset
from trainers import Trainer


def build_discriminator(type, **kwargs):
      if type == "PatchImageDiscriminator":
        return PatchImageDiscriminator(**kwargs)
      if type == "CategoricalVideoDiscriminator":
        return CategoricalVideoDiscriminator(**kwargs)


def video_transform(video, image_transform):
    vid = []
    for im in video:
        vid.append(image_transform(im))

    vid = torch.stack(vid).permute(1, 0, 2, 3)

    return vid


img_size = 64
video_length = 16
image_batch = 10
video_batch = 3

dim_z_content = 10
dim_z_motion = 10
dim_z_category = 2
print_every = 30
batches = 100000
log_folder = "./"
use_infogan = 0
use_categories = 1


image_discriminator = "PatchImageDiscriminator"
video_discriminator = "CategoricalVideoDiscriminator"
use_noise = 0
noise_sigma = 0
n_channels = 3
dataset = "data/shapes"

image_transforms = transforms.Compose([
    PIL.Image.fromarray,
    transforms.Scale(img_size),
    transforms.ToTensor(),
    lambda x: x[:n_channels, ::],
    transforms.Normalize((0.5, 0.5, .5), (0.5, 0.5, 0.5)),
])

video_transforms = functools.partial(video_transform, image_transform=image_transforms)


dataset = VideoFolderDataset(dataset, cache=None)
image_dataset = ImageDataset(dataset, image_transforms)
image_loader = DataLoader(image_dataset, batch_size=image_batch, drop_last=True, num_workers=2, shuffle=True)


video_dataset = VideoDataset(dataset, 16, 2, video_transforms)
video_loader = DataLoader(video_dataset, batch_size=video_batch, drop_last=True, num_workers=2, shuffle=True)

generator = VideoGenerator(n_channels, dim_z_content, dim_z_category, dim_z_motion, video_length)

image_discriminator = build_discriminator(image_discriminator, n_channels=n_channels,
                                          use_noise=use_noise, noise_sigma=noise_sigma)

video_discriminator = build_discriminator(video_discriminator, dim_categorical=dim_z_category,
                                          n_channels=n_channels, use_noise=use_noise,
                                          noise_sigma=noise_sigma)

if torch.cuda.is_available():
    generator.cuda()
    image_discriminator.cuda()
    video_discriminator.cuda()

trainer = Trainer(image_loader, video_loader,
                  image_loader, video_loader,
                  print_every,
                  batches,
                  log_folder,
                  use_cuda=torch.cuda.is_available(),
                  use_infogan=use_infogan,
                  use_categories=use_categories)

trainer.train(generator, image_discriminator, video_discriminator)
