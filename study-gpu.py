import torch
from G import G
from D import D
import dataset
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

generator = G().to(device)
discriminator = D().to(device)

loss_function = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))


noise_test = torch.randn([1, 100, 1, 1]).to(device)

print(torch.cuda.is_available())

for e in range(10000):
    print("==============epic {}================".format(e))
    for i, data in enumerate(dataset.dataloader, 0):

        print("==============iter {}================".format(i))

        image, _ = data

        # D
        discriminator.zero_grad()
        noises_D_train = torch.randn([len(image), 100, 1, 1], requires_grad=True).to(device)

        score_real = discriminator(image.to(device))
        objective_image = torch.Tensor([0]).repeat(len(image), 1).float().to(device)
        loss_D_real = loss_function(score_real, objective_image)
        loss_D_real.backward()

        fake = generator(noises_D_train).to(device)

        score_fake = discriminator(fake.detach()).to(device)        
        objective_fake = torch.Tensor([1]).repeat(len(image), 1).float().to(device)
        loss_D_fake = loss_function(score_fake, objective_fake)
        loss_D_fake.backward()

        optimizer_D.step()

        print("D Loss: {}".format(loss_D_fake + loss_D_real))

        # G
        generator.zero_grad()
        score_fake = discriminator(fake)
        objective_noise = torch.Tensor([0]).repeat(len(image), 1).float().to(device)

        loss_G = loss_function(score_fake, objective_noise)

        loss_G.backward()
        optimizer_G.step()

        print("G Loss: {}".format(loss_G))
    dataset.imsave(generator(noise_test).cpu().detach()[0], "{0}".format(e))

    print("===================================")

