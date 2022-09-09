from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
import os

DEST_PATH = "./images/"
if not os.path.exists(DEST_PATH):
  # Create a new directory because it does not exist 
    os.makedirs(DEST_PATH)
    print("The new image dest dir is created!")
  
def save_noise_images(output_noise_images):
    for idx,img in enumerate(output_noise_images):
        permuted_img = (img.permute(2,1,0) * 255).cpu().numpy().astype(np.uint8)
        #plt.imshow(permuted_img)
        pil_img = transforms.ToPILImage()(img)
        pil_img.save(DEST_PATH+'image_'+str(idx)+'.jpg')
        
def plot_losses(c_loss_lst,g_loss_lst):
    plt.plot(c_loss_lst, label='Discriminator Losses')
    plt.plot(g_loss_lst, label='Generator Losses')
    plt.legend()
    plt.savefig(DEST_PATH+'total_loss.png')