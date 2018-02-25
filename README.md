# ATA-GAN


Demo code for our work on Attention-Aware Generative Adversarial networks (ATA-GANs) <a href="">Available soon </a> <br />

Abstract: A new family of generative models named as Generative Adversarial Networks (GANs) has been proposed. Using two neural networks (one that generates samples and one that evaluates them as real or fake), these networks are able
to learn a mapping between an input domain and the actual data. However, the ability of the generator to locate the regions
of interest has not been studied yet. In this context, here, we are using GANs in order to generate images from HEp-2 cells
captured with Indirect Imunofluoresence (IIF) and study the ability of the discriminator to perform a weekly localization of the cell. Our contribution is four-fold. First, we demonstrate that whilst GANs can learn the mapping between the input
domain and the target distribution efficiently, the discriminator network is not able to detect the regions of interest. Secondly, we present a novel attention transfer mechanism which allows us to enforce the discriminator to put emphasis on the regions of interest via transfer learning. Thirdly, we show that this can generate more realistic images, as the discriminator learns to put emphasis on the area of interest. Fourthly, the proposed method allows one to generate both images as well as attention maps which can be useful for data annotation e.g in object detection.

<br />
Demo code will be available [soon](.)


## Some Results:



### Generated versus Real images.

Below, you can see some generated (Left) as well as some real (genuine) images (Right): <br />

<img src="gitimages/GeneratedImages_git.png" width="425"/> <img src="gitimages/RealImages_git.png" width="425"/> 

<br />
 
 
 ### Discriminator Attention Maps.
 
 Below, you can see some generated (Left) as well as some real (genuine) images (Right) together with the Soft-Class Activation Maps: <br />
 
<img src="gitimages/FAKE_D_cams.png" width="425"/> <img src="gitimages/REAL_D_cams.png" width="425"/> 

<br />
