from utils.gradcam import BrainTumorGradCAM

cam = BrainTumorGradCAM()

cam.save(

    "dataset/test/glioma/image1.jpg",

    "outputs/gradcam.png"

)
