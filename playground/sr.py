import cv2

img=cv2.imread('1.jpg')
sr = cv2.dnn_superres.DnnSuperResImpl_create()
path = "EDSR_x4.pb"
sr.readModel(path)
sr.setModel("edsr", 4) # set the model by passing the value and the upsampling ratio
result = sr.upsample(img) # upscale the input image
cv2.imwrite('img.jpg',result)
