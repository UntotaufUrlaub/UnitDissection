import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import threading
import random
import torchvision

'''
Reminder: 
You can adjust some settings: imagesubset_size, threadnumber, stepsize (inside the main function).
The result dir will not be cleaned before a run. 
If there happens to be a collision of file names (which depend on the settings) the old ones will be overwritten. 
So don't mess up your results.

It's critical that the right features are assigned to the right pixel. I think it should be correct.
It seems parallelism could be improved. this would be very useful to get away from to long runtime.
This code analyses alexnet's feature layer. 
The model is loaded the same way as in the Netdissect Lite project (given settings adjustments). 
So the results should be comparable/linked.

TODO:
compare results
calculate for all units / adjust code to allow which units to analyse
'''

def modifyModel(model):  #this should remove all the rear layers we dont need. So the remaining output layer is the one we want to analyse
    #modules = list(model.children())[:-1]
    #my = nn.Sequential(*modules)
    my = model._modules.get('features')
    return my

def generateMappingForOneImage(input_image):
    dim = input_image.size
    #print(dim)

    # resize the picture to fit to the requirements of alex net.
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    #print(input_tensor.shape)

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    #print(probabilities.shape)

    data = probabilities.cpu().detach().numpy()
    #print(data[0].shape)
    activationImage = Image.fromarray(data[0])
    activationImageResized = activationImage.resize(tuple(reversed(dim)), resample=Image.BILINEAR)
    # activationImage.convert('RGB').save("activation.jpg")
    # activationImageResized.convert('RGB').save("activation.jpg")
    activationResized = np.array(activationImageResized)
    return activationResized

def representsInt(s):
    s = str(s)
    if s[0] in ('-', '+'):
        return s[1:].isdigit()
    return s.isdigit()

def saveImageV(array, name):
    a = np.interp(array, (array.min(), array.max()), (0, 254))
    Image.fromarray(a).convert('RGB').save(name + ".jpg")

def saveImageA(array, name):
    a = np.interp(array, (0, 0.02), (0, 254))
    Image.fromarray(a).convert('RGB').save(name + ".jpg")

def saveValueImage(x, values, count2, count):
    print(x)
    saveImageV(values, 'values/v' + str(count2) + '_' + str(count))

def debugActivations(count2, activations, values):
    #if 0.15 < activations.max():
    saveImageA(activations, 'activations' + str(count2))
    index = np.unravel_index(np.argmax(activations, axis=None), activations.shape)
    print(index)
    if not (values is None): print(values[index])

def getPixelLabels(basePath, header, infos, imageBase, activations, count2):
    count = 0
    res = np.empty((224, 224), dtype=object)
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            res[i][j] = []
    values = None
    for pos in range(6, 12):
        e = infos[pos]
        if e != '':
            # print(header[pos])
            for x in e.split(';'):
                if representsInt(x):
                    values = np.zeros_like(imageBase[:, :, 0]) + int(x)
                else:
                    image = Image.open(basePath + 'images/' + x)
                    imageO = np.array(image)
                    image = image.resize(tuple(reversed(imageBase.shape[0:2])), Image.NEAREST)
                    image = np.array(image)
                    assert (image.min() == imageO.min() and image.max() == imageO.max())
                    values = image[:, :, 0] + image[:, :, 1] * 256
                for i in range(values.shape[0]):
                    for j in range(values.shape[1]):
                        # if values[i][j] != 0:
                        res[i][j].append(values[i][j])
                # debugValueImage(x, values, count2, count)
                # count += 1
    # debugActivations(count2, activations, values)
    # print(random.random())
    return res

def do(Lines, start, number):
    print(str(start) + ' started for ' + str(number) + ' lines')
    file_res = open('/home/res/res' + str(start) + '.csv', 'w+')
    for line in Lines[start:(start+number)]:
        infos = line.strip().split(",")
        image = Image.open(basePath + 'images/' + infos[0])
        imageBase = np.array(image)

        activations = generateMappingForOneImage(image) # todo: make unit selectable
        truth = getPixelLabels(basePath, None, infos, imageBase, None, None)
        assert(truth.shape == activations.shape)
        for i in range(224):
            for j in range(224):
                activation = activations[i][j]
                for label in truth[i, j]:
                    file_res.write(str(activation) + "," + str(label) + "\n")
    file_res.close()
    print(str(start) + ' finished')

if __name__ == "__main__":
    #model = torch.hub.load('pytorch/vision:v0.9.0', 'alexnet', pretrained=True)
    model = torchvision.models.__dict__["alexnet"](pretrained=True) # This should load the same alexnet as used in netdissect lite. Both analyse the last layer of Features. So the results should be comparable
    model.eval()
    print(model)
    model = modifyModel(model)
    print(model)

    basePath = '/home/NetDissect-Lite/dataset/broden1_224/'
    file1 = open(basePath + 'index.csv', 'r')
    Lines = file1.readlines()

    header = Lines[0].strip().split(",")
    print(header)
    imagesubset_size = 2000     # should not be bigger than the length of Lines.
    if imagesubset_size != 0:
        Lines = random.sample(Lines, imagesubset_size)
    stepsize = 199      # for each step and each thread an own result file is created. So the bigger the step size the bigger one file. Smaller files might be easier to handle in the r script on machines with lower RAM
    threadnumber = 1    # should be bigger than 0. One means running in the main thread
    for s in range(1, len(Lines), stepsize):
        threads = []
        for x in range(0, threadnumber):
            number = int(stepsize / threadnumber)
            start = number * x + s
            if start + number >= len(Lines):    # is needed to handle rounding effects in the calculation of number.
                print('continue')
                continue    # todo: don't skip. better adjust number accordingly.
            if threadnumber == 1:
                do(Lines, start, number)
            else:
                t = threading.Thread(target=do, args=(Lines, start, number))
                t.start()
                threads.append(t)
        for t in threads:
            t.join()