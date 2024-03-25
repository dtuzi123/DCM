"""
Train a diffusion model on images.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import time

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import  structural_similarity
from NetworkModels.MemoryUnitFramework_ import *
from NetworkModels.MemoryUnitGraphFramework_ import *

#
import argparse
import torch
from datasets.MyCIFAR10 import *
from NetworkModels.Balance_TeacherStudent_NoMPI_ import *
from NetworkModels.Teacher_Model_NoMPI_ import *

from Task_Split.Task_utilizes import *
import cv2
from cv2_imageProcess import *
from datasets.Data_Loading import *
from datasets.Fid_evaluation import *
from Task_Split.TaskFree_Split import *
from datasets.MNIST32 import *
import torchvision.transforms as transforms
import torch.utils.data as Data
from NetworkModels.TFCL_TeacherStudent_ import *
from NetworkModels.DynamicDiffusionMixture_ import *

#dad
import numpy as np

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop
#
import torch.nn.functional as F         # 函数包

import torch.distributions as td
from torch.distributions.multivariate_normal import MultivariateNormal
#
def Transfer_To_Numpy(sample):
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    mySamples = sample.unsqueeze(0).cuda().cpu()
    mySamples = np.array(mySamples)
    mySamples = mySamples[0]
    return mySamples

def Save_Image(name,image):
    cv2.imwrite("results/" + name, image)
    #cv2.waitKey(0)

def TransferNumpyToTensor(totalSetX,device):
    newSet = []
    for i in range(np.shape(totalSetX)[0]):
        arr1 = totalSetX[i]
        arr1 = torch.tensor(arr1).cuda().to(device=device, dtype=torch.float)
        newSet.append(arr1)
    return newSet

def RandomSelectionArr(memory,newXList):
    for i in range(np.shape(newXList)[0]):
        memory = RandomSelection(memory,newXList[i])
    return memory

def RandomSelection(memory,newX):
    N = np.shape(memory)[0] + 1
    j = int(random.random() * N)
    if j > 0 and j < N-2:
        memory[j] = newX
    return memory

def GiveMSE(data,reco):
    mse = nn.functional.mse_loss(data,reco)
    mse.unsqueeze(0).cuda().cpu()
    return mse
#

def Calculate_JS(TSFramework,batch,batchReco):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    miniBatch = 64

    batch = batch.reshape(np.shape(batch)[0],32*32*3)
    batchReco = batchReco.reshape(np.shape(batchReco)[0],32*32*3)
    std = np.zeros((np.shape(batch)))
    std[:,:] = 0.01
    std = torch.tensor(std).cuda().to(device=device, dtype=torch.float)

    t = 100
    diffusion = TSFramework.teacherArray[0].diffusion
    schedule_sampler = UniformSampler(diffusion)
    times, weights = schedule_sampler.sample(np.shape(batch)[0], dist_util.dev())
    for i in range(np.shape(times)[0]):
        times[i] = t

    beta = _extract_into_tensor(TSFramework.teacherArray[0].diffusion.sqrt_alphas_cumprod, times, batch.shape)

    batch = batch * beta
    batchReco = batchReco * beta

    q_z1 = td.normal.Normal(batch, std)
    q_z2 = td.normal.Normal(batchReco, std)
    score11 = td.kl_divergence(q_z1, q_z2).mean()
    score12 = td.kl_divergence(q_z2, q_z1).mean()

    score11 = score11 / miniBatch
    score12 = score12 / miniBatch
    score = (score11 + score12) / 2.0
    return score

def extract(input, t, x):
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)

def make_beta_schedule(schedule='linear', n_timesteps=1000, start=1e-5, end=1e-2):
    if schedule == 'linear':
        betas = torch.linspace(start, end, n_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, n_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    return betas
#
def q_x(x_0, t, noise=None):
    num_steps = 100
    betas = make_beta_schedule(schedule='sigmoid', n_timesteps=num_steps, start=1e-5, end=0.5e-2)
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, 0)
    alphas_prod = alphas_prod.to(x_0.device)
    #alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    if noise is None:
        noise = torch.randn_like(x_0)
    alphas_t = extract(alphas_bar_sqrt, t, x_0)
    alphas_1_m_t = extract(one_minus_alphas_bar_sqrt, t, x_0)
    return (alphas_t * x_0 + alphas_1_m_t * noise)

#
def Calculate_ExpansionScore(TSFramework,batch):
    arr = []
    #t = torch.tensor([50])
    t= 50
    for i in range(np.shape(TSFramework.teacherArray)[0]):
        currentComponent = TSFramework.teacherArray[i]

        buffer = currentComponent.memoryBuffer
        #reco1 = currentComponent.q_sample(batch,t) #q_x(buffer,t)
        #reco2 = currentComponent.q_sample(buffer,t)#q_x(batch,t)
        reco1 = batch
        reco2 = buffer

        score = Calculate_JS(TSFramework,reco1,reco2)
        score = score.cpu().detach().numpy()
        #score = score[0]
        arr.append(score)

    #arr = arr.cpu().numpy()
    arr = np.array(arr)
    maxScore = np.min(arr)
    index = np.argmin(arr)
    return maxScore,index

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def LoadModel():
    dataNmae = "mnist"
    modelName = "GraphMemory"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataStramX, dataStramY, totalTestX, totalSetY = Give_DataStream_Supervised(dataNmae)
    #dist_util.setup_dist()

    defaultTest = totalTestX
    test_data = torch.tensor(defaultTest).cuda().to(device=device, dtype=torch.float)

    threshold = 1900
    myModelName = modelName + str(threshold) + "_" + dataNmae + ".pkl"
    TSFramework = torch.load('./data/' + myModelName)

    # Evaluation
    print("Generation")
    generated = TSFramework.Give_GenerationFromTeacher(1000)
    mytest = test_data[0:np.shape(generated)[0]]
    fid1 = calculate_fid_given_paths_Byimages(mytest, generated, 50, device, 2048)
    print(fid1)

#
def main():
    #
    dataNmae = "mnist"
    modelName = "GraphMemory"
    distanceType = "WDistance"
    modelName = modelName + "_" + distanceType

    #
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #dataStream,totalTestX,defaultTest = Give_DataStream_Unsupervised(dataNmae)

    dataStramX, dataStramY, totalTestX, totalSetY = Give_DataStream_Supervised(dataNmae)
    #dist_util.setup_dist()

    defaultTest = totalTestX

    miniBatch = 64
    totalTrainingTime = int(dataStramX.shape[0] / miniBatch)

    inputSize = 32
    epoch = 1

    Tepoch = 100
    Sepoch = 100

    start = time.time()
    inputSize = 32
    TSFramework = MemoryUnitGraphFramework("myName",device,inputSize)
    TSFramework.distance_type = distanceType
    TSFramework.MaxMemoryCluster = 20

    test_data = torch.tensor(defaultTest).cuda().to(device=device, dtype=torch.float)
    batch = test_data[0:64]
    batch2 = dataStramX[0:64]

    #build the first one
    newComponent = TSFramework.Create_NewComponent()
    TSFramework.currentComponent = newComponent
    batch = dataStramX[0:miniBatch]
    newComponent.memoryBuffer = batch

    TSFramework.currentMemory = batch
    TSFramework.maxMemorySize = 2000

    memoryBuffer = []
    maxMemorySize = 2000
    maxMemorySizeDefault = 2000

    dataloader = []

    #
    threshold = 2100

    TSFramework.threshold = threshold
    epoch = 6
    runCount = 0

    currentValue = 0
    arr = []
    currentClass = 1
    componentArr = []
    classArr = []
    runStep = 0

    dataStramX = dataStramX.unsqueeze(0).cuda().cpu()
    dataStramX = np.array(dataStramX)
    dataStramX = dataStramX[0]

    for step in range(totalTrainingTime):
        batch = dataStramX[step*miniBatch:(step + 1)*miniBatch]

        if np.shape(TSFramework.MemoryClusterArr)[0] == 0:
            TSFramework.MemoryBegin(batch)

        y = dataStramY[step*miniBatch:(step + 1)*miniBatch]
        batch_cpu = y.unsqueeze(0).cuda().cpu()
        batch_cpu = batch_cpu[0]
        batch_cpu = np.array(batch_cpu)
        batch_cpu = np.argmax(batch_cpu,1)

        runStep = runStep + 1

        #TSFramework.currentMemory = batch

        maxin = np.max(batch_cpu)
        if maxin > currentClass:
            currentClass = maxin

        classArr.append(currentClass)
        componentArr.append(np.shape(TSFramework.MemoryClusterArr)[0])

        print("epoch {0}/{1}, step {2}/{3}, train ELBO: {4:.2f}, val ELBO: {5:.2f}, time: {6:.2f}"
              .format(step, totalTrainingTime, np.shape(TSFramework.MemoryClusterArr)[0], 0, 1, 0, 1))


        memoryBuffer = TSFramework.GiveMemorizedSamples()
        memoryBuffer = torch.tensor(memoryBuffer).cuda().to(device=device, dtype=torch.float)

        batch2 = torch.tensor(batch).cuda().to(device=device, dtype=torch.float)

        memoryBuffer_ = torch.cat([memoryBuffer,batch2],0)

        TSFramework.currentComponent.Train(epoch,memoryBuffer_)
        #TSFramework.TrainStudent(epoch, memoryBuffer)

        currentValue = TSFramework.AddDataBatch(batch)

        #print(TSFramework.memoryUnits)
        #print(np.shape(memoryBuffer))

        arr.append(currentValue)

    #Knolwedge transfer
    KD_epoch = 10
    #TSFramework.KnowledgeTransferForStudent(KD_epoch,memoryBuffer2)

    print("information")
    TSFramework.PrintMemoryInformation()

    arr1 = np.array(arr).astype('str')
    myThirdName = "results/ScoreCurve_MNIST" + "_" + str(threshold) + ".txt"
    #myThirdName = "results/Diffusion_Forgetting_RecoLoss_FirstTaskLearning.txt"
    f = open(myThirdName, "w", encoding="utf-8")
    for i in range(np.shape(arr1)[0]):
        f.writelines(arr1[i])
        f.writelines('\n')
    f.flush()
    f.close()

    arr1 = np.array(classArr).astype('str')
    myThirdName = "results/MNIST_Class"  + "_" + str(threshold) + ".txt"
    #myThirdName = "results/Diffusion_Forgetting_RecoLoss_FirstTaskLearning.txt"
    f = open(myThirdName, "w", encoding="utf-8")
    for i in range(np.shape(arr1)[0]):
        f.writelines(arr1[i])
        f.writelines('\n')
    f.flush()
    f.close()

    arr1 = np.array(componentArr).astype('str')
    myThirdName = "results/MNIST_Component" + "_" + str(threshold) + ".txt"
    #myThirdName = "results/Diffusion_Forgetting_RecoLoss_FirstTaskLearning.txt"
    f = open(myThirdName, "w", encoding="utf-8")
    for i in range(np.shape(arr1)[0]):
        f.writelines(arr1[i])
        f.writelines('\n')
    f.flush()
    f.close()

    end = time.time()
    print("Training times")
    print((end - start))
    print("Finish the training")

    gen = TSFramework.Give_GenerationFromTeacher(100)
    generatedImages = Transfer_To_Numpy(gen)
    name_generation = dataNmae + "_" + modelName + str(threshold) + ".png"
    Save_Image(name_generation,merge2(generatedImages[0:64], [8, 8]))

    #Evaluation
    test_data = torch.tensor(defaultTest).cuda().to(device=device, dtype=torch.float)

    batch = test_data[0:64]
    reco = TSFramework.student.Give_ReconstructionSingle(batch)
    myReco = Transfer_To_Numpy(reco)
    #myReco = merge2(myReco, [8, 8])

    realBatch = Transfer_To_Numpy(batch)
    #realBatch = merge2(realBatch, [8, 8])
    name = dataNmae + "_" + modelName + "_" + "Real_" + str(0) + ".png"
    name_small = dataNmae + "_" + modelName + "_" + "Real_small_" + str(0) + ".png"

    Save_Image(name,merge2(realBatch, [8, 8]))
    Save_Image(name_small,merge2(realBatch[0:16], [2, 8]))

    reco = Transfer_To_Numpy(reco)
    # realBatch = merge2(realBatch, [8, 8])
    name = dataNmae + "_" + modelName + "_" + "Reco_" + str(0) + ".png"
    name_small = dataNmae + "_" + modelName + "_" + "Reco_small_" + str(0) + ".png"

    Save_Image(name, merge2(reco, [8, 8]))
    Save_Image(name_small, merge2(reco[0:16], [2, 8]))

    # Evaluation
    print("Generation")
    generated = TSFramework.Give_GenerationFromTeacher(1000)
    mytest = test_data[0:np.shape(generated)[0]]
    fid1 = calculate_fid_given_paths_Byimages(mytest, generated, 50, device, 2048)
    print(fid1)

    #
    print("Reconstruction")
    
    print("FID")
    #test_data = test_data[0:1000]
    # test_data = torch.tensor(test_data).cuda().to(device=device, dtype=torch.float)
    recoData = TSFramework.student.Give_Reconstruction(test_data)
    # testing = torch.tensor(test_data).cuda().to(device=device, dtype=torch.float)
    # recoData = torch.tensor(recoData).cuda().to(device=device, dtype=torch.float)
    test_data = test_data[0:recoData.size(0)]
    fid1 = calculate_fid_given_paths_Byimages(test_data, recoData, 50, device, 2048)
    print(fid1)

    print("MSE")
    fid1 = GiveMSE(test_data,recoData)
    print(fid1)

    test_data1 = Transfer_To_Numpy(test_data)
    test_data1 = test_data1 / 255.0
    recoData = Transfer_To_Numpy(recoData)
    recoData = recoData / 255.0

    print("psnr and ssim")
    psnr = peak_signal_noise_ratio(test_data1, recoData)
    ssim = structural_similarity(test_data1, recoData, multichannel=True)
    print(psnr)
    print(ssim)

    #Save the model
    myModelName = modelName + str(threshold) + "_" + dataNmae + ".pkl"
    torch.save(TSFramework, './data/' + myModelName)

if __name__ == "__main__":
    main()
    #LoadModel()