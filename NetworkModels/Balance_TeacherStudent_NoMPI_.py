import numpy as np

#from NetworkModels.Teacher_Model_ import Teacher,Balance_Teacher
from NetworkModels.Teacher_Model_NoMPI_ import Balance_Teacher_NoMPI
from NetworkModels.VAE_Model_ import Balance_StudentModel
import torch.nn as nn

class Balance_TeacherStudent_NoMPI(nn.Module):

    def __init__(self,name,device,input_size):
        super(Balance_TeacherStudent_NoMPI, self).__init__()

        self.input_size = input_size
        self.teacher = Balance_Teacher_NoMPI(input_size)
        self.student = Balance_StudentModel(device,input_size)
        self.device = device

    def Train_WithBeta_Cpu_ForStudent(self,Tepoch,Sepoch,data,generatedData,beta):

        if np.shape(generatedData)[0] == 0:
            #self.teacher.train_self_Single_Cpu(Tepoch,data)
            self.student.Train_Self_WithBeta_Single_Cpu(Sepoch,data,beta)
        else:
            #self.teacher.Train_Self_Cpu(Tepoch,data,generatedData)
            self.student.Train_Self_WithBeta_Cpu(Sepoch,data,generatedData,beta)


    def Train_WithBeta_Cpu(self,Tepoch,Sepoch,data,generatedData,beta):

        if np.shape(generatedData)[0] == 0:
            self.teacher.train_self_Single_Cpu(Tepoch,data)
            self.student.Train_Self_WithBeta_Single_Cpu(Sepoch,data,beta)
        else:
            self.teacher.Train_Self_Cpu(Tepoch,data,generatedData)
            self.student.Train_Self_WithBeta_Cpu(Sepoch,data,generatedData,beta)


    def Train_WithBeta_DatLoad(self,Tepoch,Sepoch,data,generatedData,beta):

        if np.shape(generatedData)[0] == 0:
            self.teacher.train_self_Single_Cpu(Tepoch,data)
            self.student.Train_Self_WithBeta_Single_Cpu(Sepoch,data,beta)
        else:
            self.teacher.Train_Self_Cpu(Tepoch,data,generatedData)
            self.student.Train_Self_WithBeta_Cpu(Sepoch,data,generatedData,beta)


    def Train_WithBeta_Cpu_2(self,Tepoch,Sepoch,data,generatedData,beta):

        if np.shape(generatedData)[0] == 0:
            self.teacher.train_self_Single_Cpu(Tepoch,data)
            self.student.Train_Self_WithBeta_Single_Cpu(Sepoch,data,beta)
        else:
            self.teacher.Train_Self_Cpu(Tepoch,data,generatedData)
            self.student.Train_Self_WithBeta_Cpu(Sepoch,data,generatedData,beta)


    def Train_WithBeta(self,Tepoch,Sepoch,data,generatedData,beta):

        if np.shape(generatedData)[0] == 0:
            self.teacher.Train_Self(Tepoch,data)
            self.student.Train_Self_WithBeta_Single(Sepoch,data,beta)
        else:
            self.teacher.Train_Self_(Tepoch,data,generatedData)
            self.student.Train_Self_WithBeta(Sepoch,data,generatedData,beta)

    def Train(self,Tepoch,Sepoch,data,generatedData):

        if np.shape(generatedData)[0] == 0:
            self.teacher.Train_Self(Tepoch,data)
            self.student.Train_Self(Sepoch,data)
        else:
            self.teacher.Train_Self_(Tepoch,data,generatedData)
            self.student.Train_Self_(Sepoch,data,generatedData)


    def Train_ByLoadData_Single(self,Tepoch,Sepoch,data):
        #self.teacher.Train_Self_ByDataLoad_Single(Tepoch, data)
        self.student.Train_Self_ByDataLoad_Single(Sepoch, data)


    def Train_ByLoadData(self,Tepoch,Sepoch,data,generatedData):

        if np.shape(generatedData)[0] == 0:
            self.teacher.Train_Self_ByDataLoad(Tepoch,data)
            self.student.Train_Self(Sepoch,data)
        else:
            self.teacher.Train_Self_ByDataLoad(Tepoch,data,generatedData)
            self.student.Train_Self_ByDataLoad(Sepoch,data,generatedData)


