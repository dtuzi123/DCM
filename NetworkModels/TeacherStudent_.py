from NetworkModels.Teacher_Model_ import Teacher
from NetworkModels.VAE_Model_ import StudentModel
import torch.nn as nn

class TeacherStudent(nn.Module):

    def __init__(self,name,device,input_size):
        super(TeacherStudent, self).__init__()

        self.input_size = input_size
        self.teacher = Teacher(input_size)
        self.student = StudentModel(device,input_size)
        self.device = device

    def Train(self,Tepoch,Sepoch,data):
        self.teacher.Train_Self(Tepoch,data)
        self.student.Train_Self(Sepoch,data)

    def Train_StudentOnly(self,Tepoch,Sepoch,data):
        self.student.Train_Self(Sepoch,data)


