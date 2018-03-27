
import cv2
import numpy as np
from numpy import *
import maxflow
import math
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage import img_as_ubyte
from dijkstar import Graph, find_path

import datetime

class GraphMaker:

    foreground = 1
    background = 0

    seeds = 0
    segmented = 1

    default = 0.5
    MAXIMUM = 1000000000
    
    ns = 1
    ###各种参数###
    lamda = 0.1
    sigma1 = 0.5
    sigma2 = 4000
    sigma3 = 15

    def __init__(self):
        self.depth = None
        self.image = None
        self.superpixel_image = None
        self.overlay = None
        self.seed_overlay = None
        self.segment_overlay = None
#        衣架衣服
#        self.load_image('../RGB/8_08-40-06.jpg','../transToPNG/8_08-40-06_Depth.png')
#        花瓶
        self.load_image('../RGB/1_02-02-40.jpg','../transToPNG/1_02-02-40_Depth.png')
#        禁止鸣笛
#        self.load_image('../RGB/2_07-12-31.jpg','../transToPNG/2_07-12-31_Depth.png')
#        衣服
#        self.load_image('../RGB/8_08-27-10.jpg','../transToPNG/8_08-27-10_Depth.png')


        self.background_seeds = []
        self.foreground_seeds = []
        self.foreground_superseeds = []
        self.background_superseeds = []
        self.nodes = []
        self.edges = []
        self.current_overlay = self.seeds
        
        self.ave_LAB = None
        self.ave_normal = None
        self.ave_depth = None
        self.confident_map = None
        self.cue_selection = None
        self.superpixel_segment = None
        self.super_edge = None
        
        self.normal_map = None
        self.LAB_map = None
        self.depth_map = None
        self.cfd_LAB = None
        self.cfd_normal = None
        self.cfd_depth = None
        

    def load_image(self, filename, depth_filename):
        self.image = cv2.imread(filename)
        self.image = cv2.resize(self.image, (400,450))
        self.depth = cv2.imread(depth_filename,cv2.IMREAD_ANYDEPTH)
        self.depth = cv2.resize(self.depth, (400,450))
        self.superpixel_image = self.image.copy()
        self.seed_overlay = np.zeros_like(self.image)
        self.segment_overlay = np.zeros_like(self.image)
        
    def add_seed(self, x, y, type):
        if self.image is None:
            print('Please load an image before adding seeds.')
        if type == self.background:
            if not self.background_seeds.__contains__((x, y)):
                self.background_seeds.append((x, y))
                cv2.rectangle(self.seed_overlay, (x-1, y-1), (x+1, y+1), (0, 0, 255), -1)
        elif type == self.foreground:
            if not self.foreground_seeds.__contains__((x, y)):
                self.foreground_seeds.append((x, y))
                cv2.rectangle(self.seed_overlay, (x-1, y-1), (x+1, y+1), (0, 255, 0), -1)

    def clear_seeds(self):
        self.background_seeds = []
        self.foreground_seeds = []
        self.seed_overlay = np.zeros_like(self.seed_overlay)


    def get_image_with_overlay(self, overlayNumber):
        if overlayNumber == self.seeds:
            return cv2.addWeighted(self.image, 0.9, self.seed_overlay, 0.4, 0.1)
        else:
            return cv2.addWeighted(self.image, 0.3, self.segment_overlay, 0.7, 0.1)
    
    def create_graph(self):
        starttime = datetime.datetime.now()
#        self.test()

        
        print("Making graph")        
        #########生成超像素#########
        self.getSuperpixel()
        #########构建map#########
        self.getCueValue()
        #########获取置信图#########
        self.getConfidentMap()    ######！！！！该步骤同时获取了背景点！！！！！######

        if len(self.background_superseeds) == 0 or len(self.foreground_superseeds) == 0:
            print("Please enter at least one foreground and background seed.")
            return
        
        self.populate_graph()
        
        self.cut_graph()
        endtime = datetime.datetime.now()
        print("Complete! Total: " + str((endtime - starttime).seconds))

    def test(self):
        starttime = datetime.datetime.now()
        
        cost_func = lambda u, v, e, prev_e: e['cost']

        depth = self.depth
        G_depth = Graph()
        for i in range(0, len(depth)):
            for j in range(0, len(depth[0])):
                u = i * len(depth[0]) + j
                if i != 0:
                    v = (i - 1) * len(depth[0]) + j
                    w = np.abs(int(depth[i, j]) - int(depth[i - 1, j]))
                    G_depth.add_edge(u, v, {'cost': w})
                    G_depth.add_edge(v, u, {'cost': w})

                if i != len(depth) - 1:
                    v = (i + 1) * len(depth[0]) + j
                    w = np.abs(int(depth[i, j]) - int(depth[i + 1, j]))
                    G_depth.add_edge(u, v, {'cost': w})
                    G_depth.add_edge(v, u, {'cost': w})

                if j != 0:
                    v = i * len(depth[0]) + j - 1
                    w = np.abs(int(depth[i, j]) - int(depth[i, j - 1]))
                    G_depth.add_edge(u, v, {'cost': w})
                    G_depth.add_edge(v, u, {'cost': w})

                if j != len(depth[0]) - 1:
                    v = i * len(depth[0]) + j + 1
                    w = np.abs(int(depth[i, j]) - int(depth[i, j + 1]))
                    G_depth.add_edge(u, v, {'cost': w})
                    G_depth.add_edge(v, u, {'cost': w})

        endtime = datetime.datetime.now()
        print("construct map: " + str((endtime - starttime).seconds))
        starttime = endtime
        for i in range(0, 100):
            info = find_path(G_depth, i, (len(depth) / 2) * len(depth[0]) + len(depth[0]) / 2, cost_func=cost_func)
#            print(info)
#            print(str(i))
#        info = find_path(G_depth, 0, len(depth) * len(depth[0]) - 1, cost_func=cost_func)
        endtime = datetime.datetime.now()
        print("test: " + str((endtime - starttime).seconds))
            
    def getSuperpixel(self):
        starttime = datetime.datetime.now()
        self.superpixel_segment = self.get_superpixel()
        endtime = datetime.datetime.now()
        print("get superpixel: " + str((endtime - starttime).seconds))
         #init_superpixel
        self.n_seg = np.amax(self.superpixel_segment) + 1  #  1 + num for superpixels
        self.num_seg = np.zeros(self.n_seg, dtype=int)    #count for every cluster
        for x in range(0, len(self.superpixel_segment)):
            for y in range(0, len(self.superpixel_segment[0])):
                self.num_seg[self.superpixel_segment[x,y]] += 1
                            
        #n-link for superpixels
        self.super_edge = [[] for i in range(0, self.n_seg)]
        for x in range(0, len(self.superpixel_segment)):
            for y in range(0, len(self.superpixel_segment[0])):
                for i in range(-1,2):
                    for j in range(-1,2):
                        if i == 0 and j == 0:
                            continue
                        if(x+i < 0 or x+i >= len(self.superpixel_segment) or y+j < 0 or y+j >= len(self.superpixel_segment[0])):
                            continue
                        if(self.superpixel_segment[x,y] != self.superpixel_segment[x+i,y+j]):
                            if self.superpixel_segment[x,y] not in self.super_edge[self.superpixel_segment[x+i,y+j]]:
                                self.super_edge[self.superpixel_segment[x+i,y+j]].append(self.superpixel_segment[x,y])
                                self.super_edge[self.superpixel_segment[x,y]].append(self.superpixel_segment[x+i,y+j])
        endtime = datetime.datetime.now()
        print("get superpixel and edge: " + str((endtime - starttime).seconds))
        
        #########check edge on superpixels#########
        temp_img = self.superpixel_image.copy()
        ave_cor = [[0, 0] for i in range(0, self.n_seg)]
        for x in range(0, len(self.superpixel_segment)):
            for y in range(0, len(self.superpixel_segment[0])):
                ave_cor[self.superpixel_segment[x, y]][0] += x
                ave_cor[self.superpixel_segment[x, y]][1] += y
        for i in range(0, self.n_seg):
            ave_cor[i] /= self.num_seg[i]
        for i in range(0, self.n_seg):
            for j in self.super_edge[i]:
                if j < i:
                    continue
                cv2.line(temp_img, (int(ave_cor[i][1]),int(ave_cor[i][0])) ,(int(ave_cor[j][1]), int(ave_cor[j][0])), (0, 0, 255), 1)
        temp_img = temp_img.astype('uint8')
        cv2.imwrite("./results/edgeImg.jpg", temp_img)
        
    def getCueValue(self):
        starttime = datetime.datetime.now()

        self.normal_map = self.normalMap()
        self.LAB_map = self.LABMap()
        self.depth_map = self.depth
        
        #LABMap for superpixel
        self.ave_LAB = np.zeros((self.n_seg,3),dtype=np.float)
        for x in range(0, len(self.superpixel_segment)):
            for y in range(0, len(self.superpixel_segment[0])):
                self.ave_LAB[self.superpixel_segment[x,y]] += self.LAB_map[x,y]
        for i in range (0, self.n_seg):
            self.ave_LAB[i] /= self.num_seg[i]
        #normalMap for superpixel
        self.ave_normal = np.zeros((self.n_seg,3),dtype=float)
        for x in range(0, len(self.superpixel_segment)):
            for y in range(0, len(self.superpixel_segment[0])):
                self.ave_normal[self.superpixel_segment[x,y]] += self.normal_map[x,y]
        for i in range (0, self.n_seg):
            self.ave_normal[i] = self.normalizeVector(self.ave_normal[i])
        #depthMap for superpixel
        self.ave_depth = np.zeros(self.n_seg, dtype=float)
        for x in range(0, len(self.superpixel_segment)):
            for y in range(0, len(self.superpixel_segment[0])):
                self.ave_depth[self.superpixel_segment[x,y]] += self.depth_map[x,y]
        for i in range (0, self.n_seg):
            self.ave_depth[i] /= self.num_seg[i]
            
        endtime = datetime.datetime.now()
        print("get superpixel value for each cue: " + str((endtime - starttime).seconds))
    def getConfidentMap(self):
#==============================================================================
#         我们采用简单有效的重量削减方法来解决这个问题，而不是探索复杂的功能和复杂的补丁外观距离度量。
#         补丁外观距离简单地取为两个斑块的平均颜色（LAB颜色空间）之间的差异（归一化为[0,1]）。 对于
#         每个补丁，我们选择其与所有邻居的最小外观距离，然后我们选择“无意义”距离阈值作为与所有补丁的
#         所有这样最小距离的平均值。如果任何距离小于该阈值，则被认为是不显著的且被限制为0。这种内部
#         边缘权重的计算是非常有效的，其有效性如图3所示。
#==============================================================================
        for coordinate in self.foreground_seeds:
            if self.superpixel_segment[coordinate[1] - 1,coordinate[0] - 1] not in self.foreground_superseeds:
                self.foreground_superseeds.append(self.superpixel_segment[coordinate[1] - 1,coordinate[0] - 1])
        for coordinate in self.background_seeds:
            if self.superpixel_segment[coordinate[1] - 1,coordinate[0] - 1] not in self.background_superseeds:
                self.background_superseeds.append(self.superpixel_segment[coordinate[1] - 1,coordinate[0] - 1])            

        cost_func = lambda u, v, e, prev_e: e['cost']

        #######For LAB foreground#######
        starttime = datetime.datetime.now()

        G_LAB = Graph()
        #add edges
        weight = [[] for i in range(0, self.n_seg)]
        aveMinWeight = 0
        for u in range(0, self.n_seg):
            minWeight = self.MAXIMUM
            for v in self.super_edge[u]:
                w = self.eu_dis(self.ave_LAB[u], self.ave_LAB[v])
                weight[u].append((v, w))
                if minWeight > w:
                    minWeight = w
            aveMinWeight += minWeight
        aveMinWeight /= self.n_seg
        for u in range(0, self.n_seg):
            for v, w in weight[u]:
                if w < aveMinWeight:
                    G_LAB.add_edge(u, v,{'cost': 0})
                else:
                    G_LAB.add_edge(u, v,{'cost': w})
                    
        for v in self.foreground_superseeds:
            G_LAB.add_edge(v, 's', {'cost': 0})

        Lab_disFore = np.zeros(self.n_seg,dtype=float)
        Lab_disFore.fill(self.MAXIMUM)
#        for fg in self.foreground_superseeds:
        for v in range(0, self.n_seg):
            info = find_path(G_LAB, v, 's', cost_func=cost_func)
            Lab_disFore[v] = info.total_cost
                       
        Maxdis = 0
        for i in range(0, self.n_seg):
            if Maxdis < Lab_disFore[i]:
                Maxdis = Lab_disFore[i]
        for i in range(0, self.n_seg):
            Lab_disFore[i] /= Maxdis
        
        endtime = datetime.datetime.now()
        print("compute LAB disfore: " + str((endtime - starttime).seconds))
        #######For Normal foreground#######
        starttime = datetime.datetime.now()
        
        G_normal = Graph()
        #add edges
        weight = [[] for i in range(0, self.n_seg)]
        aveMinWeight = 0
        for u in range(0, self.n_seg):
            minWeight = self.MAXIMUM
            for v in self.super_edge[u]:
                w = 1 - self.cosine_sim(self.ave_normal[u], self.ave_normal[v])
                weight[u].append((v, w))
                if minWeight > w:
                    minWeight = w
            aveMinWeight += minWeight
        aveMinWeight /= self.n_seg
        for u in range(0, self.n_seg):
            for v, w in weight[u]:
                if w < aveMinWeight:
                    G_normal.add_edge(u, v,{'cost': 0})
                else:
                    G_normal.add_edge(u, v,{'cost': w})
                    
        for v in self.foreground_superseeds:
            G_normal.add_edge(v, 's', {'cost': 0})        
    
        Normal_disFore = np.zeros(self.n_seg,dtype=float)
        Normal_disFore.fill(self.MAXIMUM)
        for v in range(0, self.n_seg):
            info = find_path(G_normal, v, 's', cost_func=cost_func)
            Normal_disFore[v] = info.total_cost
    
        Maxdis = 0
        for i in range(0, self.n_seg):
            if Maxdis < Normal_disFore[i]:
                Maxdis = Normal_disFore[i]
        for i in range(0, self.n_seg):
            Normal_disFore[i] /= Maxdis 
                          
        endtime = datetime.datetime.now()
        print("compute Normal disfore: " + str((endtime - starttime).seconds))
        #######For depth foreground#######
        starttime = datetime.datetime.now()

        G_depth = Graph()
        #add edges
        weight = [[] for i in range(0, self.n_seg)]
        aveMinWeight = 0
        for u in range(0, self.n_seg):
            minWeight = self.MAXIMUM
            for v in self.super_edge[u]:
                w = np.abs(self.ave_depth[u] - self.ave_depth[v])
                weight[u].append((v, w))
                if minWeight > w:
                    minWeight = w
            aveMinWeight += minWeight
        aveMinWeight /= self.n_seg
        for u in range(0, self.n_seg):
            for v, w in weight[u]:
                if w < aveMinWeight:
                    G_depth.add_edge(u, v,{'cost': 0})
                else:
                    G_depth.add_edge(u, v,{'cost': w})        

        for v in self.foreground_superseeds:
            G_depth.add_edge(v, 's', {'cost': 0})  

        Depth_disFore = np.zeros(self.n_seg,dtype=float)
        Depth_disFore.fill(self.MAXIMUM)
        for v in range(0, self.n_seg):
            info = find_path(G_depth, v, 's', cost_func=cost_func)
            Depth_disFore[v] = info.total_cost
                         
        Maxdis = 0
        for i in range(0, self.n_seg):
            if Maxdis < Depth_disFore[i]:
                Maxdis = Depth_disFore[i]
        for i in range(0, self.n_seg):
            Depth_disFore[i] /= Maxdis
                         
        endtime = datetime.datetime.now()
        print("compute Depth disfore: " + str((endtime - starttime).seconds))
        #######get background_seed#######
        starttime = datetime.datetime.now()
        
        boundary_superpixel = []
        for x in range(0, len(self.image)):
            if self.superpixel_segment[x, 0] not in boundary_superpixel:
                boundary_superpixel.append(self.superpixel_segment[x, 0])
            if self.superpixel_segment[x, len(self.image[0]) - 1] not in boundary_superpixel:
                boundary_superpixel.append(self.superpixel_segment[x, len(self.image[0]) - 1])
        for y in range(0, len(self.image[0])):
            if self.superpixel_segment[0, y] not in boundary_superpixel:
                boundary_superpixel.append(self.superpixel_segment[0, y])
            if self.superpixel_segment[len(self.image) - 1, y] not in boundary_superpixel:
                boundary_superpixel.append(self.superpixel_segment[len(self.image) - 1, y])
        for boundary in boundary_superpixel:
#            print(Depth_disFore[boundary])
            if Depth_disFore[boundary] > 0.1 and boundary not in self.background_superseeds:
                self.background_superseeds.append(boundary)
        
        
        for v in self.background_superseeds:
            G_normal.add_edge(v, 't', {'cost': 0})
            G_LAB.add_edge(v, 't', {'cost': 0})
            G_depth.add_edge(v, 't', {'cost': 0})
        endtime = datetime.datetime.now()
        print("get background superseeds: " + str((endtime - starttime).seconds))
        
        #######For LAB background#######
        starttime = datetime.datetime.now()

        Lab_disBack = np.zeros(self.n_seg, dtype = float)
        Lab_disBack.fill(self.MAXIMUM)
        for v in range(0, self.n_seg):
            info = find_path(G_LAB, v, 't', cost_func=cost_func)
            Lab_disBack[v] = info.total_cost
                       
        Maxdis = 0
        for i in range(0, self.n_seg):
            if Maxdis < Lab_disBack[i]:
                Maxdis = Lab_disBack[i]
        for i in range(0, self.n_seg):
            Lab_disBack[i] /= Maxdis
        endtime = datetime.datetime.now()
        print("compute LAB disback: " + str((endtime - starttime).seconds))
        #######For Normal background#######
        starttime = datetime.datetime.now()

        Normal_disBack = np.zeros(self.n_seg, dtype = float)
        Normal_disBack.fill(self.MAXIMUM)
        for v in range(0, self.n_seg):
            info = find_path(G_normal, v, 't', cost_func=cost_func)
            Normal_disBack[v] = info.total_cost
                          
        Maxdis = 0
        for i in range(0, self.n_seg):
            if Maxdis < Normal_disBack[i]:
                Maxdis = Normal_disBack[i]
        for i in range(0, self.n_seg):
            Normal_disBack[i] /= Maxdis        
        endtime = datetime.datetime.now()
        print("compute Normal disback: " + str((endtime - starttime).seconds))        
        #######For Depth background#######
        starttime = datetime.datetime.now()

        Depth_disBack = np.zeros(self.n_seg, dtype = float)
        Depth_disBack.fill(self.MAXIMUM)
        for v in range(0, self.n_seg):
            info = find_path(G_depth, v, 't', cost_func=cost_func)
            Depth_disBack[v] = info.total_cost
                         
        Maxdis = 0
        for i in range(0, self.n_seg):
            if Maxdis < Depth_disBack[i]:
                Maxdis = Depth_disBack[i]
        for i in range(0, self.n_seg):
            Depth_disBack[i] /= Maxdis
        endtime = datetime.datetime.now()
        print("compute Depth disback: " + str((endtime - starttime).seconds))           
        #######confident map for each cue#######
        starttime = datetime.datetime.now()

        self.cfd_LAB = np.zeros(self.n_seg, dtype=float)
        self.cfd_normal = np.zeros(self.n_seg, dtype=float)
        self.cfd_depth = np.zeros(self.n_seg, dtype=float)
        self.confident_map = np.zeros(self.n_seg, dtype=float)
        self.cue_selection = np.zeros(self.n_seg, dtype=np.uint8)  ##label for each cues, 0 for default, 1 for LAB, 2 for depth, 3 for normal
        for i in range(0, self.n_seg):
            self.cfd_LAB[i] = Lab_disBack[i] / (Lab_disBack[i] + Lab_disFore[i])
            self.cfd_normal[i] = Normal_disBack[i] / (Normal_disBack[i] + Normal_disFore[i])
            self.cfd_depth[i] = Depth_disBack[i] / (Depth_disBack[i] + Depth_disFore[i])
            
        for i in range(0, self.n_seg):
            temp = 0
            if np.abs(1 - 2 * self.cfd_LAB[i]) > np.abs(1 - 2 * self.cfd_depth[i]):
                self.cue_selection[i] = 1
                temp = np.abs(1 - 2 * self.cfd_LAB[i])
                if np.abs(1 - 2 * self.cfd_normal[i]) > temp:
                    self.cue_selection[i] = 3
                    temp = np.abs(1 - 2 * self.cfd_normal[i])
            else:
                self.cue_selection[i] = 2
                temp = np.abs(1 - 2 * self.cfd_depth[i])
                if np.abs(1 - 2 * self.cfd_normal[i]) > temp:
                    self.cue_selection[i] = 3
                    temp = np.abs(1 - 2 * self.cfd_normal[i])
            if self.cue_selection[i] == 1:
                self.confident_map[i] = self.cfd_LAB[i]
            elif self.cue_selection[i] == 2:
                self.confident_map[i] = self.cfd_depth[i]
            elif self.cue_selection[i] == 3:
                self.confident_map[i] = self.cfd_normal[i]
            else:
                print("wrong label in confident map!")
        endtime = datetime.datetime.now()
        print("compute confident map: " + str((endtime - starttime).seconds))    
                
    def construct_confident(self):
        
        self.segment_overlay = np.zeros_like(self.segment_overlay)
        
        ####生成normal map图
        temp_img = np.zeros_like(self.image)
        for i in range(0, len(temp_img)):
            for j in range(0, len(temp_img[0])):
                temp_img[i, j] = (self.normal_map[i, j] * 0.5 + 0.5) * 255
        temp_img = temp_img.astype('uint8')
        temp_img = cv2.cvtColor(temp_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite("./results/normalMap.jpg", temp_img)
        

        temp_img = np.zeros_like(self.image)
        for i in range(0, len(temp_img)):
            for j in range(0, len(temp_img[0])):
                temp_img[i, j] = self.depth_map[i, j] * 0.1
        temp_img = temp_img.astype('uint8')
        cv2.imwrite("./results/depthMap.jpg", temp_img)        
        

        
        temp_img = self.superpixel_image.astype('uint8')
        cv2.imwrite("./results/sourceImg.jpg", temp_img)
        
        starttime = datetime.datetime.now()



        #########check edge on superpixels#########
        temp_img = self.superpixel_image.copy()
        ave_cor = [[0, 0] for i in range(0, self.n_seg)]
        for x in range(0, len(self.superpixel_segment)):
            for y in range(0, len(self.superpixel_segment[0])):
                ave_cor[self.superpixel_segment[x, y]][0] += x
                ave_cor[self.superpixel_segment[x, y]][1] += y
        for i in range(0, self.n_seg):
            ave_cor[i] /= self.num_seg[i]
        for i in range(0, self.n_seg):
            for j in self.super_edge[i]:
                if j < i:
                    continue
                cv2.line(temp_img, (int(ave_cor[i][1]),int(ave_cor[i][0])) ,(int(ave_cor[j][1]), int(ave_cor[j][0])), (0, 0, 255), 1)
        temp_img = temp_img.astype('uint8')
        cv2.imwrite("./results/edgeImg.jpg", temp_img)
        endtime = datetime.datetime.now()
        print("n_link for superpixel: " + str((endtime - starttime).seconds))
        starttime = endtime
        
        cost_func = lambda u, v, e, prev_e: e['cost']
        
        endtime = datetime.datetime.now()
        print("choose cue for confident map: " + str((endtime - starttime).seconds))
        starttime = endtime              
                                
    @staticmethod   
    def eu_dis(v1, v2):
        return np.sqrt((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2 + (v1[2] - v2[2])**2)
    @staticmethod
    def cosine_sim(vector1,vector2):
        return np.dot(vector1,vector2)/(np.linalg.norm(vector1)*(np.linalg.norm(vector2))) 
    
    def populate_graph(self):
        starttime = datetime.datetime.now()

        self.nodes = []
        self.edges = []
        # make all s and t connections for the graph
        for i in range (0, self.n_seg):
            self.nodes.append((i, 1 - self.confident_map[i], self.confident_map[i]))
        for u in range(0, self.n_seg):
            for v in self.super_edge[u]:
                if self.cue_selection[u] == 1:
                    weight1 = self.lamda * np.e**(-(self.eu_dis(self.ave_LAB[v], self.ave_LAB[u])**2)/self.sigma1)
                elif self.cue_selection[u] == 2:
                    weight1 = self.lamda * np.e**(-(np.abs(self.ave_depth[v] - self.ave_depth[u])**2)/self.sigma2)
                elif self.cue_selection[u] == 3:
                    weight1 = self.lamda * np.e**(-(self.cosine_sim(self.ave_normal[v], self.ave_normal[u])**2)/self.sigma3)
                    
                if self.cue_selection[v] == self.cue_selection[u]:
                    weight2 = weight1
                elif self.cue_selection[v] == 1:
                    weight2 = self.lamda * np.e**(-(self.eu_dis(self.ave_LAB[v], self.ave_LAB[u])**2)/self.sigma1)
                elif self.cue_selection[v] == 2:
                    weight2 = self.lamda * np.e**(-(np.abs(self.ave_depth[v] - self.ave_depth[u])**2)/self.sigma2)
                elif self.cue_selection[v] == 3:
                    weight2 = self.lamda * np.e**(-(self.cosine_sim(self.ave_normal[v], self.ave_normal[u])**2)/self.sigma3)

                weight = min(weight1, weight2)
#                weight = 0
                self.edges.append((u, v, weight))
        endtime = datetime.datetime.now()
        print("compute weight for graphCut: " + str((endtime - starttime).seconds)) 
                
    def cut_graph(self):
        starttime = datetime.datetime.now()

        self.segment_overlay = np.zeros_like(self.segment_overlay)
        g = maxflow.Graph[float](len(self.nodes), len(self.edges))
        
        nodelist = g.add_nodes(len(self.nodes))

        for node in self.nodes:
            g.add_tedge(nodelist[node[0]], node[1], node[2])

        for edge in self.edges:
            g.add_edge(edge[0], edge[1], edge[2], edge[2])

        flow = g.maxflow()
        
        for x in range(0, len(self.superpixel_segment)):
            for y in range(0, len(self.superpixel_segment[0])):
                vect = self.superpixel_segment[x,y]
                if g.get_segment(self.superpixel_segment[x, y]) == 1:
                    if self.cue_selection[vect] == 1:
                        self.segment_overlay[x, y] = (0, 0, 255)
                    elif self.cue_selection[vect] == 2:
                        self.segment_overlay[x, y] = (0, 255, 255)
                    elif self.cue_selection[vect] == 3:
                        self.segment_overlay[x, y] = (0, 153, 255)                    
                else:
                    if self.cue_selection[vect] == 1:
                        self.segment_overlay[x, y] = (0, 255, 0)
                    elif self.cue_selection[vect] == 2:
                        self.segment_overlay[x, y] = (255, 0, 0)
                    elif self.cue_selection[vect] == 3:
                        self.segment_overlay[x, y] = (255, 0, 153)                  
                        
        endtime = datetime.datetime.now()
        
        print("cut graph: " + str((endtime - starttime).seconds)) 
        temp_img = self.image.copy()
        for x in range(0, len(temp_img)):
            for y in range(0, len(temp_img[0])):
                if self.superpixel_segment[x, y] in self.background_superseeds:
                    temp_img[x, y] = (0, 0, 255)
                if self.superpixel_segment[x, y] in self.foreground_superseeds:
                    temp_img[x, y] = (0, 255, 0)
        cv2.imwrite("./results/backseeds.jpg", temp_img)
        
        temp_img = self.superpixel_image.copy()
        cv2.imwrite("./results/superpixel.jpg", temp_img)
        
        temp_img = self.LAB_map.copy()
        cv2.imwrite("./results/LABmap.jpg", temp_img)
        
        temp_img = self.depth_map / 20
        cv2.imwrite("./results/depthmap.jpg", temp_img)

        temp_img = self.normal_map * 255
        cv2.imwrite("./results/normalmap.jpg", temp_img)
        
        temp_img = np.zeros_like(self.image)
        for x in range(0, len(self.superpixel_segment)):
            for y in range(0, len(self.superpixel_segment[0])):
                vect = self.superpixel_segment[x,y]
                temp_img[x, y] = self.cfd_LAB[vect] * 255
        cv2.imwrite("./results/confident_LAB.jpg", temp_img)
        
        temp_img = np.zeros_like(self.image)
        for x in range(0, len(self.superpixel_segment)):
            for y in range(0, len(self.superpixel_segment[0])):
                vect = self.superpixel_segment[x,y]
                temp_img[x, y] = self.cfd_depth[vect] * 255
        cv2.imwrite("./results/confident_depth.jpg", temp_img)     
        
        temp_img = np.zeros_like(self.image)
        for x in range(0, len(self.superpixel_segment)):
            for y in range(0, len(self.superpixel_segment[0])):
                vect = self.superpixel_segment[x,y]
                temp_img[x, y] = self.cfd_normal[vect] * 255
        cv2.imwrite("./results/confident_normal.jpg", temp_img)    
        
        temp_img = np.zeros_like(self.image)
        for x in range(0, len(self.superpixel_segment)):
            for y in range(0, len(self.superpixel_segment[0])):
                vect = self.superpixel_segment[x,y]
                temp_img[x, y] = self.confident_map[vect] * 255
        cv2.imwrite("./results/confident_all.jpg", temp_img)  
        
        temp_img = np.zeros_like(self.image)
        for x in range(0, len(self.superpixel_segment)):
            for y in range(0, len(self.superpixel_segment[0])):
                vect = self.superpixel_segment[x,y]
                if g.get_segment(vect) == 1:
                    temp_img[x, y] = (255, 255, 255)                  
                else:
                    temp_img[x, y] = (0, 0, 0) 
                    
        temp_img = cv2.addWeighted(self.image, 0.2, temp_img, 0.8, 0.1)
        cv2.imwrite("./results/segment.jpg", temp_img)
        
        temp_img = cv2.addWeighted(self.image, 0.2, self.segment_overlay, 0.8, 0.1)
        cv2.imwrite("./results/segment_cues.jpg", temp_img)

    
    
    #SLIC SuperPixel
    def get_superpixel(self):
        segments = slic(self.image, n_segments=800)

        self.superpixel_image = img_as_ubyte(mark_boundaries(self.image, segments))
        
        return segments

    
    #construct normalMap
    def normalMap(self):
        normal_map = np.zeros_like(self.image,dtype=float)
        #normals = np.zeros_like(self.image)
        width = self.depth.shape[1]
        height = self.depth.shape[0]
        
        for x in range(1,height-1):
            for y in range(1,width-1):
                dzdx = ((int)(self.depth[x+1,y]) - (int)(self.depth[x-1,y])) / 2
                dzdy = ((int)(self.depth[x,y+1]) - (int)(self.depth[x,y-1])) / 2
                d = (-dzdx, -dzdy, 1.0)
                n1 = self.normalizeVector(d)
                
                dzdxy = ((int)(self.depth[x+1,y+1]) - (int)(self.depth[x-1,y-1])) / 2.828
                dzdyx = ((int)(self.depth[x-1,y+1]) - (int)(self.depth[x+1,y-1])) / 2.828
                d = ((dzdyx-dzdxy),(-dzdxy-dzdyx),2)
                n2 = self.normalizeVector(d)
                
                d = n1 + n2
                n = self.normalizeVector(d)
                
                normal_map[x,y] = n
        return normal_map
        
    #construct LABMap
    def LABMap(self):
        LAB_map_raw = cv2.cvtColor(self.image, cv2.COLOR_RGB2Lab)
        LAB_map = np.zeros_like(LAB_map_raw,dtype=np.int8)
        for i in range(len(LAB_map)):
            for j in range(len(LAB_map[0])):
                LAB_map[i,j][0] = LAB_map_raw[i,j][0] / 255 * 100
                LAB_map[i,j][1] = LAB_map_raw[i,j][1] - 128
                LAB_map[i,j][2] = LAB_map_raw[i,j][2] - 128
        return LAB_map
        
    #####归一化#####
    @staticmethod
    def normalizeVector(d):
        return d / (np.sqrt(d[0]**2 + d[1]**2 + d[2]**2))