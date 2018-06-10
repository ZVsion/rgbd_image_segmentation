import cv2
import numpy as np
import maxflow
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage import img_as_ubyte
from dijkstar import Graph, find_path
import os

import datetime


class GraphMaker:
    foreground = 1
    background = 0

    seeds = 0
    segmented = 1

    default = 0.5
    MAXIMUM = 1000000000

    ImageName = "8_08-40-06"

    ns = 1
    ###parameters###
    lamda = 0.2
    NORMAL = True  # Consider normal vector or not?

    def __init__(self):
        self.depth = None
        self.image = None
        self.superpixel_image = None
        self.overlay = None
        self.seed_overlay = None
        self.segment_overlay = None
        self.load_image(os.path.join('Imgs', 'RGB', self.ImageName+'.jpg'),
                        os.path.join('Imgs', 'Depth', self.ImageName+'_Depth.png'))
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
        self.super_label = None  # label of superpixels，0~5 for
                                 # 0:LAB FG  1:LAB BG  2:depth FG
                                 # 3:depth BG  4:normal FG  5:normal BG

    def computeSigma(self):

        self.sigma1 = 0
        self.sigma2 = 0
        self.sigma3 = 0

        for i in range(self.n_seg):
            for j in self.super_edge[i]:
                if j > i:
                    if self.sigma1 < self.eu_dis(self.ave_LAB[i], self.ave_LAB[j]):
                        self.sigma1 = self.eu_dis(self.ave_LAB[i], self.ave_LAB[j])
                    if self.sigma2 < np.abs(self.ave_depth[i] - self.ave_depth[j]):
                        self.sigma2 = np.abs(self.ave_depth[i] - self.ave_depth[j])
                    if self.NORMAL and self.sigma3 < (1 - self.cosine_sim(self.ave_normal[i], self.ave_normal[j])):
                        self.sigma3 = (1 - self.cosine_sim(self.ave_normal[i], self.ave_normal[j]))

        self.sigma1 = self.sigma1 ** 2 * 1
        self.sigma2 = self.sigma2 ** 2 * 1
        self.sigma3 = self.sigma3 ** 2 * 0.5
        print("sigma1 = " + str(self.sigma1))
        print("sigma2 = " + str(self.sigma2))
        print("sigma3 = " + str(self.sigma3))

    def load_image(self, filename, depth_filename):
        self.image = cv2.imread(filename)
        self.depth = cv2.imread(depth_filename, cv2.IMREAD_ANYDEPTH)
        self.superpixel_image = self.image.copy()
        self.seed_overlay = np.zeros_like(self.image)
        self.segment_overlay = np.zeros_like(self.image)

    def add_seed(self, x, y, type):
        if self.image is None:
            print('Please load an image before adding seeds.')
        if type == self.background:
            if not self.background_seeds.__contains__((x, y)):
                self.background_seeds.append((x, y))
                cv2.rectangle(self.seed_overlay, (x - 1, y - 1), (x + 1, y + 1), (0, 0, 255), -1)
        elif type == self.foreground:
            if not self.foreground_seeds.__contains__((x, y)):
                self.foreground_seeds.append((x, y))
                cv2.rectangle(self.seed_overlay, (x - 1, y - 1), (x + 1, y + 1), (0, 255, 0), -1)

    def clear_seeds(self):
        self.background_seeds = []
        self.foreground_seeds = []
        self.background_superseeds = []
        self.foreground_superseeds = []
        self.seed_overlay = np.zeros_like(self.seed_overlay)

    def get_image_with_overlay(self, overlayNumber):
        if overlayNumber == self.seeds:
            return cv2.addWeighted(self.image, 0.9, self.seed_overlay, 0.4, 0.1)
        else:
            return cv2.addWeighted(self.image, 0.3, self.segment_overlay, 0.7, 0.1)

    def create_graph(self):
        starttime = datetime.datetime.now()
        print("Making graph")
        self.getSuperpixel()  # get superpixel
        self.getCueValue_mean()  # calculate average value of superpixels
        self.getConfidentMap()  # meanwhile we get background seeds in this func

        if len(self.background_superseeds) == 0 or len(self.foreground_superseeds) == 0:
            print("Please enter at least one foreground and background seed.")
            return

        self.computeSigma()  # get sigma

        # clear results
        if not os.path.exists(os.path.join('Results', self.ImageName)):
            os.mkdir(os.path.join('Results', self.ImageName))
        for item in os.listdir(os.path.join('Results', self.ImageName)):
            itemsrc = os.path.join(os.path.join('Results', self.ImageName), item)
            os.remove(itemsrc)

        self.swap(10)  # alpha-beta swap
        endtime = datetime.datetime.now()
        print("total run time: " + str((endtime - starttime).seconds))

        self.getImage()

    def getSuperpixel(self):
        starttime = datetime.datetime.now()
        self.superpixel_segment = self.get_superpixel()
        # init_superpixel
        self.n_seg = np.amax(self.superpixel_segment) + 1  # 1 + num for superpixels
        self.num_seg = np.zeros(self.n_seg, dtype=int)  # count for every cluster
        for x in range(0, len(self.superpixel_segment)):
            for y in range(0, len(self.superpixel_segment[0])):
                self.num_seg[self.superpixel_segment[x, y]] += 1
        self.super_label = np.ones(self.n_seg, dtype=int) * 0

        self.get_SuperEdge(self.superpixel_segment)

        endtime = datetime.datetime.now()
        print("get superpixel: " + str((endtime - starttime).seconds))

    def getCueValue_mean(self):
        starttime = datetime.datetime.now()

        self.normal_map = self.normalMap()
        self.LAB_map = self.LABMap()
        self.depth_map = self.depth

        # normalMap for superpixel
        if self.NORMAL:
            self.ave_normal = np.zeros((self.n_seg, 3), dtype=float)
            for x in range(0, len(self.superpixel_segment)):
                for y in range(0, len(self.superpixel_segment[0])):
                    self.ave_normal[self.superpixel_segment[x, y]] += self.normal_map[x, y]
            for i in range(0, self.n_seg):
                self.ave_normal[i] = self.normalizeVector(self.ave_normal[i])

        # depthMap for superpixel
        self.ave_depth = np.zeros(self.n_seg, dtype=float)
        for x in range(0, len(self.superpixel_segment)):
            for y in range(0, len(self.superpixel_segment[0])):
                self.ave_depth[self.superpixel_segment[x, y]] += self.depth_map[x, y]
        for i in range(0, self.n_seg):
            self.ave_depth[i] /= self.num_seg[i]

        # LABMap for superpixel
        self.ave_LAB = np.zeros((self.n_seg, 3), dtype=float)
        for x in range(0, len(self.superpixel_segment)):
            for y in range(0, len(self.superpixel_segment[0])):
                self.ave_LAB[self.superpixel_segment[x, y]] += self.image[x, y]
        for i in range(0, self.n_seg):
            self.ave_LAB[i] /= self.num_seg[i]

        endtime = datetime.datetime.now()
        print("get mean-super-value for each cue: " + str((endtime - starttime).seconds))

    def getConfidentMap(self):
        starttime = datetime.datetime.now()
        for coordinate in self.foreground_seeds:
            if self.superpixel_segment[coordinate[1] - 1, coordinate[0] - 1] not in self.foreground_superseeds:
                self.foreground_superseeds.append(self.superpixel_segment[coordinate[1] - 1, coordinate[0] - 1])
        for coordinate in self.background_seeds:
            if self.superpixel_segment[coordinate[1] - 1, coordinate[0] - 1] not in self.background_superseeds:
                self.background_superseeds.append(self.superpixel_segment[coordinate[1] - 1, coordinate[0] - 1])

        cost_func = lambda u, v, e, prev_e: e['cost']

        #######For LAB foreground#######
        G_LAB = Graph()
        # add edges
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
                    G_LAB.add_edge(u, v, {'cost': w / 3})
                else:
                    G_LAB.add_edge(u, v, {'cost': w})

        for v in self.foreground_superseeds:
            G_LAB.add_edge(v, 's', {'cost': 0})

        Lab_disFore = np.zeros(self.n_seg, dtype=float)
        Lab_disFore.fill(self.MAXIMUM)

        for v in range(0, self.n_seg):
            info = find_path(G_LAB, v, 's', cost_func=cost_func)
            Lab_disFore[v] = info.total_cost

        endtime = datetime.datetime.now()
        print("compute LAB disfore: " + str((endtime - starttime).seconds))
        #######For Normal foreground#######
        if self.NORMAL:
            starttime = datetime.datetime.now()

            G_normal = Graph()
            # add edges
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
                        G_normal.add_edge(u, v, {'cost': w / 3})
                    else:
                        G_normal.add_edge(u, v, {'cost': w})

            for v in self.foreground_superseeds:
                G_normal.add_edge(v, 's', {'cost': 0})

            Normal_disFore = np.zeros(self.n_seg, dtype=float)
            Normal_disFore.fill(self.MAXIMUM)
            for v in range(0, self.n_seg):
                info = find_path(G_normal, v, 's', cost_func=cost_func)
                Normal_disFore[v] = info.total_cost

            endtime = datetime.datetime.now()
            print("compute Normal disfore: " + str((endtime - starttime).seconds))
        #######For depth foreground#######
        starttime = datetime.datetime.now()

        G_depth = Graph()
        # add edges
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
                    G_depth.add_edge(u, v, {'cost': w / 3})
                else:
                    G_depth.add_edge(u, v, {'cost': w})

        for v in self.foreground_superseeds:
            G_depth.add_edge(v, 's', {'cost': 0})

        Depth_disFore = np.zeros(self.n_seg, dtype=float)
        Depth_disFore.fill(self.MAXIMUM)
        for v in range(0, self.n_seg):
            info = find_path(G_depth, v, 's', cost_func=cost_func)
            Depth_disFore[v] = info.total_cost

        Maxdis = 0
        for i in range(0, self.n_seg):
            if Maxdis < Depth_disFore[i]:
                Maxdis = Depth_disFore[i]

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
            if Depth_disFore[boundary] / Maxdis > 0.1 and boundary not in self.background_superseeds:
                self.background_superseeds.append(boundary)

        for v in self.background_superseeds:
            if self.NORMAL:
                G_normal.add_edge(v, 't', {'cost': 0})
            G_LAB.add_edge(v, 't', {'cost': 0})
            G_depth.add_edge(v, 't', {'cost': 0})
        endtime = datetime.datetime.now()
        print("get background superseeds: " + str((endtime - starttime).seconds))

        #######For LAB background#######
        starttime = datetime.datetime.now()
        Lab_disBack = np.zeros(self.n_seg, dtype=float)
        Lab_disBack.fill(self.MAXIMUM)
        for v in range(0, self.n_seg):
            info = find_path(G_LAB, v, 't', cost_func=cost_func)
            Lab_disBack[v] = info.total_cost

        endtime = datetime.datetime.now()
        print("compute LAB disback: " + str((endtime - starttime).seconds))
        #######For Normal background#######
        if self.NORMAL:
            starttime = datetime.datetime.now()

            Normal_disBack = np.zeros(self.n_seg, dtype=float)
            Normal_disBack.fill(self.MAXIMUM)
            for v in range(0, self.n_seg):
                info = find_path(G_normal, v, 't', cost_func=cost_func)
                Normal_disBack[v] = info.total_cost

            endtime = datetime.datetime.now()
            print("compute Normal disback: " + str((endtime - starttime).seconds))
        #######For Depth background#######
        starttime = datetime.datetime.now()

        Depth_disBack = np.zeros(self.n_seg, dtype=float)
        Depth_disBack.fill(self.MAXIMUM)
        for v in range(0, self.n_seg):
            info = find_path(G_depth, v, 't', cost_func=cost_func)
            Depth_disBack[v] = info.total_cost

        endtime = datetime.datetime.now()
        print("compute Depth disback: " + str((endtime - starttime).seconds))
        #######confident map for each cue#######
        starttime = datetime.datetime.now()

        self.cfd_LAB = np.zeros(self.n_seg, dtype=float)
        if self.NORMAL:
            self.cfd_normal = np.zeros(self.n_seg, dtype=float)
        self.cfd_depth = np.zeros(self.n_seg, dtype=float)

        for i in range(0, self.n_seg):
            self.cfd_LAB[i] = Lab_disBack[i] / (Lab_disBack[i] + Lab_disFore[i])
            if self.NORMAL:
                self.cfd_normal[i] = Normal_disBack[i] / (Normal_disBack[i] + Normal_disFore[i])
            self.cfd_depth[i] = Depth_disBack[i] / (Depth_disBack[i] + Depth_disFore[i])

        endtime = datetime.datetime.now()
        print("compute confident map: " + str((endtime - starttime).seconds))

    def swap(self, MaxIteration=4):
        starttime = datetime.datetime.now()

        oldEnergy = self.computeEnerge()
        print("initial energe：" + str(oldEnergy))

        for i in range(MaxIteration):

            self.oneSwapIteration(i)
            newEnergy = self.computeEnerge()

            print("iter" + str(i + 1) + " Energe：" + str(newEnergy))
            if newEnergy >= oldEnergy:
                break
            oldEnergy = newEnergy
        endtime = datetime.datetime.now()
        print("alpha-beta-swap: " + str((endtime - starttime).seconds))

    def oneSwapIteration(self, iteration):
        # self.get_seg_now(str(iteration) + "-00")
        old = self.computeEnerge()
        for i in range(0, 6):
            if not self.NORMAL and i >= 4:
                continue
            for j in range(i + 1, 6):
                if not self.NORMAL and j >= 4:
                    continue
                self.alpha_beta_swap(i, j)
                new = self.computeEnerge()
                # if old > new:
                #     self.get_seg_now(str(iteration) + "-" + str(i) + str(j))
                old = new

    def alpha_beta_swap(self, alpha, beta):
        ###add nodes and edges for graphCuts###
        self.nodes = []
        self.edges = []
        reflect = []  ##discontinuity to continuity

        if alpha == 0:
            cap_source = 1 - self.cfd_LAB
        elif alpha == 1:
            cap_source = self.cfd_LAB
        elif alpha == 2:
            cap_source = 1 - self.cfd_depth
        elif alpha == 3:
            cap_source = self.cfd_depth
        elif alpha == 4:
            cap_source = 1 - self.cfd_normal
        elif alpha == 5:
            cap_source = self.cfd_normal

        if beta == 0:
            cap_sink = 1 - self.cfd_LAB
        elif beta == 1:
            cap_sink = self.cfd_LAB
        elif beta == 2:
            cap_sink = 1 - self.cfd_depth
        elif beta == 3:
            cap_sink = self.cfd_depth
        elif beta == 4:
            cap_sink = 1 - self.cfd_normal
        elif beta == 5:
            cap_sink = self.cfd_normal

        for i in range(self.n_seg):
            if self.super_label[i] == alpha or self.super_label[i] == beta:
                reflect.append(i)
                source_add = 0
                sink_add = 0
                for neighbor in self.super_edge[i]:
                    if self.super_label[neighbor] != alpha and self.super_label[neighbor] != beta:
                        source_add += self.smoothWeight(i, neighbor, alpha)
                        sink_add += self.smoothWeight(i, neighbor, beta)
                self.nodes.append((reflect.index(i), cap_source[i] + source_add, cap_sink[i] + sink_add))

        for n in self.nodes:
            u = reflect[n[0]]
            for v in self.super_edge[u]:
                if (self.super_label[v] == alpha or self.super_label[v] == beta) and v > u:
                    # print(str(u) + "," + str(v))
                    weight = self.smoothWeight(u, v)
                    self.edges.append((reflect.index(u), reflect.index(v), weight))

        ####GraphCuts####
        g = maxflow.Graph[float](len(self.nodes), len(self.edges))

        nodelist = g.add_nodes(len(self.nodes))
        for node in self.nodes:
            g.add_tedge(nodelist[node[0]], node[1], node[2])

        for edge in self.edges:
            g.add_edge(edge[0], edge[1], edge[2], edge[2])

        flow = g.maxflow()

        for vect in self.nodes:
            v = vect[0]
            if g.get_segment(v) == 0:  # beta
                self.super_label[reflect[v]] = beta
            else:  # alpha
                self.super_label[reflect[v]] = alpha

    def computeEnerge(self):
        return (self.giveDataEnerge() + self.giveSmoothEnerge())

    def giveDataEnerge(self):
        energe = 0
        for i in range(self.n_seg):
            if self.super_label[i] == 0:
                energe += (1 - self.cfd_LAB[i])
            elif self.super_label[i] == 1:
                energe += self.cfd_LAB[i]
            elif self.super_label[i] == 2:
                energe += (1 - self.cfd_depth[i])
            elif self.super_label[i] == 3:
                energe += self.cfd_depth[i]
            elif self.super_label[i] == 4:
                energe += (1 - self.cfd_normal[i])
            elif self.super_label[i] == 5:
                energe += self.cfd_normal[i]
            if energe != energe:
                print(str(i) + ":" + str(self.super_label[i]))
                break
        return energe

    def giveSmoothEnerge(self):  # compute SmoothEnerge
        energe = 0
        for u in range(self.n_seg):
            for v in self.super_edge[u]:
                if v < u:
                    continue
                if self.super_label[u] == self.super_label[v]:
                    continue
                energe += self.smoothWeight(u, v)
        return energe

    def smoothWeight(self, u, v, alpha=-1):
        if u not in self.super_edge[v] or v not in self.super_edge[u]:
            return 0

        if alpha == -1:
            alpha = self.super_label[u]

        if alpha == 0 or alpha == 1:
            weight1 = self.lamda * np.e ** (-(self.eu_dis(self.ave_LAB[u], self.ave_LAB[v]) ** 2) / self.sigma1)
        elif alpha == 2 or alpha == 3:
            weight1 = self.lamda * np.e ** (-(np.abs(self.ave_depth[u] - self.ave_depth[v]) ** 2) / self.sigma2)
        elif alpha == 4 or alpha == 5:
            weight1 = self.lamda * np.e ** (
                    -((1 - self.cosine_sim(self.ave_normal[v], self.ave_normal[u])) ** 2) / self.sigma3)

        if self.super_label[v] == 0 or self.super_label[v] == 1:
            weight2 = self.lamda * np.e ** (-(self.eu_dis(self.ave_LAB[u], self.ave_LAB[v]) ** 2) / self.sigma1)
        elif self.super_label[v] == 2 or self.super_label[v] == 3:
            weight2 = self.lamda * np.e ** (-(np.abs(self.ave_depth[u] - self.ave_depth[v]) ** 2) / self.sigma2)
        elif self.super_label[v] == 4 or self.super_label[v] == 5:
            weight2 = self.lamda * np.e ** (
                    -((1 - self.cosine_sim(self.ave_normal[v], self.ave_normal[u])) ** 2) / self.sigma3)

        return min(weight1, weight2)

    @staticmethod
    def eu_dis(v1, v2):
        return np.sqrt((v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2 + (v1[2] - v2[2]) ** 2)

    @staticmethod
    def cosine_sim(vector1, vector2):
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * (np.linalg.norm(vector2)))

    # SLIC SuperPixel
    def get_superpixel(self):
        segments = slic(self.image, n_segments=800)

        self.superpixel_image = img_as_ubyte(mark_boundaries(self.image, segments))

        return segments

    def get_SuperEdge(self, segments):
        self.super_edge = [[] for _ in range(0, self.n_seg)]
        for x in range(0, len(segments)):
            for y in range(0, len(segments[0])):
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if i == 0 and j == 0:
                            continue
                        if (x + i < 0 or x + i >= len(segments) or y + j < 0 or y + j >= len(
                                segments[0])):
                            continue
                        if (segments[x, y] != segments[x + i, y + j]):
                            if segments[x, y] not in self.super_edge[
                                segments[x + i, y + j]]:
                                self.super_edge[segments[x + i, y + j]].append(
                                    segments[x, y])
                                self.super_edge[segments[x, y]].append(
                                    segments[x + i, y + j])

    # construct normalMap
    def normalMap(self):
        normal_map = np.zeros_like(self.image, dtype=float)
        width = self.depth.shape[1]
        height = self.depth.shape[0]

        for x in range(1, height - 1):
            for y in range(1, width - 1):
                dzdx = ((int)(self.depth[x + 1, y]) - (int)(self.depth[x - 1, y])) / 2
                dzdy = ((int)(self.depth[x, y + 1]) - (int)(self.depth[x, y - 1])) / 2
                d = (-dzdx, -dzdy, 1.0)
                n1 = self.normalizeVector(d)

                dzdxy = ((int)(self.depth[x + 1, y + 1]) - (int)(self.depth[x - 1, y - 1])) / 2.828
                dzdyx = ((int)(self.depth[x - 1, y + 1]) - (int)(self.depth[x + 1, y - 1])) / 2.828
                d = ((dzdyx - dzdxy), (-dzdxy - dzdyx), 2)
                n2 = self.normalizeVector(d)

                d = n1 + n2
                n = self.normalizeVector(d)

                normal_map[x, y] = n
        return normal_map

    # construct LABMap
    def LABMap(self):
        LAB_map_raw = cv2.cvtColor(self.image, cv2.COLOR_RGB2Lab)
        LAB_map = np.zeros_like(LAB_map_raw, dtype=np.int8)
        for i in range(len(LAB_map)):
            for j in range(len(LAB_map[0])):
                LAB_map[i, j][0] = LAB_map_raw[i, j][0] / 255 * 100
                LAB_map[i, j][1] = LAB_map_raw[i, j][1] - 128
                LAB_map[i, j][2] = LAB_map_raw[i, j][2] - 128
        return LAB_map

    @staticmethod
    def normalizeVector(d):
        return d / (np.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2))

    def compute_alpha_beta_energy(self, alpha, beta, reflect):
        energy = 0

        if alpha == 0:
            cap_source = 1 - self.cfd_LAB
        elif alpha == 1:
            cap_source = self.cfd_LAB
        elif alpha == 2:
            cap_source = 1 - self.cfd_depth
        elif alpha == 3:
            cap_source = self.cfd_depth
        elif alpha == 4:
            cap_source = 1 - self.cfd_normal
        elif alpha == 5:
            cap_source = self.cfd_normal

        if beta == 0:
            cap_sink = 1 - self.cfd_LAB
        elif beta == 1:
            cap_sink = self.cfd_LAB
        elif beta == 2:
            cap_sink = 1 - self.cfd_depth
        elif beta == 3:
            cap_sink = self.cfd_depth
        elif beta == 4:
            cap_sink = 1 - self.cfd_normal
        elif beta == 5:
            cap_sink = self.cfd_normal

        energyout = 0
        energyadd = 0
        energysmoothout = 0
        # for node in self.nodes:
        for i in range(self.n_seg):
            if self.super_label[i] == alpha:
                energy += cap_source[i]
            elif self.super_label[i] == beta:
                energy += cap_sink[i]
            else:
                if self.super_label[i] == 0:
                    energyout += (1 - self.cfd_LAB[i])
                elif self.super_label[i] == 1:
                    energyout += self.cfd_LAB[i]
                elif self.super_label[i] == 2:
                    energyout += (1 - self.cfd_depth[i])
                elif self.super_label[i] == 3:
                    energyout += self.cfd_depth[i]
                elif self.super_label[i] == 4:
                    energyout += (1 - self.cfd_normal[i])
                elif self.super_label[i] == 5:
                    energyout += self.cfd_normal[i]

            if self.super_label[i] == alpha or self.super_label[i] == beta:
                for neighbor in self.super_edge[i]:
                    if self.super_label[neighbor] != alpha and self.super_label[neighbor] != beta:
                        energyadd += self.smoothWeight(i, neighbor)
            else:
                for neighbor in self.super_edge[i]:
                    if self.super_label[neighbor] != alpha and self.super_label[neighbor] != beta and neighbor > i:
                        energysmoothout += self.smoothWeight(i, neighbor)


        energysmooth = 0
        for edge in self.edges:
            if self.super_label[reflect[edge[0]]] != self.super_label[reflect[edge[1]]]:
                energysmooth += edge[2]
        energy += (energysmooth)
        return energy

    def getImage(self):
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
                cv2.line(temp_img, (int(ave_cor[i][1]), int(ave_cor[i][0])), (int(ave_cor[j][1]), int(ave_cor[j][0])),
                         (0, 0, 255), 1)
        temp_img = temp_img.astype('uint8')
        cv2.imwrite(os.path.join('Results', self.ImageName, "edgeImg.jpg"), temp_img)
        ###get overlay###
        for x in range(0, len(self.superpixel_segment)):
            for y in range(0, len(self.superpixel_segment[0])):
                vect = self.superpixel_segment[x, y]
                if self.super_label[vect] % 2 == 0:
                    self.segment_overlay[x, y] = (255, 255, 255)
                else:
                    self.segment_overlay[x, y] = (0, 0, 0)

        ###get image with annotation###
        temp_img = self.image.copy()
        for x in range(0, len(temp_img)):
            for y in range(0, len(temp_img[0])):
                if self.superpixel_segment[x, y] in self.background_superseeds:
                    temp_img[x, y] = (0, 0, 255)
                if self.superpixel_segment[x, y] in self.foreground_superseeds:
                    temp_img[x, y] = (0, 255, 0)
        cv2.imwrite(os.path.join('Results', self.ImageName, "backseeds.jpg"), temp_img)

        ###get superpixels image###
        temp_img = self.superpixel_image.copy()
        cv2.imwrite(os.path.join('Results', self.ImageName, "superpixel.jpg"), temp_img)

        ###get LAB image###
        temp_img = self.LAB_map.copy()
        cv2.imwrite(os.path.join('Results', self.ImageName, "LABmap.jpg"), temp_img)

        ###get depth image###
        temp_img = self.depth_map / np.amax(self.depth_map) * 255
        cv2.imwrite(os.path.join('Results', self.ImageName, "depthmap.jpg"), temp_img)

        ###get normal image###
        if self.NORMAL:
            temp_img = self.normal_map * 255
            cv2.imwrite(os.path.join('Results', self.ImageName, "normalmap.jpg"), temp_img)

        ###get configent map of lab###
        temp_img = np.zeros_like(self.image)
        for x in range(0, len(self.superpixel_segment)):
            for y in range(0, len(self.superpixel_segment[0])):
                vect = self.superpixel_segment[x, y]
                temp_img[x, y] = self.cfd_LAB[vect] * 255
        cv2.imwrite(os.path.join('Results', self.ImageName, "confident_LAB.jpg"), temp_img)

        ###get confident map of depth###
        temp_img = np.zeros_like(self.image)
        for x in range(0, len(self.superpixel_segment)):
            for y in range(0, len(self.superpixel_segment[0])):
                vect = self.superpixel_segment[x, y]
                temp_img[x, y] = self.cfd_depth[vect] * 255
        cv2.imwrite(os.path.join('Results', self.ImageName, "confident_depth.jpg"), temp_img)
        ###get confident map of normal###
        if self.NORMAL:
            temp_img = np.zeros_like(self.image)
            for x in range(0, len(self.superpixel_segment)):
                for y in range(0, len(self.superpixel_segment[0])):
                    vect = self.superpixel_segment[x, y]
                    temp_img[x, y] = self.cfd_normal[vect] * 255
            cv2.imwrite(os.path.join('Results', self.ImageName, "confident_normal.jpg"), temp_img)

        ###get Results###
        temp_img = cv2.addWeighted(self.image, 0.2, self.segment_overlay, 0.8, 0.1)
        cv2.imwrite(os.path.join('Results', self.ImageName, "segment.jpg"), temp_img)
        ###Results with cues###
        self.segment_overlay = np.zeros_like(self.image)
        for x in range(0, len(self.superpixel_segment)):
            for y in range(0, len(self.superpixel_segment[0])):
                vect = self.superpixel_segment[x, y]
                if self.super_label[vect] == 0:
                    self.segment_overlay[x, y] = (0, 0, 255)
                elif self.super_label[vect] == 1:
                    self.segment_overlay[x, y] = (0, 255, 0)
                elif self.super_label[vect] == 2:
                    self.segment_overlay[x, y] = (0, 255, 255)
                elif self.super_label[vect] == 3:
                    self.segment_overlay[x, y] = (255, 0, 0)
                elif self.super_label[vect] == 4:
                    self.segment_overlay[x, y] = (0, 153, 255)
                elif self.super_label[vect] == 5:
                    self.segment_overlay[x, y] = (255, 0, 153)

        temp_img = cv2.addWeighted(self.image, 0.2, self.segment_overlay, 0.8, 0.1)
        cv2.imwrite(os.path.join('Results', self.ImageName, "segment_cues.jpg"), temp_img)

        temp_img = np.zeros_like(self.image)
        w = np.amax(self.depth_map)
        for x in range(0, len(self.superpixel_segment)):
            for y in range(0, len(self.superpixel_segment[0])):
                temp_img[x, y] = self.ave_depth[self.superpixel_segment[x, y]] / w * 255
        cv2.imwrite(os.path.join('Results', self.ImageName, "ave_depth.jpg"), temp_img)

    def get_seg_now(self, path):  # get current seg
        for x in range(0, len(self.superpixel_segment)):
            for y in range(0, len(self.superpixel_segment[0])):
                vect = self.superpixel_segment[x, y]
                if self.super_label[vect] == 0:
                    self.segment_overlay[x, y] = (0, 0, 255)
                elif self.super_label[vect] == 1:
                    self.segment_overlay[x, y] = (0, 255, 0)
                elif self.super_label[vect] == 2:
                    self.segment_overlay[x, y] = (0, 255, 255)
                elif self.super_label[vect] == 3:
                    self.segment_overlay[x, y] = (255, 0, 0)
                elif self.super_label[vect] == 4:
                    self.segment_overlay[x, y] = (0, 153, 255)
                elif self.super_label[vect] == 5:
                    self.segment_overlay[x, y] = (255, 0, 153)
        temp = img_as_ubyte(mark_boundaries(self.get_image_with_overlay(self.segmented), self.superpixel_segment))
        cv2.imwrite(os.path.join('Results', self.ImageName, path + ".jpg"), temp)
        self.segment_overlay = np.zeros_like(self.segment_overlay)
