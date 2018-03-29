% files = dir('D:\Works\天远视\RGBD-CUT\Datasets\Saliency\Saliency\smoothedDepth\*.mat');
% fileLength = length(files);
% for i = 1 : fileLength
%     load(strcat('D:\Works\天远视\RGBD-CUT\Datasets\Saliency\Saliency\smoothedDepth\',files(i).name));
%     imwrite(smoothedDepth,strcat(files(i).name(1:end-3),'png'),'png')
% end
sample = imread('1_02-02-40_Depth.png')
