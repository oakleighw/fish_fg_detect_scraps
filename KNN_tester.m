clear,clc,close all;


initDir = "C:\Users\Computing\OneDrive - University of Lincoln (1)\Documents\1fish_tracking\dataset";
data = initDir + "/wetransfer-b9b88f/wetransfer-b9b88f/multi_trackdata_61ac.mp4";

d = "C:\Users\Computing\OneDrive - University of Lincoln (1)\Documents\1fish_tracking\dataset\pooled_Vids\1_6_salmon.mp4";
videoSource = VideoReader(d);
fs = {};

n = videoSource.NumFrames;


import clib.opencv.*;
import vision.opencv.*;
history = 15;
threshold = 50;
shadow = true;

cvPtr = cv.createBackgroundSubtractorKNN(history,threshold,shadow);
kNNBase = util.getBasePtr(cvPtr);

passed = false;

kNNBase.setkNNSamples(3);% k nearest neighbours

foregroundmask = zeros(videoSource.Height,videoSource.Width,videoSource.NumFrames);

se = strel("disk", 15);


fishCount = 0;

%mask out part of frame
cents = [];%centroids of detections


%[r,c] = size(readFrame(videoSource));

vidName = "outputs/KNN";
fname = vidName + "_good";
v = VideoWriter(fname);
open(v);


while hasFrame(videoSource)
    %get frame
    frame = readFrame(videoSource);
    [inMat,imgInput] = util.createMat(frame);
    [outMat,outImg] = util.createMat();
    kNNBase.apply(imgInput,outImg);
    %get knn foreground mask
    foregroundmask = util.getImage(outImg);
    
    foregroundmask = rescale(foregroundmask);
    foregroundmask = cast(foregroundmask,"like",frame);

    foreground(:,:,1) = frame(:,:,1).*foregroundmask;
    foreground(:,:,2) = frame(:,:,2).*foregroundmask;
    foreground(:,:,3) = frame(:,:,3).*foregroundmask;
    

    g = rgb2gray(foreground);
    

    imshow(g);

    writeVideo(v,g);
    % hold off;
    pause(0.01);
end
close(v)
