%Background mixture models (Background Subtraction)
%Zoran Zivkovic and Ferdinand van der Heijden. Efficient adaptive density estimation per image pixel for the task of background subtraction. Pattern recognition letters, 27(7):773–780, 2006.


%MATLAB https://uk.mathworks.com/help/vision/ug/subtract-image-background-using-opencv-code-in-matlab.html
%put these on github!!, and then try with multple fish in one shot.


%history is the number of frames used to build the statistic model of the background. The smaller the value is, the faster changes in the background will be taken into account by the model and thus be considered as background. And vice versa.
%dist2Threshold is a threshold to define whether a pixel is different from the background or not. The smaller the value is, the more sensitive movement detection is. And vice versa.
%detectShadows : If set to true, shadows will be displayed in gray on the generated mask.


%WORKING BASIC NAIVE KNN TRACK & Count ATTEMPT

clear,clc,close all;
fg = {};
initDir = "POOLED VIDEO FOLDER";
data = initDir + "VIDEO";
videoSource = VideoReader(data);

import clib.opencv.*;
import vision.opencv.*;
history = 40;
threshold = 1000;
shadow = true;

cvPtr = cv.createBackgroundSubtractorKNN(history,threshold,shadow);
kNNBase = util.getBasePtr(cvPtr);

passed = false;

kNNBase.setkNNSamples(6);% k nearest neighbours

foregroundmask = zeros(videoSource.Height,videoSource.Width,videoSource.NumFrames);

se = strel("disk", 15);

lineThresh = 315; %threshold for line
fishCount = 0;

%mask out part of frame
cents = [];%centroids of detections


[r,c] = size(readFrame(videoSource));

vidName = "outputs/multi_trackdata_61ac_KNN_no_track";
fname = vidName + "_bg_subtraction";
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
    
    %fg{end+1} = uint8(foreground);
    
    %binarize
    g = rgb2gray(foreground);
    f = medfilt2(g);
    bw = imbinarize(f);
    
    %close to join disconnected fish components
    clse = imclose(bw,se);
    
    [r,c] = size(clse);

    %get bounding box of connected comps over a certain area (fish)
    stat = regionprops(clse,'Area', 'BoundingBox','Centroid');
    [maxValue,index] = max([stat.Area]);

    %image(foreground,Parent=gca); %to show KNN
    imshow(frame); %to show output frame
    hold on;
    if maxValue %fish are between these sizes maxValue > 600 & maxValue < 100000
        cent = stat(index).Centroid;
        cents = [cents; [cent(:,1),cent(:,2)]];

        if cent(:,1) < 1700 %if centroid of detection is not in "whiteboard" zone (reflections/waterflow). Change according to video;
            %disp("fish");
            rectangle('Position', stat(index).BoundingBox, 'EdgeColor', 'r');
            plot(cent(:,1), cent(:,2),'xr');
            
                if fishCount == 0 & cent(:,2) < lineThresh
                    fishCount = fishCount + 1;
                end
            
        end
    end
    line([0 c], [lineThresh lineThresh],Color='yellow');
    writeVideo(v,getframe(gcf).cdata);
    hold off;
    pause(0.01);
end

close(v);

