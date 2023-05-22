%Background mixture models (Background Subtraction)
%Zoran Zivkovic and Ferdinand van der Heijden. Efficient adaptive density estimation per image pixel for the task of background subtraction. Pattern recognition letters, 27(7):773–780, 2006.


%MATLAB https://uk.mathworks.com/help/vision/ug/subtract-image-background-using-opencv-code-in-matlab.html


%history is the number of frames used to build the statistic model of the background. The smaller the value is, the faster changes in the background will be taken into account by the model and thus be considered as background. And vice versa.
%dist2Threshold is a threshold to define whether a pixel is different from the background or not. The smaller the value is, the more sensitive movement detection is. And vice versa.
%detectShadows : If set to true, shadows will be displayed in gray on the generated mask.

clear,clc,close all;
fg = {};
initDir = "VIDEO DIRECTORY";
data = initDir + "VIDEO FILE";
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

lineThresh = 400; %threshold for line
fishCount = 0;

%mask out part of frame
cents = [];%centroids of detections


[r,c] = size(readFrame(videoSource));

vidName = "outputs/multi_trackdata_61ac_KNN_Kalman_track";
fname = vidName + "_bg_subtraction";
v = VideoWriter(fname);
open(v);

tracks = initializeTracks();

% ID of the next track.
nextId = 1;

% Set the global parameters.
option.gatingThresh         = 0.9;              % A threshold to reject a candidate match between a detection and a track.
option.gatingCost           = 100;              % A large value for the assignment cost matrix that enforces the rejection of a candidate match.
option.costOfNonAssignment  = 10;               % A tuning parameter to control the likelihood of creation of a new track.
option.timeWindowSize       = 16;               % A tuning parameter to specify the number of frames required to stabilize the confidence score of a track.
option.ageThresh            = 1;                % A threshold to determine the minimum length required for a track being true positive.
option.visThresh            = 0.1;              % A threshold to determine the minimum visibility value for a track being true positive.


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
    %
    %imshow(frame); %to show output frame
    % hold on;
    bboxes = [];
    cents = [];
    %get detections
    for i=1: length(stat)
        cen = stat(i).Centroid;
        ar = stat(i).Area;
        bb = stat(i).BoundingBox;
        if cen(:,1) < 1400 &  cen(:,1) > 400 %if centroid of detection is not in "whiteboard" zone (reflections/waterflow). Change according to video
            if ar > 600 & ar < 100000 %fish are between these sizes
                cents = [cents; cen];
                bboxes = [bboxes; bb];
            end
        end
    end

    %predict new locations
    for i = 1:length(tracks)
            % Get the last bounding box on this track.
            bbox = tracks(i).bboxes(end, :);

            % Predict the current location of the track.
            predictedCentroid = predict(tracks(i).kalmanFilter);

            % Shift the bounding box so that its center is at the predicted location.
            tracks(i).predPosition = [predictedCentroid - bbox(3:4)/2, bbox(3:4)];
    end
    %DETECTION to track assignment

    % Compute the overlap ratio between the predicted boxes and the
    % detected boxes, and compute the cost of assigning each detection
    % to each track. The cost is minimum when the predicted bbox is
    % perfectly aligned with the detected bbox (overlap ratio is one)
    predBboxes = reshape([tracks(:).predPosition], 4, [])';
    cost = 1 - bboxOverlapRatio(predBboxes, bboxes);

    % Force the optimization step to ignore some matches by
    % setting the associated cost to be a large number. Note that this
    % number is different from the 'costOfNonAssignment' below.
    % This is useful when gating (removing unrealistic matches)
    % technique is applied.
    cost(cost > option.gatingThresh) = 1 + option.gatingCost;

    % Solve the assignment problem.
    [assignments, unassignedTracks, unassignedDetections] = ...
    assignDetectionsToTracks(cost, option.costOfNonAssignment);
    %update assigned tracks
    numAssignedTracks = size(assignments, 1);
    for i = 1:numAssignedTracks
        trackIdx = assignments(i, 1);
        detectionIdx = assignments(i, 2);
    
        centroid = cents(detectionIdx, :);
        bbox = bboxes(detectionIdx, :);
    
        % Correct the estimate of the object's location
        % using the new detection.
        correct(tracks(trackIdx).kalmanFilter, centroid);
    
        % Stabilize the bounding box by taking the average of the size
        % of recent (up to) 4 boxes on the track.
        T = min(size(tracks(trackIdx).bboxes,1), 4);
        w = mean([tracks(trackIdx).bboxes(end-T+1:end, 3); bbox(3)]);
        h = mean([tracks(trackIdx).bboxes(end-T+1:end, 4); bbox(4)]);
        tracks(trackIdx).bboxes(end+1, :) = [centroid - [w, h]/2, w, h];
    
        % Update track's age.
        tracks(trackIdx).age = tracks(trackIdx).age + 1;
    
    
        % Update visibility.
        tracks(trackIdx).totalVisibleCount = ...
            tracks(trackIdx).totalVisibleCount + 1;
    
    end

    %delete lost tracks (func)
    if ~isempty(tracks)

    

        % Compute the fraction of the track's age for which it was visible.
        ages = [tracks(:).age]';
        totalVisibleCounts = [tracks(:).totalVisibleCount]';
        visibility = totalVisibleCounts ./ ages;
    
    
        % Find the indices of 'lost' tracks.
        lostInds = (ages <= option.ageThresh & visibility <= option.visThresh);
    
        % Delete lost tracks.
        tracks = tracks(~lostInds);
    end

    %createNewTracks
    unassignedCentroids = cents(unassignedDetections, :);
    unassignedBboxes = bboxes(unassignedDetections, :);

    for i = 1:size(unassignedBboxes, 1)
        centroid = unassignedCentroids(i,:);
        bbox = unassignedBboxes(i, :);

        % Create a Kalman filter object.
        kalmanFilter = configureKalmanFilter('ConstantVelocity', ...
            centroid, [2, 1], [5, 5], 100);

        % Create a new track.
        newTrack = struct(...
            'id', nextId, ...
            'color', 255*rand(1,3), ...
            'bboxes', bbox, ...
            'kalmanFilter', kalmanFilter, ...
            'age', 1, ...
            'totalVisibleCount', 1, ...
            'predPosition', bbox);

        % Add it to the array of tracks.
        tracks(end + 1) = newTrack; %#ok<AGROW>

        % Increment the next id.
        nextId = nextId + 1;
    end
    
    %display boxes
    if ~isempty(tracks)
        ages = [tracks(:).age]';
        noDispInds = (ages < option.ageThresh) | ...
                   (ages < option.ageThresh / 2);
    
        for i = 1:length(tracks)
            if ~noDispInds(i)
    
                % scale bounding boxes for display
                bb = tracks(i).bboxes(end, :);
                % bb(:,1:2) = (bb(:,1:2)-1)*displayRatio + 1;
                % bb(:,3:4) = bb(:,3:4) * displayRatio;
    
    
                frame = insertShape(frame, ...
                                        'FilledRectangle', bb, ...
                                        'Color', tracks(i).color);
                % frame = insertObjectAnnotation(frame, ...
                %                         'rectangle', bb, ...
                %                         'Color', tracks(i).color);
            end
        end
    end
    imshow(frame);



    %line([0 c], [lineThresh lineThresh],Color='yellow');
    %text(20,20,int2str(fishCount),'Color','yellow','FontSize',30);
    writeVideo(v,frame);
    % hold off;
    pause(0.01);
end

close(v);

 function tracks = initializeTracks()
        % Create an empty array of tracks
        tracks = struct(...
            'id', {}, ...
            'color', {}, ...
            'bboxes', {}, ...
            'kalmanFilter', {}, ...
            'age', {}, ...
            'totalVisibleCount', {}, ...
            'predPosition', {});
 end

 

% end










